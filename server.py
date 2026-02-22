import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from profiler.monitor import extrapolate, profile
from services.co2_service import co2_service
from services.storage import load_from_db, save_to_db
from utils.converters import PowerTranslator, aggregate_intervals

ELECTRICITYMAPS_TOKEN = "4N4rpoLvWqfvQ0yJdHN0"
ELECTRICITYMAPS_BASE = "https://api.electricitymaps.com/v3"

app = FastAPI(title="Profiler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/carbon-intensity")
async def carbon_intensity(zones: str = Query(..., description="Comma-separated zone codes")):
    """
    Fetch the latest carbon intensity (gCO₂eq/kWh) for each requested zone
    from the Electricity Maps API.

    Example: GET /carbon-intensity?zones=IE,PL,DE,FR
    """
    zone_list = [z.strip() for z in zones.split(",") if z.strip()]
    if not zone_list:
        raise HTTPException(status_code=400, detail="No zones provided.")

    headers = {"auth-token": ELECTRICITYMAPS_TOKEN}
    result: dict[str, Optional[int]] = {}

    for zone in zone_list:
        try:
            resp = requests.get(
                f"{ELECTRICITYMAPS_BASE}/carbon-intensity/latest",
                params={"zone": zone},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            result[zone] = round(data.get("carbonIntensity", 0))
        except Exception as e:
            print(f"Failed to fetch carbon intensity for {zone}: {e}")
            result[zone] = None

    return JSONResponse(
        content={"status": "success", "result": result},
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print("Got request...")
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only Python files are supported.")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    file_hash = hashlib.sha256(content).hexdigest()

    # Check if we already have a stored result
    existing_data = load_from_db(file_hash)
    if existing_data:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {
            "status": "success",
            "message": "Returned cached profiling results.",
            "file_hash": file_hash,
            "result": existing_data,
        }

    try:
        # --- Process ---

        # Run the profiler on the uploaded file
        out_stem = Path(file.filename).stem
        step_energies, step_times, steps_per_epoch, config = profile(temp_file_path)

        # Extrapolate
        output = extrapolate(
            step_energies, step_times, steps_per_epoch, config, out_stem
        )

        # --- Prepare Output ---

        # Aggregate intervals to reduce data size
        agg_energies, agg_times = aggregate_intervals(
            np.array(output["step_energy_J"]),
            np.array(output["step_time_s"]),
            num_blocks=200,
        )

        # Update output with aggregated data
        output["step_energy_J"] = agg_energies.tolist()
        output["step_time_s"] = agg_times.tolist()

        # Translate to power timeseries
        translator = PowerTranslator(output)
        power_timeseries = translator.to_timeseries(resolution_s=1.0)
        power_intervals = translator.get_intervals()

        output["power_timeseries"] = power_timeseries
        output["power_intervals"] = power_intervals

        # --- Store ---
        save_to_db(file_hash, output)

        return {
            "status": "success",
            "message": "Profiling completed successfully.",
            "file_hash": file_hash,
            "result": output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/co2")
async def get_co2(file_hash: str, location: str):
    print(file_hash, location)
    try:
        result = co2_service.get_co2_emissions(file_hash, location)
        return JSONResponse(
            content={"status": "success", "result": result},
            headers={"Cache-Control": "no-store"},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aggregate")
async def aggregate(file_hash: str, location: str):
    """
    Return the duration-vs-CO2 trade-off curve.

    For each optimisation step an additional pause of width ``eff_delta``
    is inserted, increasing wall-clock duration while (generally)
    reducing CO2.  The returned arrays let the frontend plot
    *duration* on the X-axis and *CO2 emissions* on the Y-axis,
    together with the ids of the policies that correspond to each point.
    """
    try:
        result = co2_service.get_aggregate(file_hash, location)
        return JSONResponse(
            content={"status": "success", "result": result},
            headers={"Cache-Control": "no-store"},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule")
async def schedule(
    file: UploadFile = File(...), location: str = Form(...), policy: str = Form(...)
):
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only Python files are supported.")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    file_hash = hashlib.sha256(content).hexdigest()
    cache_key = f"{file_hash}_{policy}"

    # Check if we already have a stored result
    existing_data = load_from_db(cache_key)
    if existing_data:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {
            "status": "success",
            "message": "Returned cached scheduling results.",
            "result": existing_data,
        }

    try:
        # Load the policy from the policies.json file
        policies_file = f"{file_hash}_policies.json"
        if not os.path.exists(policies_file):
            raise HTTPException(status_code=500, detail=f"{policies_file} not found.")

        with open(policies_file, "r") as f:
            policies = json.load(f)

        if policy not in policies:
            raise HTTPException(status_code=404, detail=f"Policy '{policy}' not found.")

        policy_data = policies[policy]["policy"]
        # The executor CLI expects a JSON string; if the policy is already
        # a parsed list/dict we need to serialize it back to a string.
        policy_str = (
            json.dumps(policy_data) if not isinstance(policy_data, str) else policy_data
        )

        out_stem = Path(file.filename).stem

        # Run the executor
        cmd = [
            "python",
            "-m",
            "executor.executor",
            temp_file_path,
            "--policy",
            policy_str,
            "--out",
            out_stem,
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"Executor failed: {process.stderr}"
            )

        # Read the output from the executor
        assets_dir = Path("profiler/assets")
        json_path = assets_dir / f"{out_stem}_execution.json"

        if not json_path.exists():
            raise HTTPException(
                status_code=500, detail="Executor did not produce output JSON."
            )

        with open(json_path, "r") as f:
            output = json.load(f)

        # Aggregate intervals to reduce data size
        agg_energies, agg_times = aggregate_intervals(
            np.array(output["step_energy_J"]),
            np.array(output["step_time_s"]),
            num_blocks=200,
        )

        # Update output with aggregated data
        output["step_energy_J"] = agg_energies.tolist()
        output["step_time_s"] = agg_times.tolist()

        # Translate to power timeseries
        translator = PowerTranslator(output)
        power_timeseries = translator.to_timeseries(resolution_s=1.0)
        power_intervals = translator.get_intervals()

        output["power_timeseries"] = power_timeseries
        output["power_intervals"] = power_intervals

        # --- Store ---
        save_to_db(cache_key, output)

        return {
            "status": "success",
            "message": "Scheduling completed successfully.",
            "result": output,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
