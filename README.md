# Profiler

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
python monitor.py <path_to_user_job.py>
```

## Running the Server

To start the FastAPI server, run:

```bash
python server.py
```
The server will be available at `http://0.0.0.0:8000`.

## API

### `POST /analyze`

Upload a Python file to be profiled.

**Request:**
- `file`: The Python file to profile (multipart/form-data)

**Response:**
- `status`: "success" or "error"
- `message`: Status message
- `file_hash`: SHA-256 hash of the uploaded file
- `result`: Profiling results including energy and time estimates

### `POST /schedule`

Schedule a training job with a specific throttle policy.

**Request:**
- `file`: The Python file to execute (multipart/form-data)
- `location`: The location for the execution (form data)
- `policy`: The ID of the policy to use from `policies.json` (form data)

**Response:**
- `status`: "success" or "error"
- `message`: Status message
- `result`: Execution data including time, throttle, GPU utilization, and power draw

### `GET /co2`

Get CO2 emissions estimates for a previously profiled file.

**Request:**
- `file_hash`: The SHA-256 hash returned from the `/analyze` endpoint
- `location`: The location for which to calculate emissions

**Response:**
- `status`: "success" or "error"
- `result`: CO2 emissions data including:
  - `emissions_per_hour`: Array of hourly CO2 emission values
  - `times`: Array of timestamps corresponding to the measurements
  - `total_emissions`: Total CO2 emissions
  - `location`: The requested location
