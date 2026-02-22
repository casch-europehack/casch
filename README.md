# CaSch

This project is a profiling and scheduling service for machine learning jobs. It analyzes Python scripts to estimate energy consumption, execution time, and carbon emissions, and allows scheduling jobs with specific power-throttling policies to optimize carbon footprint.

## Key Dependencies

This project relies on several important Python packages:

- **[FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)**: High-performance framework and server for building the REST API.
- **[PyTorch](https://pytorch.org/)**: Used for machine learning model execution and profiling.
- **[Zeus-ML](https://ml.energy/zeus/)**: A framework for deep learning energy measurement and optimization.
- **[NumPy](https://numpy.org/) & [SciPy](https://scipy.org/)**: Fundamental packages for scientific computing and data manipulation.
- **[Matplotlib](https://matplotlib.org/)**: Used for plotting and visualizing profiling data.
- **[Requests](https://requests.readthedocs.io/)**: For making HTTP requests to external services (e.g., Electricity Maps API).

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

Alternatively, in case you'd like to deploy the client and the server in different networks, execute:

```bash
./start.sh
```

This will start the server and create an [ngrok](https://ngrok.com/?homepage-cta-docs=test) tunneling channel.

## Example Jobs

There is an example job named `test_job.py` under the jobs directory.