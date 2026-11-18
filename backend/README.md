# PathSnap Backend API Server (Laptop 1)

This folder contains the FastAPI backend for the PathSnap distributed two-laptop deployment.

## How to Run:
```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

When started, it listens on port 8000 and serves route prediction and map-matching requests from Laptop 2.
