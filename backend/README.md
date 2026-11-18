# Backend API

FastAPI service for loading the PathSnap LightGBM model, processing a selected student-test trip, decoding a route, and returning results to the evaluator interface.

## Requirements

- Python 3.10 or newer
- The supplied `Student folder/` at the repository root or its parent
- `backend/requirements.txt`
- `backend/lightgbm_model.joblib`, or locally generated training data in `shared_data_v2/` if the bundled model is absent

## Start the service

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The service listens on port 8000. Check `GET /health`; browse the interactive API schema at `/docs`. The evaluator interface calls `GET /trips` and `POST /predict_trip/{trip_id}`.

The shared preprocessing implementation is `PES2UG24CS064_LightGBM/src/prepare_high_acc_data.py`. The backend imports this implementation directly.
