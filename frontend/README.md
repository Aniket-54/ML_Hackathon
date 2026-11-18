# Evaluator Interface

Streamlit application for selecting student-test trips and displaying the observed GPS track, reference route when available, and decoded route on a Folium map.

## Start the interface

In a separate terminal from the backend, from the repository root:

```bash
python -m pip install -r frontend/requirements.txt
cd frontend
streamlit run app.py
```

Set the backend URL in the sidebar. Use `http://127.0.0.1:8000` for a same-machine run, or `http://<backend-host-ip>:8000` when the backend runs on another computer. The frontend host must be able to reach the backend on port 8000.
