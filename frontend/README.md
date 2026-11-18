# PathSnap Evaluator Interface (Laptop 2)

Evaluator-facing user interface for the PathSnap distributed two-laptop deployment.

## How to Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, enter Laptop 1's backend API URL (e.g., `http://192.168.1.15:8000`), select any of the 10 evaluation trips from the dropdown, and click **Reconstruct Route**.
