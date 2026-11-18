# PES2UG24CS102 — Random Forest first-review bundle

This folder contains a reproducible preprocessing and Random Forest walkthrough:

- `01_first_review.ipynb` — run the workflow interactively, cell by cell or with **Run All**.
- `src/` — the Python scripts called by the notebook.
- `requirements.txt` — Python packages needed to run the scripts/notebook.
- `FIRST_REVIEW.md` — beginner-first explanation, preprocessing counts, file guide, and measured Random Forest results.

## Raw data location

The raw trip data is not copied into this folder or committed to the repository. Place/extract the supplied `Student folder` at the repository root, beside this `PES2UG24CS102_RandomForest` folder. It should contain `train/` and `student_test/`, each with the trip files. The preprocessing code reads from that location.

Expected layout:

```text
ML_Hackathon/
├── Student folder/
│   ├── train/
│   └── student_test/
└── PES2UG24CS102_RandomForest/
    ├── 01_first_review.ipynb
    ├── src/
    └── requirements.txt
```

## Run it

From the repository root, create/activate a Python environment and install the listed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r PES2UG24CS102_RandomForest/requirements.txt
python -m ipykernel install --user --name pathsnap-review --display-name "PathSnap review"
```

Open `PES2UG24CS102_RandomForest/01_first_review.ipynb` in Jupyter, select the **PathSnap review** kernel, and choose **Run All**. The notebook invokes the scripts from this folder using the selected kernel's Python interpreter. Generated datasets, results, model, and map files are written under this folder when the relevant cells run.

The notebook and scripts are portable; they do not require the original author's absolute laptop path. Each teammate can independently regenerate the same processed rows when using the same raw files and code/settings.
