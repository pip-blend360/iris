Iris dataset analysis — principal data scientist take-home

This repository contains an analysis of the Iris dataset. It includes:

- Data loading and preprocessing
- Exploratory Data Analysis (XKCD style plots)
- Dimensionality reduction with UMAP
- Classification models and evaluation with a held-out test set
- An interactive 3D Plotly visualization comparing predicted vs. actual classes
- A generated static HTML report (report.html)

Structure
- data/: (generated during run) contains saved CSVs
- notebooks/: optional notebooks
- src/: source code
- outputs/: generated images and HTML report

To run:
1. Create a virtual environment and install requirements from requirements.txt
   pwsh: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
2. Run: python src/run_analysis.py

Files of interest
- src/run_analysis.py — main script
- src/report_template.html — Jinja2 template for HTML report
- outputs/report.html — generated report

