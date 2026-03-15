# PharmaEase — Pharmacy Management System
**Internship Project | Brainybeam Info-Tech PVT LTD**
**Field: Data Science & Machine Learning with Data Analytics**

---

## Project Structure

```
PharmaEase/
├── data/
│   ├── generate_data.py     ← Run this first to create all CSVs
│   ├── medicines.csv
│   ├── sales.csv
│   ├── prescriptions.csv
│   ├── employees.csv
│   └── customers.csv
├── notebooks/
│   ├── phase2_eda.ipynb     ← Phase 2: Exploratory Data Analysis
│   └── phase3_ml.ipynb      ← Phase 3: ML Models
├── app/
│   └── main.py              ← Phase 4: Streamlit App
├── models/
│   └── (saved .pkl files)   ← Phase 3 output
└── requirements.txt
```

---

## Phase Plan

| Phase | Task | Deliverable |
|-------|------|-------------|
| 1 | Setup & data preparation | 5 CSV files + project folder |
| 2 | EDA in Jupyter | Notebook with insights & charts |
| 3 | ML models | Forecasting + stock-out prediction |
| 4 | Streamlit dashboard | 5-module working app |
| 5 | Report & presentation | Final report + live demo |

---

## Setup Instructions

### Step 1 — Install libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Generate datasets
```bash
cd data
python generate_data.py
```

### Step 3 — Open Jupyter for EDA
```bash
jupyter notebook notebooks/phase2_eda.ipynb
```

### Step 4 — Run Streamlit app
```bash
streamlit run app/main.py
```

---

## Tech Stack
- **Python** — primary language
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn / Plotly** — visualizations
- **Scikit-learn** — ML models
- **Streamlit** — dashboard UI
- **Jupyter Notebook** — EDA & experimentation
