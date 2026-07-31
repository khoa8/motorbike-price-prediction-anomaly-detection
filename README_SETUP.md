# Project 2 Streamlit Scaffold

## 1. Copy deployment bundle

Copy the contents of the Colab ZIP into this project:

- `artifacts/*` -> `artifacts/`
- `reports/*` -> `reports/`
- `examples/*` -> `examples/`

Do not commit the raw `data_motobikes.xlsx` dataset.

## 2. Create the environment on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python generate_requirements.py
python -m pip install -r requirements.txt
```

## 3. Validate files

```bash
python -m compileall app.py src
python check_project.py
```

## 4. Run

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Expected batch-template behavior

- Missing `km`: valid; the pipeline imputes it.
- Missing `capacity`: invalid row.
- Year `1979`: valid but should be presented as an old/rare case.
