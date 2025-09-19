What to install to run this pipeline


# Pin the interpreter version to the repo
uv python pin 3.10.8            # writes a .python-version file


use 'uv venv' -----> to use python version 3.10.8 for this project

# Clone the repo
git clone https://github.com/Bradsmars/uof-risk-app/tree/main
cd dissertation-project

for the supervisor
pip install uv

uv venv && uv sync

uv run python run_pipeline.py

uv run zenml up (once) or  uv run zenml login --local --blocking --port 8237

To run the streamlit app use --> uv run streamlit run app/app.py


# In PowerShell, inside the project folder
git clone https://github.com/Bradsmars/uof-risk-app/tree/main
irm https://astral.sh/uv/install.ps1 | iex     # install uv (one time)
uv --version                                   # sanity check
uv sync                                        # install project deps
# Ensure data files exist in .\data\ (2022/2023/2024 .xlsx)

uv run python run_pipeline.py                  # run the pipeline
uv run streamlit run app/app.py                # launch the app




