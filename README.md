What to install to run this pipeline


Install uv first (one‑time per machine).


# Pin the interpreter version to the repo
uv python pin 3.10.8            # writes a .python-version file


use 'uv venv' -----> to use python version 3.10.8 for this project






# 0) Install uv once (see above), then…

# 1) Clone the repo
git clone https://github.com/you/dissertation-project.git
cd dissertation-project



for the supervisor
pip install uv

uv venv && uv sync

uv run python run_pipeline.py

uv run zenml up (once) or  uv run zenml login --local --blocking --port 8237

To run the streamlit app use --> uv run streamlit run app/app.py



