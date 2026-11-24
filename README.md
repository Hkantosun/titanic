# Titanic Passenger Insights

Titanic Passenger Insights is an interactive Streamlit dashboard for exploring survival patterns in the Titanic passenger dataset and understanding common data-preparation techniques.

Users can filter passengers, compare demographic groups, inspect the data at different stages of transformation, and download filtered records.

## Features

- Passenger filtering by class, sex, and age group
- Passenger count, survivor count, survival rate, and average fare metrics
- Survival charts for passenger class and sex
- Age-distribution and median-fare visualizations
- Automatic identification of the largest observed survival-rate difference
- Missing-value, normalization, age discretization, and encoding previews
- Raw, cleaned, and transformed data views
- CSV export of filtered passenger records

## Stack

- **Language:** Python 3.14
- **Application framework:** Streamlit
- **Data processing:** Pandas
- **Visualization:** Altair
- **Dependency management:** uv
- **Hosting:** Streamlit Community Cloud

## Project structure

```text
app.py                 Streamlit entry point
streamlit_app.py       Dashboard, preprocessing, charts, and data views
pyproject.toml         Python and uv project configuration
requirements.txt       Hosted deployment dependencies
uv.lock                Locked dependency versions
.python-version        Python version selection
.devcontainer/         Python 3.14 development container configuration
.github/               Dependabot and repository ownership configuration
LICENSE                Apache License 2.0
```

The `app.py` entry point executes `streamlit_app.py`, keeping the public Streamlit main-file setting simple while preserving the main application in a separate module.

## How it works

`streamlit_app.py` downloads the Titanic dataset from Seaborn's public GitHub dataset source and caches it with `@st.cache_data`. The `prepare_data()` function then:

1. Selects the fields used by the analysis.
2. Fills missing ages with the median age.
3. Fills missing embarkation values with the most frequent port.
4. Scales age and fare to a 0–1 range.
5. Divides ages into Child, Teen, Young Adult, Adult, and Older Adult groups.
6. Creates binary sex and numeric embarkation encoding columns.

The dashboard uses four tabs:

- **Overview** — Summary metrics and overall outcomes
- **Explore** — Passenger-group comparisons and finding summary
- **Preprocessing** — Explanations and previews of data transformations
- **Data Explorer** — Raw, cleaned, and transformed records

## Run locally

### Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/) installed
- Internet access for downloading the Titanic dataset

### Start the app

```bash
git clone https://github.com/Hkantosun/titanic.git
cd titanic
uv sync
uv run streamlit run app.py
```

Open `http://localhost:8501` in a browser.

## Run in a development container

The repository includes a Python 3.14 development-container configuration for GitHub Codespaces or VS Code Dev Containers. It forwards Streamlit's default port `8501` and starts the application automatically when attached.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app in [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select this repository and the `main` branch.
4. Set the main file to `app.py`.
5. Deploy the app and use the generated `*.streamlit.app` URL.

No API keys or environment variables are required. Hosted deployment dependencies are listed in `requirements.txt`.

## Data source

The application loads the Titanic dataset from:

<https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv>

The summary statistics and charts describe this dataset only. They do not establish causal relationships between passenger characteristics and survival.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
