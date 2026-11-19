from urllib.error import URLError

import pandas as pd
import streamlit as st
import altair as alt

TITANIC_DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
FEATURE_COLUMNS = ["survived", "pclass", "sex", "age", "fare", "embarked"]
AGE_GROUPS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
EMBARKED_CODES = {"C": 0, "Q": 1, "S": 2}
CLASS_LABELS = {1: "First", 2: "Second", 3: "Third"}
CLASS_ORDER = ["First", "Second", "Third"]
CLASS_COLORS = ["#c65a46", "#248277", "#d39a34"]
SEX_LABELS = {"female": "Female", "male": "Male"}
SEX_ORDER = ["Female", "Male"]
SEX_COLORS = ["#248277", "#c65a46"]
OUTCOME_LABELS = {0: "Did not survive", 1: "Survived"}
OUTCOME_ORDER = ["Did not survive", "Survived"]
OUTCOME_COLORS = ["#c65a46", "#248277"]


@st.cache_data
def load_titanic_data() -> pd.DataFrame:
    return pd.read_csv(TITANIC_DATA_URL)


def min_max_normalize(values: pd.Series) -> pd.Series:
    value_range = values.max() - values.min()
    if value_range == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / value_range


def prepare_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    analysis_data = raw_data.loc[:, FEATURE_COLUMNS].copy()
    missing_before = analysis_data.isna().sum()

    analysis_data["age"] = analysis_data["age"].fillna(analysis_data["age"].median())
    analysis_data["embarked"] = analysis_data["embarked"].fillna(
        analysis_data["embarked"].mode().iloc[0]
    )
    missing_after = analysis_data.isna().sum()

    analysis_data["age_norm"] = min_max_normalize(analysis_data["age"])
    analysis_data["fare_norm"] = min_max_normalize(analysis_data["fare"])
    analysis_data["age_group"] = pd.cut(
        analysis_data["age"],
        bins=[float("-inf"), 12, 18, 35, 60, float("inf")],
        labels=AGE_GROUPS,
    )
    encoded_data = analysis_data.copy()
    encoded_data["sex_female"] = analysis_data["sex"].eq("female").astype(int)
    encoded_data["embarked_code"] = analysis_data["embarked"].map(EMBARKED_CODES).astype(int)
    return analysis_data, encoded_data, missing_before, missing_after


def survival_rate_chart(
    data: pd.DataFrame,
    dimension: str,
    title: str,
    order: list[str],
    colors: list[str],
    labels: dict | None = None,
) -> alt.Chart:
    rates = (
        data.groupby(dimension, observed=True)["survived"]
        .mean()
        .mul(100)
        .rename("Survival rate (%)")
        .rename_axis(dimension)
        .reset_index()
    )
    if labels:
        rates[dimension] = rates[dimension].map(labels)

    return (
        alt.Chart(rates)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{dimension}:N", title=title, sort=order),
            y=alt.Y(
                "Survival rate (%):Q",
                title="Survival rate (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                f"{dimension}:N",
                scale=alt.Scale(domain=order, range=colors),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{dimension}:N", title=title),
                alt.Tooltip("Survival rate (%):Q", format=".1f"),
            ],
        )
        .properties(title=title, height=260)
    )


def category_rate_series(data: pd.DataFrame, dimension: str, labels: dict | None = None) -> pd.Series:
    rates = data.groupby(dimension, observed=True)["survived"].mean().mul(100)
    if labels:
        rates.index = rates.index.map(labels)
    return rates


st.set_page_config(page_title="Titanic Passenger Insights", page_icon="🚢", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #d9e1e5; }
    [data-testid="stMetric"] { background: #f3f6f5; padding: 1rem; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups.")

try:
    raw_data = load_titanic_data()
except (OSError, pd.errors.ParserError, URLError) as error:
    st.error(f"The Titanic dataset could not be loaded: {error}")
    st.stop()

analysis_data, encoded_data, missing_before, missing_after = prepare_data(raw_data)
st.sidebar.header("Passenger filters")
selected_classes = st.sidebar.multiselect(
    "Passenger class",
    options=sorted(analysis_data["pclass"].unique().tolist()),
    default=sorted(analysis_data["pclass"].unique().tolist()),
)
selected_sexes = st.sidebar.multiselect(
    "Sex",
    options=sorted(analysis_data["sex"].unique().tolist()),
    default=sorted(analysis_data["sex"].unique().tolist()),
)
selected_age_groups = st.sidebar.multiselect(
    "Age group",
    options=AGE_GROUPS,
    default=AGE_GROUPS,
)

filtered_data = analysis_data.loc[
    analysis_data["pclass"].isin(selected_classes)
    & analysis_data["sex"].isin(selected_sexes)
    & analysis_data["age_group"].isin(selected_age_groups)
]
overview_tab, explore_tab, preparation_tab, data_tab = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

with overview_tab:
    st.subheader("Passenger and survival summary")
    if filtered_data.empty:
        st.info("No passengers match the selected filters.")
    else:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Passengers in selection", f"{len(filtered_data):,}")
        metric_columns[1].metric("Survivors", f"{int(filtered_data['survived'].sum()):,}")
        metric_columns[2].metric("Survival rate", f"{filtered_data['survived'].mean() * 100:.1f}%")
        metric_columns[3].metric("Average fare", f"{filtered_data['fare'].mean():.2f}")

        outcome_counts = (
            filtered_data["survived"]
            .map(OUTCOME_LABELS)
            .value_counts()
            .reindex(OUTCOME_ORDER, fill_value=0)
            .rename_axis("Outcome")
            .rename("Passengers")
            .reset_index()
        )
        overview_chart_columns = st.columns(2)
        with overview_chart_columns[0]:
            st.altair_chart(
                alt.Chart(outcome_counts)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Outcome:N", sort=OUTCOME_ORDER, title="Outcome"),
                    y=alt.Y("Passengers:Q", title="Passengers"),
                    color=alt.Color(
                        "Outcome:N",
                        scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                        legend=None,
                    ),
                    tooltip=["Outcome:N", "Passengers:Q"],
                )
                .properties(title="Survival counts", height=300),
                width="stretch",
            )
        with overview_chart_columns[1]:
            st.altair_chart(
                survival_rate_chart(
                    filtered_data,
                    "pclass",
                    "Passenger class",
                    CLASS_ORDER,
                    CLASS_COLORS,
                    CLASS_LABELS,
                ),
                width="stretch",
            )
        st.caption("Observed rates describe this dataset only; they do not establish cause.")

with explore_tab:
    st.subheader("Compare passenger groups")
    if filtered_data.empty:
        st.info("No passengers match the selected filters.")
    else:
        first_row = st.columns(2)
        with first_row[0]:
            st.altair_chart(
                survival_rate_chart(
                    filtered_data, "pclass", "Passenger class", CLASS_ORDER, CLASS_COLORS, CLASS_LABELS
                ),
                width="stretch",
            )
        with first_row[1]:
            st.altair_chart(
                survival_rate_chart(
                    filtered_data, "sex", "Sex", SEX_ORDER, SEX_COLORS, SEX_LABELS
                ),
                width="stretch",
            )

        second_row = st.columns(2)
        with second_row[0]:
            age_distribution = filtered_data.assign(
                Outcome=filtered_data["survived"].map(OUTCOME_LABELS)
            )
            st.altair_chart(
                alt.Chart(age_distribution)
                .mark_bar(opacity=0.85)
                .encode(
                    x=alt.X("age:Q", bin=alt.Bin(maxbins=22), title="Age (years)"),
                    y=alt.Y("count():Q", title="Passengers"),
                    color=alt.Color(
                        "Outcome:N",
                        scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                        title="Outcome",
                    ),
                    tooltip=[alt.Tooltip("count():Q", title="Passengers")],
                )
                .properties(title="Age distribution by outcome", height=280),
                width="stretch",
            )
        with second_row[1]:
            fare_by_class = (
                filtered_data.groupby("pclass", observed=True)["fare"]
                .median()
                .rename("Median fare")
                .rename_axis("pclass")
                .reset_index()
            )
            fare_by_class["Passenger class"] = fare_by_class["pclass"].map(CLASS_LABELS)
            st.altair_chart(
                alt.Chart(fare_by_class)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Passenger class:N", sort=CLASS_ORDER, title="Passenger class"),
                    y=alt.Y("Median fare:Q", title="Median fare"),
                    color=alt.Color(
                        "Passenger class:N",
                        scale=alt.Scale(domain=CLASS_ORDER, range=CLASS_COLORS),
                        legend=None,
                    ),
                    tooltip=["Passenger class:N", alt.Tooltip("Median fare:Q", format=".2f")],
                )
                .properties(title="Median fare by class", height=280),
                width="stretch",
            )

        st.subheader("Finding")
        comparison_series = {
            "passenger class": category_rate_series(filtered_data, "pclass", CLASS_LABELS),
            "sex": category_rate_series(filtered_data, "sex", SEX_LABELS),
            "age group": category_rate_series(filtered_data, "age_group"),
        }
        comparisons = {
            dimension: rates.dropna().sort_values()
            for dimension, rates in comparison_series.items()
            if rates.dropna().size > 1
        }
        if comparisons:
            dimension, rates = max(
                comparisons.items(),
                key=lambda item: item[1].iloc[-1] - item[1].iloc[0],
            )
            st.info(
                f"In this selection, {rates.index[-1]} passengers had a higher observed survival rate "
                f"than {rates.index[0]} passengers ({rates.iloc[-1]:.1f}% vs. {rates.iloc[0]:.1f}%)."
            )
        else:
            st.info("The current filters leave only one group for comparison.")

with preparation_tab:
    with st.expander("Dataset inspection", expanded=False):
        st.write(f"Dataset shape: **{raw_data.shape[0]:,} rows x {raw_data.shape[1]} columns**")
        st.write("Column names: " + ", ".join(raw_data.columns))
        column_details = pd.DataFrame(
            {
                "Column": raw_data.columns,
                "Data type": raw_data.dtypes.astype(str).to_numpy(),
                "Missing values": raw_data.isna().sum().to_numpy(),
            }
        )
        st.dataframe(column_details, hide_index=True, width="stretch")
        st.write("First five rows")
        st.dataframe(raw_data.iloc[:5, :], hide_index=True, width="stretch")

    with st.expander("1. Missing values", expanded=True):
        st.caption(
            f"Age is imputed with its median ({raw_data['age'].median():.1f}) and embarked "
            f"with its most frequent category ({raw_data['embarked'].mode().iloc[0]}); "
            "deck is excluded because it has many missing values."
        )
        missing_comparison = pd.concat(
            [missing_before.rename("Before"), missing_after.rename("After")],
            axis=1,
        ).rename_axis("Analysis field")
        st.dataframe(
            missing_comparison.loc[["age", "embarked"]],
            width="stretch",
        )

    with st.expander("2. Min-Max normalization", expanded=False):
        st.caption(
            "Age and fare are scaled to the 0-1 range; their original columns remain unchanged."
        )
        st.dataframe(
            analysis_data.loc[:, ["age", "age_norm", "fare", "fare_norm"]].head(5),
            hide_index=True,
            width="stretch",
        )

    with st.expander("3. Age discretization", expanded=False):
        st.caption(
            "The imputed age value is mapped to Child, Teen, Young Adult, Adult, or Older Adult."
        )
        st.dataframe(
            analysis_data.loc[:, ["age", "age_group"]].head(5),
            hide_index=True,
            width="stretch",
        )

    with st.expander("4. Dummy coding", expanded=False):
        st.caption(
            "Sex uses one binary dummy (female=1, male=0). Embarked has three categories, "
            "so its single column uses nominal codes (C=0, Q=1, S=2), not a dummy variable. "
            "Survived is not encoded."
        )
        encoded_preview = pd.concat(
            [
                analysis_data.loc[:, ["sex", "embarked"]].head(5).add_prefix("Original: "),
                encoded_data.loc[:, ["sex_female", "embarked_code"]]
                .head(5)
                .add_prefix("Encoded: "),
            ],
            axis=1,
        )
        st.dataframe(encoded_preview, hide_index=True, width="stretch")

with data_tab:
    st.subheader("Passenger records")
    st.caption("Select a data stage and fields to inspect; shared sidebar filters apply to every view.")
    data_view = st.selectbox("View", ["Raw", "Cleaned", "Transformed"])
    data_views = {
        "Raw": raw_data.loc[filtered_data.index],
        "Cleaned": analysis_data.loc[filtered_data.index, FEATURE_COLUMNS],
        "Transformed": encoded_data.loc[filtered_data.index],
    }
    view_data = data_views[data_view]
    if data_view == "Raw":
        default_columns = FEATURE_COLUMNS
    elif data_view == "Cleaned":
        default_columns = FEATURE_COLUMNS
    else:
        default_columns = [
            "survived",
            "pclass",
            "age",
            "age_norm",
            "age_group",
            "fare",
            "fare_norm",
            "sex_female",
            "embarked_code",
        ]
        default_columns = [column for column in default_columns if column in view_data.columns]

    count_column, column_picker = st.columns([1, 3])
    count_column.metric("Rows after filters", f"{len(view_data):,}")
    selected_columns = column_picker.multiselect(
        "Columns to display",
        options=view_data.columns.tolist(),
        default=default_columns,
    )
    if view_data.empty:
        st.info("No passenger records match the selected filters.")
    elif not selected_columns:
        st.info("Select at least one column to display records.")
    else:
        row_limit = st.slider(
            "Sample rows",
            min_value=1,
            max_value=min(200, len(view_data)),
            value=min(50, len(view_data)),
        )
        st.caption(f"Showing {min(row_limit, len(view_data)):,} of {len(view_data):,} filtered passengers.")
        st.dataframe(
            view_data.loc[:, selected_columns].head(row_limit),
            hide_index=True,
            height=520,
            width="stretch",
        )
        st.download_button(
            "Download filtered data",
            data=view_data.loc[:, selected_columns].to_csv(index=False).encode("utf-8"),
            file_name=f"titanic_{data_view.lower()}_filtered.csv",
            mime="text/csv",
        )
