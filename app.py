import re
from datetime import time
from io import BytesIO

import pandas as pd
import streamlit as st


st.set_page_config(page_title="TV Integration Valuator", page_icon="📺", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    entered_password = st.text_input("Password", type="password")
    if entered_password == st.secrets["PASSWORD"]:
        st.session_state.authenticated = True
        st.rerun()
    st.error("Incorrect password.")
    st.stop()


SUMMARY_MARKERS = ("total", "summary", "subtotal", "grand total")
HEADER_KEYWORDS = (
    "episode",
    "tx",
    "duration",
    "seconds",
    "date",
    "match",
    "brand",
    "rate",
)
METADATA_COLUMNS = {
    "TX",
    "EPISODE",
    "DURATION",
    "SECONDS",
    "RATE",
    "VALUE",
    "NOTES",
    "INTEGRATION POINT",
    "DESCRIPTION",
    "WEEK",
    "SEG/TIME",
}
IGNORED_HEADER_VALUES = {"GUARANTEED", "DELIVERED", "DIFFERENCE"}
SPORT_GENERIC_HEADERS = {
    "DAY", "NIGHT", "NINE", "GEM", "GEM/GO", "PLAN", "DELIVERED",
    "SECONDS", "DURATION", "DATE", "TIME", "TX", "EPISODE", "WEEK",
}


def _normalise_header(value: object) -> str:
    """Return a comparable column name while preserving the original data."""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _find_header_row(raw_data: pd.DataFrame) -> int:
    """Find the strongest header candidate among the first 25 rows."""
    best_row = None
    best_score = 0
    for row_number, row in raw_data.iloc[:25].iterrows():
        row_text = row.fillna("").astype(str).str.lower()
        score = sum(
            row_text.str.contains(keyword, regex=False, na=False).any()
            for keyword in HEADER_KEYWORDS
        )
        if score > best_score:
            best_row = row_number
            best_score = score
    if best_row is None:
        raise ValueError("Could not identify a header row in the first 25 rows.")
    return best_row


def _find_column(columns: list[str], markers: tuple[str, ...]) -> str | None:
    for column in columns:
        normalised_column = _normalise_header(column)
        if any(marker.upper() in normalised_column for marker in markers):
            return column
    return None


def _is_duration_column(value: object) -> bool:
    text = str(value).lower()
    return "second" in text or "duration" in text


def _parse_seconds(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second) + (
            value.microsecond / 1_000_000
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return 0.0
    if text.endswith("s"):
        try:
            return float(text[:-1].strip())
        except ValueError:
            return 0.0
    if ":" in text:
        try:
            parts = [float(part.strip()) for part in text.split(":")]
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60 + seconds
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _deduplicate_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for header in headers:
        suffix = counts.get(header, 0)
        result.append(header if suffix == 0 else f"{header}.{suffix}")
        counts[header] = suffix + 1
    return result


def _is_missing_header(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed"):
        return True
    if text.upper() in IGNORED_HEADER_VALUES:
        return True
    numeric_value = pd.to_numeric(text.replace(",", ""), errors="coerce")
    return pd.notna(numeric_value)


def _integration_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if _normalise_header(column) not in METADATA_COLUMNS
        and _normalise_header(column) not in {"TIME", "DATE"}
        and not _normalise_header(column).startswith(("COLUMN", "UNNAMED"))
        and column != "Duration"
    ]


def _has_integration_entry(values: pd.Series) -> pd.Series:
    text_values = values.where(values.notna(), "").astype(str).str.strip()
    numeric_values = pd.to_numeric(text_values, errors="coerce")
    return text_values.ne("") & (numeric_values.isna() | numeric_values.gt(0))


def _load_integrations(
    uploaded_file, sheet_name: str, tracker_format: str
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    raw_data = pd.read_excel(
        BytesIO(uploaded_file.getvalue()), sheet_name=sheet_name, header=None
    )
    header_row = _find_header_row(raw_data)

    headers = []
    for column_index, value in enumerate(raw_data.iloc[header_row].tolist()):
        header_value = value
        if _is_missing_header(header_value):
            for row_index in range(header_row - 1, -1, -1):
                candidate = raw_data.iat[row_index, column_index]
                if not _is_missing_header(candidate):
                    header_value = candidate
                    break
        headers.append(
            str(header_value).strip()
            if not _is_missing_header(header_value)
            else f"Column {column_index + 1}"
        )
    data = raw_data.iloc[header_row + 1 :].copy()
    headers = _deduplicate_headers(headers)
    data.columns = headers
    data = data.dropna(how="all").dropna(axis=1, how="all")
    summary_start_mask = (
        data.iloc[:, :3]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.contains(r"grand\s+total|totals", case=False, regex=True, na=False)
    )
    summary_positions = summary_start_mask.to_numpy().nonzero()[0]
    if len(summary_positions):
        data = data.iloc[: summary_positions[0]].copy()
    row_labels = (
        data.iloc[:, :2]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    labeled_rows = row_labels.str.contains(
        r"\btotal\b|subtotal|pre\s+tournaments|\bweek\b",
        case=False,
        regex=True,
        na=False,
    )
    data = data.loc[~labeled_rows].copy()

    sport_groups: dict[str, dict[str, str]] = {}
    if tracker_format == "Sport (Multi-Duration)" and header_row > 0:
        names = None
        for row_index in range(
            header_row - 1, max(-1, header_row - 4), -1
        ):
            candidate_row = raw_data.iloc[row_index]
            valid_names = candidate_row[
                candidate_row.map(
                    lambda value: isinstance(value, str)
                    and bool(value.strip())
                    and value.strip().upper() not in SPORT_GENERIC_HEADERS
                )
            ]
            if not valid_names.empty:
                names = candidate_row.ffill()
                break
        if names is None:
            names = raw_data.iloc[header_row - 1].ffill()
        anchors = []
        for index, value in enumerate(names):
            text = str(value).strip()
            if (
                text
                and text.upper() not in SPORT_GENERIC_HEADERS
                and not _is_duration_column(text)
                and any(char.isalpha() for char in text)
            ):
                if not anchors or anchors[-1][1] != text:
                    anchors.append((index, text))
        for anchor_index, (start, name) in enumerate(anchors):
            end = anchors[anchor_index + 1][0] if anchor_index + 1 < len(anchors) else len(headers)
            group = sport_groups.setdefault(name, {})
            for index in range(start, end):
                header_text = str(raw_data.iat[header_row, index]).lower()
                if "second" in header_text or "duration" in header_text:
                    group["seconds"] = headers[index]
                elif "delivered" in header_text:
                    group["delivered"] = headers[index]

    duration_columns = [column for column in data.columns if _is_duration_column(column)]
    if tracker_format == "Sport (Multi-Duration)" and sport_groups:
        duration_columns = [
            group["seconds"] for group in sport_groups.values() if "seconds" in group
        ]
    duration_column = duration_columns[0] if duration_columns else None
    if duration_column is None:
        raise ValueError("The detected header row does not contain a duration column.")

    if tracker_format == "Sport (Multi-Duration)" and sport_groups:
        duration_values = data[duration_columns].apply(pd.to_numeric, errors="coerce").where(
            lambda values: values <= 5000, 0
        ).fillna(0).sum(axis=1)
    else:
        duration_text = data[duration_column].where(data[duration_column].notna(), "").astype(str).str.strip()
        data = data.loc[duration_text.ne("")].copy()
        duration_values = pd.to_numeric(
            duration_text.loc[data.index].str.replace(r"[^\d.+-]", "", regex=True),
            errors="coerce",
        )
    row_text = data.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    summary_rows = row_text.str.contains("|".join(map(re.escape, SUMMARY_MARKERS)), regex=True)

    data = data.loc[duration_values.notna() & ~summary_rows].copy()
    data["Duration"] = duration_values.loc[data.index]
    data = data.loc[data["Duration"] > 0].copy()
    if data.empty:
        raise ValueError("No integration rows with a positive duration were found.")

    return data, sport_groups


st.title("TV Integration Valuator")
st.caption("Upload a tracking sheet to estimate the value of your TV integrations.")

with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload Excel tracking sheet", type=["xlsx"])
    sheet_name = None
    if uploaded_file is not None:
        xls = pd.ExcelFile(BytesIO(uploaded_file.getvalue()))
        sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
    tracker_format = st.radio(
        "Tracker Format",
        ["Entertainment (Single Duration)", "Sport (Multi-Duration)"],
    )
    client_rate = st.number_input(
        "30s Client Rate",
        min_value=0.0,
        value=1824.50,
        step=0.01,
        format="%.2f",
    )
    compare_market = st.checkbox("Compare to Market Rate", value=False)
    market_rate = None
    if compare_market:
        market_rate = st.number_input(
            "30s Market Rate",
            min_value=0.0,
            value=65000.00,
            step=0.01,
            format="%.2f",
        )

if uploaded_file is None:
    st.info("Upload an .xlsx tracking sheet from the sidebar to begin.")
    st.stop()

try:
    integrations, sport_groups = _load_integrations(
        uploaded_file, sheet_name, tracker_format
    )
except (ValueError, ImportError, OSError) as error:
    st.error(str(error))
    st.stop()

available_integrations = [
    column
    for column in (
        list(sport_groups)
        if tracker_format == "Sport (Multi-Duration)"
        else _integration_columns(integrations)
    )
    if str(column).strip()
    and str(column).strip().lower() != "nan"
    and not str(column).strip().lower().startswith(("unnamed", "column"))
]
selected_integrations = available_integrations

integrations["Rate"] = client_rate
integrations["Client Value"] = (client_rate / 30) * integrations["Duration"] * 2
total_client_value = integrations["Client Value"].sum()

if compare_market:
    integrations["Market Value"] = (market_rate / 30) * integrations["Duration"] * 2
    total_market_value = integrations["Market Value"].sum()
    client_metric, market_metric = st.columns(2)
    client_metric.metric("Total Client Value", f"${total_client_value:,.2f}")
    market_metric.metric("Total Market Value", f"${total_market_value:,.2f}")
else:
    st.metric("Total Client Value", f"${total_client_value:,.2f}")

st.subheader("Value by Integration")
selected_integrations = [
    integration
    for integration in selected_integrations
    if tracker_format != "Sport (Multi-Duration)"
    or (
        integration in sport_groups
        and "seconds" in sport_groups[integration]
    )
]
integration_totals = {
    integration: {
        "Amount Delivered": (
            pd.to_numeric(integrations[sport_groups[integration]["delivered"]], errors="coerce")
            .fillna(0).sum()
            if tracker_format == "Sport (Multi-Duration)"
            and integration in sport_groups
            and "delivered" in sport_groups[integration]
            else int(entry_mask.sum())
        ),
        "Total Seconds": (
            pd.to_numeric(integrations[sport_groups[integration]["seconds"]], errors="coerce")
            .where(lambda values: values <= 5000, 0).fillna(0).sum()
            if tracker_format == "Sport (Multi-Duration)" and integration in sport_groups
            else integrations.loc[entry_mask, "Duration"].sum()
        ),
        "Total Client Value": (
            (client_rate / 30)
            * pd.to_numeric(integrations[sport_groups[integration]["seconds"]], errors="coerce")
            .where(lambda values: values <= 5000, 0).fillna(0).sum()
            * 2
            if tracker_format == "Sport (Multi-Duration)" and integration in sport_groups
            else integrations.loc[entry_mask, "Client Value"].sum()
        ),
        **(
            {
                "Total Market Value": (
                    (market_rate / 30)
                    * pd.to_numeric(integrations[sport_groups[integration]["seconds"]], errors="coerce")
                    .where(lambda values: values <= 5000, 0).fillna(0).sum()
                    * 2
                    if tracker_format == "Sport (Multi-Duration)" and integration in sport_groups
                    else integrations.loc[entry_mask, "Market Value"].sum()
                )
            }
            if compare_market
            else {}
        ),
    }
    for integration in selected_integrations
    for entry_mask in [
        _has_integration_entry(
            integrations[
                sport_groups[integration]["seconds"]
                if tracker_format == "Sport (Multi-Duration)" and integration in sport_groups
                else integration
            ]
        )
    ]
}
integration_summary = pd.DataFrame(
    [{"Integration": integration, **totals} for integration, totals in integration_totals.items()]
)
if integration_summary.empty:
    st.info("Select at least one integration column to view its valuation.")
else:
    integration_summary["Short Name"] = integration_summary["Integration"].map(
        lambda value: f"{value[:42]}..." if len(value) > 45 else value
    )
    integration_summary["Total Seconds"] = integration_summary["Total Seconds"].map(
        lambda value: f"{value:,.0f}s"
    )
    integration_summary["Total Client Value"] = integration_summary["Total Client Value"].map(
        lambda value: f"${value:,.2f}"
    )
    if compare_market:
        integration_summary["Total Market Value"] = integration_summary["Total Market Value"].map(
            lambda value: f"${value:,.2f}"
        )
    st.dataframe(
        integration_summary.drop(columns=["Short Name"]),
        use_container_width=True,
        hide_index=True,
    )
