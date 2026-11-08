import re

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


# ============================================================
# App setup
# ============================================================
st.set_page_config(
    page_title="SA Horse Racing Predictor",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🇿🇦 SA Horse Racing Predictor")
st.caption("Personal-use racecard reader and ratings tool")

HORSE_COLUMNS = [
    "No",
    "Draw",
    "Horse Name",
    "Weight (kg)",
    "C-Form Rating",
    "Speed Rating",
    "Jockey/Trainer %",
    "Strong Finish Index",
    "Odds",
    "Notes",
]


def empty_race_data():
    return pd.DataFrame(columns=HORSE_COLUMNS)


def ensure_horse_columns(df):
    """Add missing columns and return them in the expected order."""
    df = df.copy()

    for col in HORSE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[HORSE_COLUMNS]


if "race_data" not in st.session_state:
    st.session_state.race_data = empty_race_data()


# ============================================================
# PDF parsing
# ============================================================
def parse_racecard_pdf(pdf_file):
    """
    Best-effort extraction of runner rows resembling:
        1-1 NICOLSON LANDING 60
        2-3 GREEN STREET 58.5

    PDF layouts vary. Ratings not extracted with confidence are left blank
    for manual entry in the editable grid.
    """
    reader = PdfReader(pdf_file)
    lines = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            lines.extend(page_text.splitlines())

    # Match runner number, draw, horse name and weight.
    # This intentionally does not invent missing ratings.
    runner_pattern = re.compile(
        r"^\s*(\d+)\s*-\s*(\d+)\s+"
        r"([A-Z0-9][A-Z0-9'&./ -]{1,45}?)\s+"
        r"(\d{2}(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    runners = []
    seen = set()

    for line in lines:
        cleaned = re.sub(r"\s+", " ", line).strip()
        match = runner_pattern.search(cleaned)

        if not match:
            continue

        number, draw, horse_name, weight = match.groups()
        horse_name = re.sub(r"\s+", " ", horse_name).strip().upper()

        # Avoid duplicate rows when the PDF repeats information.
        key = (int(number), horse_name)
        if key in seen:
            continue
        seen.add(key)

        runners.append(
            {
                "No": int(number),
                "Draw": int(draw),
                "Horse Name": horse_name,
                "Weight (kg)": float(weight),
                "C-Form Rating": np.nan,
                "Speed Rating": np.nan,
                "Jockey/Trainer %": np.nan,
                "Strong Finish Index": np.nan,
                "Odds": "",
                "Notes": "",
            }
        )

    if not runners:
        return empty_race_data()

    return ensure_horse_columns(
        pd.DataFrame(runners).sort_values(["No", "Draw"]).reset_index(drop=True)
    )


# ============================================================
# Ranking and shortlist
# ============================================================
def compute_rankings(df, speed_bias, form_bias, jt_bias, finish_bias):
    ranked = ensure_horse_columns(df)

    rating_columns = [
        "C-Form Rating",
        "Speed Rating",
        "Jockey/Trainer %",
        "Strong Finish Index",
    ]

    for col in rating_columns:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce").fillna(0)

    ranked["Calculated Score"] = (
        (ranked["Speed Rating"] / 100.0) * speed_bias
        + (ranked["C-Form Rating"] / 50.0) * form_bias
        + (ranked["Jockey/Trainer %"] / 30.0) * jt_bias
        + (ranked["Strong Finish Index"] / 10.0) * finish_bias
    )

    # Odds should be entered as decimal odds, for example 4.5.
    ranked["Decimal Odds"] = pd.to_numeric(ranked["Odds"], errors="coerce")
    ranked["Value Score"] = np.nan

    valid_odds = ranked["Decimal Odds"] > 0
    ranked.loc[valid_odds, "Value Score"] = (
        ranked.loc[valid_odds, "Calculated Score"]
        / ranked.loc[valid_odds, "Decimal Odds"]
    )

    return ranked.sort_values(
        "Calculated Score", ascending=False
    ).reset_index(drop=True)


def safe_runner_label(row):
    number = row.get("No", "")
    name = row.get("Horse Name", "")
    number_text = str(number) if pd.notna(number) and str(number) else "?"
    name_text = str(name) if pd.notna(name) and str(name) else "Unnamed runner"
    return f"#{number_text} {name_text}"


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["📄 Racecard PDF", "📊 Edit race data", "🏆 Predictions"]
)


# ============================================================
# Tab 1: PDF import
# ============================================================
with tab1:
    st.subheader("Import a Computaform racecard")

    st.write(
        "Upload a text-based PDF. The parser attempts to extract horse number, "
        "draw, horse name, and weight. Check the results in the editable grid; "
        "enter ratings and odds manually if they were not extracted."
    )

    uploaded_file = st.file_uploader(
        "Choose a racecard PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button("Extract runners from PDF"):
            try:
                extracted = parse_racecard_pdf(uploaded_file)

                if extracted.empty:
                    st.warning(
                        "No runner rows were detected. The PDF may be scanned "
                        "or its layout may not match the parser."
                    )
                else:
                    st.session_state.race_data = extracted
                    # Reset editor state so it displays the newly imported data.
                    st.session_state.pop("race_grid", None)
                    st.success(f"Extracted {len(extracted)} runners.")
                    st.dataframe(extracted, use_container_width=True)

            except Exception as exc:
                st.error(f"Could not read the PDF: {exc}")

    st.info(
        "If the PDF is a scan or photo rather than selectable text, this "
        "parser cannot read it without OCR."
    )


# ============================================================
# Tab 2: Editable data
# ============================================================
with tab2:
    st.subheader("Review and edit runner data")

    st.caption(
        "Ratings and odds that were not extracted are left blank. "
        "Enter decimal odds, such as 4.5."
    )

    edited_data = st.data_editor(
        ensure_horse_columns(st.session_state.race_data),
        num_rows="dynamic",
        use_container_width=True,
        key="race_grid",
    )

    if st.button("Apply grid changes"):
        st.session_state.race_data = ensure_horse_columns(edited_data)
        st.success("Current race data updated.")

    current_data = ensure_horse_columns(edited_data)

    if not current_data.empty:
        csv_data = current_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current race as CSV",
            data=csv_data,
            file_name="race_data.csv",
            mime="text/csv",
        )


# ============================================================
# Tab 3: Predictions
# ============================================================
with tab3:
    st.subheader("Rankings and shortlist")

    race_data = ensure_horse_columns(st.session_state.race_data)

    if race_data.empty or len(race_data) < 2:
        st.warning(
            "Upload a racecard or add at least two runners in the editable grid."
        )
    else:
        missing_ratings = race_data[
            [
                "C-Form Rating",
                "Speed Rating",
                "Jockey/Trainer %",
                "Strong Finish Index",
            ]
        ].apply(pd.to_numeric, errors="coerce").isna().any().any()

        if missing_ratings:
            st.warning(
                "Some ratings are blank. Blank ratings count as zero in the "
                "calculation, so fill them in for more useful comparisons."
            )

        st.markdown("#### Adjust model factors")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            speed_bias = st.slider(
                "Speed rating factor", 0.0, 1.0, 0.40, 0.05
            )
        with col2:
            form_bias = st.slider(
                "C-Form factor", 0.0, 1.0, 0.25, 0.05
            )
        with col3:
            jt_bias = st.slider(
                "Jockey/Trainer factor", 0.0, 1.0, 0.20, 0.05
            )
        with col4:
            finish_bias = st.slider(
                "Strong finish factor", 0.0, 1.0, 0.15, 0.05
            )

        ranked = compute_rankings(
            race_data,
            speed_bias,
            form_bias,
            jt_bias,
            finish_bias,
        )

        st.markdown("---")
        st.markdown("#### Banker / danger / value shortlist")

        banker_col, danger_col, value_col = st.columns(3)

        if not ranked.empty:
            banker = ranked.iloc[0]
            with banker_col:
                st.markdown("##### 🥇 Banker — highest model score")
                st.metric("Runner", safe_runner_label(banker))
                st.write(f"Score: **{banker['Calculated Score']:.3f}**")

            if len(ranked) > 1:
                danger = ranked.iloc[1]
                with danger_col:
                    st.markdown("##### 🥈 Main danger — next highest score")
                    st.metric("Runner", safe_runner_label(danger))
                    st.write(f"Score: **{danger['Calculated Score']:.3f}**")
            else:
                danger_col.info("Add another runner to show a main danger.")

            value_candidates = ranked[
                ranked["Value Score"].notna()
            ].sort_values("Value Score", ascending=False)

            with value_col:
                st.markdown("##### 💎 Value — score compared with decimal odds")

                if value_candidates.empty:
                    st.write(
                        "Enter decimal odds in the race grid to calculate "
                        "a value ranking."
                    )
                else:
                    for _, runner in value_candidates.head(3).iterrows():
                        st.write(
                            f"**{safe_runner_label(runner)}** — "
                            f"value score {runner['Value Score']:.3f}, "
                            f"odds {runner['Decimal Odds']:.2f}"
                        )

        st.markdown("---")
        st.markdown("#### Full ranking")
        st.dataframe(ranked, use_container_width=True)

        st.caption(
            "These scores are simple comparisons based on the ratings you enter; "
            "they are not guaranteed outcomes or professional betting advice."
)
