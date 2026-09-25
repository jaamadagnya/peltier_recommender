from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Peltier Cooling Configuration Recommender",
    page_icon="❄️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDIFFERENCE_MARGIN = 0.001
GRID_STEP = 0.01

CRITERIA = [
    "DeltaT_C",
    "OCPI",
    "DeltaP_Pa",
    "CO2_ppm",
    "HCHO_mg_m3",
    "TVOC_mg_m3",
]

LABELS = {
    "DeltaT_C": "Temperature reduction, ΔT (°C)",
    "OCPI": "OCPI (W/Pa)",
    "DeltaP_Pa": "Pressure drop, ΔP (Pa)",
    "CO2_ppm": "CO₂ (ppm)",
    "HCHO_mg_m3": "HCHO (mg/m³)",
    "TVOC_mg_m3": "TVOC (mg/m³)",
}

BENEFIT = {
    "DeltaT_C": True,
    "OCPI": True,
    "DeltaP_Pa": False,
    "CO2_ppm": False,
    "HCHO_mg_m3": False,
    "TVOC_mg_m3": False,
}

PRESETS = {
    "Balanced": np.array([0.40, 0.20, 0.40]),
    "Maximum cooling": np.array([0.70, 0.10, 0.20]),
    "Low pressure drop": np.array([0.20, 0.65, 0.15]),
    "IAQ priority": np.array([0.15, 0.10, 0.75]),
    "Cooling–IAQ": np.array([0.45, 0.10, 0.45]),
}

DIMENSIONS = ["Cooling_robust", "Pressure_robust", "IAQ_robust"]
DIMENSION_LABELS = ["Cooling", "Pressure control", "Indoor-air quality"]


@st.cache_data
def load_data():
    observations = pd.read_csv(DATA_DIR / "analysis_dataset.csv")
    dimensions = pd.read_csv(DATA_DIR / "dimension_scores.csv")
    merged = observations.merge(
        dimensions[["Alternative", "Pareto"] + DIMENSIONS],
        on="Alternative",
        how="inner",
        validate="one_to_one",
    )
    merged["Pareto"] = merged["Pareto"].astype(bool)
    return merged


def normalize_weights(raw_weights):
    raw = np.asarray(raw_weights, dtype=float)
    if raw.sum() <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return raw / raw.sum()


def apply_constraints(frame, constraints):
    mask = np.ones(len(frame), dtype=bool)
    for criterion, limit in constraints.items():
        if BENEFIT[criterion]:
            mask &= frame[criterion].to_numpy() >= limit
        else:
            mask &= frame[criterion].to_numpy() <= limit
    return frame.loc[mask].copy()


def score_alternatives(frame, weights):
    scored = frame.copy()
    scored["Recommendation_score"] = scored[DIMENSIONS].to_numpy() @ weights
    return scored.sort_values(
        ["Recommendation_score", "Alternative"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def nearest_counterfactual(frame, weights, current_alternative):
    n = int(round(1 / GRID_STEP))
    grid = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            grid.append([i / n, j / n, (n - i - j) / n])
    grid = np.asarray(grid)

    scores = grid @ frame[DIMENSIONS].to_numpy().T
    order = np.argsort(-scores, axis=1, kind="stable")
    best = order[:, 0]
    second = order[:, 1]
    margins = scores[np.arange(len(grid)), best] - scores[np.arange(len(grid)), second]
    winners = frame.iloc[best]["Alternative"].to_numpy()

    valid = (winners != current_alternative) & (margins > INDIFFERENCE_MARGIN)
    if not valid.any():
        return None

    distances = np.linalg.norm(grid - weights, axis=1)
    candidate_indices = np.flatnonzero(valid)
    nearest = candidate_indices[np.argmin(distances[candidate_indices])]
    return {
        "Alternative": winners[nearest],
        "Weights": grid[nearest],
        "Distance": distances[nearest],
        "Margin": margins[nearest],
    }


def closest_to_constraints(frame, constraints):
    if not constraints:
        return pd.DataFrame()
    ranges = {c: max(frame[c].max() - frame[c].min(), 1e-12) for c in CRITERIA}
    rows = []
    for _, row in frame.iterrows():
        violations = {}
        total = 0.0
        for criterion, limit in constraints.items():
            if BENEFIT[criterion]:
                violation = max(0.0, limit - row[criterion])
            else:
                violation = max(0.0, row[criterion] - limit)
            violations[criterion] = violation
            total += violation / ranges[criterion]
        rows.append(
            {
                "Alternative": row["Alternative"],
                "Material": row["Material"],
                "Thickness_mm": row["Thickness_mm"],
                "Density_kg_m3": row["Density_kg_m3"],
                "Total_normalized_violation": total,
                "Violations": violations,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Total_normalized_violation", "Alternative"], kind="stable"
    )


def explanation(top, runner, weights):
    priorities = dict(zip(DIMENSION_LABELS, weights))
    primary = max(priorities, key=priorities.get)
    strengths = {
        "Cooling": top["Cooling_robust"],
        "Pressure control": top["Pressure_robust"],
        "Indoor-air quality": top["IAQ_robust"],
    }
    strongest = max(strengths, key=strengths.get)
    weakest = min(strengths, key=strengths.get)
    return (
        f"{top['Alternative']} is recommended because it gives the highest aggregate "
        f"fuzzy satisfaction under the selected priorities. The largest user priority is "
        f"**{primary.lower()}**, while the configuration's strongest satisfaction dimension "
        f"is **{strongest.lower()}**. Its main trade-off is **{weakest.lower()}**. "
        f"The closest ranked alternative is {runner['Alternative']}."
    )


df = load_data()

st.title("Explainable Peltier–Evaporative Cooling Recommender")
st.caption(
    "Select priorities and optional engineering limits to identify the most suitable "
    "experimentally tested packing configuration."
)

with st.sidebar:
    st.header("Decision preferences")
    scenario = st.selectbox(
        "Preference scenario",
        list(PRESETS) + ["Custom"],
        index=0,
    )

    if scenario == "Custom":
        cooling_raw = st.slider("Cooling priority", 0, 100, 40, 1)
        pressure_raw = st.slider("Pressure-control priority", 0, 100, 20, 1)
        iaq_raw = st.slider("IAQ priority", 0, 100, 40, 1)
        weights = normalize_weights([cooling_raw, pressure_raw, iaq_raw])
    else:
        weights = PRESETS[scenario].copy()

    st.write(
        f"Normalized weights: **{weights[0]:.0%} cooling · "
        f"{weights[1]:.0%} pressure · {weights[2]:.0%} IAQ**"
    )

    st.divider()
    st.header("Optional requirements")
    use_constraints = st.toggle("Apply engineering limits", value=False)
    constraints = {}

    if use_constraints:
        for criterion in CRITERIA:
            direction = "Minimum" if BENEFIT[criterion] else "Maximum"
            enabled = st.checkbox(
                f"Use {direction.lower()} {LABELS[criterion]}",
                key=f"enable_{criterion}",
            )
            if enabled:
                minimum = float(df[criterion].min())
                maximum = float(df[criterion].max())
                default = float(df[criterion].quantile(0.50))
                step = max((maximum - minimum) / 100, 1e-5)
                constraints[criterion] = st.number_input(
                    f"{direction} {LABELS[criterion]}",
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    format="%.5f" if maximum < 1 else "%.2f",
                    key=f"limit_{criterion}",
                )

    st.divider()
    st.caption(
        "The tool recommends only among the 48 observed configurations and does not "
        "predict untested materials, thicknesses or densities."
    )


eligible = df.loc[df["Pareto"]].copy()
feasible = apply_constraints(eligible, constraints)

if feasible.empty:
    st.error("No Pareto-efficient tested configuration satisfies all selected requirements.")
    closest = closest_to_constraints(df, constraints).iloc[0]
    st.subheader("Closest tested configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternative", closest["Alternative"])
    c2.metric("Material", closest["Material"])
    c3.metric("Thickness", f"{closest['Thickness_mm']:.0f} mm")
    c4.metric("Density", f"{closest['Density_kg_m3']:.0f} kg/m³")

    violations = closest["Violations"]
    violation_rows = []
    selected = df.loc[df["Alternative"] == closest["Alternative"]].iloc[0]
    for criterion, amount in violations.items():
        if amount > 0:
            violation_rows.append(
                {
                    "Requirement": LABELS[criterion],
                    "Observed": selected[criterion],
                    "Requested limit": constraints[criterion],
                    "Violation": amount,
                }
            )
    if violation_rows:
        st.warning("The closest configuration still violates the following requirement(s):")
        st.dataframe(pd.DataFrame(violation_rows), hide_index=True, use_container_width=True)
    st.stop()


ranked = score_alternatives(feasible, weights)
top = ranked.iloc[0]
runner = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
margin = float(top["Recommendation_score"] - runner["Recommendation_score"])
near_tie = len(ranked) > 1 and margin <= INDIFFERENCE_MARGIN

if near_tie:
    st.warning(
        f"Indifference region detected: {top['Alternative']} and {runner['Alternative']} "
        f"differ by only {margin:.6f}, which is within the {INDIFFERENCE_MARGIN:.3f} threshold."
    )
    recommendation_title = f"Near-tied alternatives: {top['Alternative']} and {runner['Alternative']}"
else:
    st.success(f"Recommended configuration: {top['Alternative']}")
    recommendation_title = f"Recommended configuration: {top['Alternative']}"

st.subheader(recommendation_title)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Material", top["Material"])
c2.metric("Packing thickness", f"{top['Thickness_mm']:.0f} mm")
c3.metric("Packing density", f"{top['Density_kg_m3']:.0f} kg/m³")
c4.metric("Fuzzy score", f"{top['Recommendation_score']:.4f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runner-up", runner["Alternative"])
c2.metric("Decision margin", f"{margin:.4f}")
c3.metric("Feasible Pareto designs", len(feasible))
c4.metric("Decision status", "Near tie" if near_tie else "Unique")

st.markdown(explanation(top, runner, weights))

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Measured performance")
    performance = pd.DataFrame(
        {
            "Response": [LABELS[c] for c in CRITERIA],
            top["Alternative"]: [top[c] for c in CRITERIA],
            runner["Alternative"]: [runner[c] for c in CRITERIA],
        }
    )
    st.dataframe(performance, hide_index=True, use_container_width=True)

    top_rows = ranked.head(min(8, len(ranked))).copy()
    fig_bar = px.bar(
        top_rows.sort_values("Recommendation_score"),
        x="Recommendation_score",
        y="Alternative",
        color="Material",
        orientation="h",
        labels={"Recommendation_score": "Fuzzy recommendation score"},
        title="Highest-ranked feasible configurations",
    )
    fig_bar.update_layout(legend_title_text="Packing material", height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Dimension-level satisfaction")
    radar = go.Figure()
    for _, row in ranked.head(min(2, len(ranked))).iterrows():
        values = [row[c] for c in DIMENSIONS]
        radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=DIMENSION_LABELS + [DIMENSION_LABELS[0]],
                fill="toself",
                name=row["Alternative"],
                opacity=0.65,
            )
        )
    radar.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        height=430,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(radar, use_container_width=True)

    if len(feasible) > 1:
        cf = nearest_counterfactual(feasible, weights, top["Alternative"])
        st.subheader("Nearest preference change")
        if cf is None:
            st.info("No different unique winner was found on the 0.01 preference grid.")
        else:
            cw, pw, iw = cf["Weights"]
            st.write(
                f"The nearest preference point that uniquely recommends **{cf['Alternative']}** is "
                f"**{cw:.0%} cooling, {pw:.0%} pressure control and {iw:.0%} IAQ**."
            )
            st.caption(
                f"Euclidean preference distance: {cf['Distance']:.4f}; "
                f"decision margin at that point: {cf['Margin']:.4f}."
            )


st.subheader("Recommendation record")
record = pd.DataFrame(
    [
        {
            "Scenario": scenario,
            "Cooling_weight": weights[0],
            "Pressure_weight": weights[1],
            "IAQ_weight": weights[2],
            "Status": "Near tie" if near_tie else "Unique recommendation",
            "Recommended_alternative": top["Alternative"],
            "Runner_up": runner["Alternative"],
            "Margin": margin,
            "Material": top["Material"],
            "Thickness_mm": top["Thickness_mm"],
            "Density_kg_m3": top["Density_kg_m3"],
            **{c: top[c] for c in CRITERIA},
        }
    ]
)
st.dataframe(record, hide_index=True, use_container_width=True)
st.download_button(
    "Download recommendation as CSV",
    data=record.to_csv(index=False).encode("utf-8"),
    file_name="peltier_recommendation.csv",
    mime="text/csv",
)

with st.expander("Method and interpretation"):
    st.markdown(
        """
        The application uses the interval-fuzzy robust dimension scores generated by the
        study notebook. Only six-objective Pareto-efficient configurations are ranked.
        User priorities are normalized to sum to one, and the recommendation score is the
        weighted sum of cooling, pressure-control and IAQ satisfaction.

        A rank-1 versus rank-2 score margin of 0.001 or less is reported as an
        **indifference region**, not as a definitive winner. Optional engineering limits
        are applied before ranking. Response perturbations in the research study represent
        hypothetical sensitivity scenarios and are not experimental confidence intervals.
        """
    )

