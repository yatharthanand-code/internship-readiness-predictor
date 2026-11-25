"""Internship Placement Readiness Predictor - Streamlit app.

Loads the model bundle saved by training/Internship_Readiness_Training.ipynb,
scores a student profile and suggests the smallest change that would help most.
Educational project only - not a guarantee of placement.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.joblib"

# Fallbacks only; the real ranges/defaults are stored in the model bundle.
DEFAULT_RANGES = {
    "cgpa": {"min": 4.0, "max": 10.0, "default": 7.5, "step": 0.1},
    "aptitude_score": {"min": 0.0, "max": 100.0, "default": 60.0, "step": 1.0},
    "soft_skills": {"min": 1.0, "max": 10.0, "default": 6.0, "step": 0.5},
    "projects": {"min": 0.0, "max": 10.0, "default": 2.0, "step": 1.0},
}

st.set_page_config(page_title="Internship Readiness Predictor", page_icon="🎯", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .score-card {border-radius: 16px; padding: 1.4rem 1.6rem; text-align: center; margin: .5rem 0 1rem 0;}
    .score-card.ready {background: #ecfdf5; border: 2px solid #10b981;}
    .score-card.upskill {background: #fffbeb; border: 2px solid #f59e0b;}
    .score-label {font-size: .9rem; letter-spacing: .08em; text-transform: uppercase; color: #475569;}
    .score-value {font-size: 2.6rem; font-weight: 800; line-height: 1.15; color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path: str) -> dict:
    return joblib.load(path)


def get_bundle() -> dict:
    """Load the model or stop the app with a friendly message."""
    if not MODEL_PATH.exists():
        st.error(
            "**Model file not found.** `model.joblib` should sit next to `app.py`. "
            "Run `training/Internship_Readiness_Training.ipynb` to generate it, then refresh this page."
        )
        st.stop()
    try:
        return load_bundle(str(MODEL_PATH))
    except Exception as exc:  # corrupted file, sklearn version mismatch, ...
        st.error(
            "**The model file could not be loaded.** It may be corrupted or was trained with a different "
            "scikit-learn version than the one installed. Re-run the training notebook and try again."
        )
        st.caption(f"Technical details: {type(exc).__name__}: {exc}")
        st.stop()


def build_row(values: dict, features: list) -> pd.DataFrame:
    """One-row DataFrame in the EXACT feature order used during training.

    The UI works with "Yes"/"No"; the model was trained with internship encoded as 1/0.
    """
    row = {**values, "internship": 1 if values["internship"] in ("Yes", 1) else 0}
    return pd.DataFrame([row])[features]


def score(bundle: dict, values: dict) -> float:
    return float(bundle["pipeline"].predict_proba(build_row(values, bundle["features"]))[0, 1])


def make_feedback(bundle: dict, values: dict, ranges: dict, base: float) -> str:
    """Re-score the student after one small change at a time and describe the most useful lever."""
    thr = bundle["threshold"]
    cgpa_step = 0.5 if ranges["cgpa"]["max"] <= 10 else 5.0

    # Ordered roughly from least to most effort.
    levers = []
    if values["projects"] < ranges["projects"]["max"]:
        levers.append(("complete one more hands-on project", {"projects": values["projects"] + 1}))
    if values["soft_skills"] < ranges["soft_skills"]["max"]:
        levers.append(("raise your soft-skills and communication rating by 1 point",
                       {"soft_skills": min(ranges["soft_skills"]["max"], values["soft_skills"] + 1)}))
    if values["aptitude_score"] < ranges["aptitude_score"]["max"]:
        levers.append(("improve your aptitude/technical test score by 10 points",
                       {"aptitude_score": min(ranges["aptitude_score"]["max"], values["aptitude_score"] + 10)}))
    if values["internship"] == "No":
        levers.append(("complete an internship", {"internship": "Yes"}))
    if values["cgpa"] < ranges["cgpa"]["max"]:
        levers.append((f"lift your CGPA/percentage by {cgpa_step:g}",
                       {"cgpa": min(ranges["cgpa"]["max"], values["cgpa"] + cgpa_step)}))
    if not levers:
        return "Your profile is already maxed out on every lever - focus on interview preparation."

    probs = [score(bundle, {**values, **change}) for _, change in levers]
    gains = [p - base for p in probs]
    best = max(range(len(levers)), key=lambda i: gains[i])

    if base >= thr:
        if gains[best] >= 0.01:
            return (f"You're above the placement-ready threshold. To stay competitive, the strongest next step "
                    f"is to {levers[best][0]} (about +{gains[best] * 100:.0f} points).")
        return "You're above the placement-ready threshold - keep your profile balanced and sharpen interview skills."

    crossing = [i for i, p in enumerate(probs) if p >= thr]
    if crossing:
        i = crossing[0]  # least-effort lever that crosses the threshold
        return (f"Quickest fix: {levers[i][0]}. On its own that lifts your score to about "
                f"{probs[i] * 100:.0f}%, above the {thr * 100:.0f}% threshold.")

    ranked = sorted(range(len(levers)), key=lambda i: gains[i], reverse=True)
    text = (f"No single change gets you over the {thr * 100:.0f}% line, but the biggest jump comes if you "
            f"{levers[ranked[0]][0]} (about +{gains[ranked[0]] * 100:.0f} points)")
    if len(ranked) > 1 and gains[ranked[1]] > 0:
        text += f" and then {levers[ranked[1]][0]}"
    return text + "."


# ----------------------------------------------------------------------------- UI
bundle = get_bundle()
ranges = {**DEFAULT_RANGES, **bundle.get("ranges", {})}
for _r in ranges.values():  # keep defaults on the slider's step grid and inside [min, max]
    _d = _r["min"] + round((_r["default"] - _r["min"]) / _r["step"]) * _r["step"]
    _r["default"] = min(max(_d, _r["min"]), _r["max"])

st.title("🎯 Internship Placement Readiness Predictor")
st.write("Enter your current profile to see how competitive it is for internship placements "
         "and which single improvement would help most.")

with st.sidebar:
    st.header("About this model")
    st.write(f"**Algorithm:** {bundle['model_name']}")
    m = bundle.get("metrics", {})
    if m:
        st.write(f"**Test ROC-AUC:** {m.get('ROC-AUC', float('nan')):.2f}  \n**F1:** {m.get('F1', float('nan')):.2f}")
    st.write(f"**Training rows:** {bundle.get('n_training_rows', 'n/a'):,}")
    if bundle.get("data_source") == "synthetic":
        st.warning("This demo model was trained on **synthetic** data. Retrain on the Kaggle dataset "
                   "for meaningful results (see README).")
    st.divider()
    st.caption("Educational model - not career, hiring or admissions advice.")

st.subheader("Your profile")
c1, c2 = st.columns(2)
with c1:
    r = ranges["cgpa"]
    cgpa = st.slider("📚 CGPA / academic score", r["min"], r["max"], r["default"], r["step"])
    r = ranges["aptitude_score"]
    aptitude = st.slider("🧠 Aptitude / technical test score", r["min"], r["max"], r["default"], r["step"])
    r = ranges["soft_skills"]
    soft = st.slider("🗣️ Soft skills & communication (1-10)", r["min"], r["max"], r["default"], r["step"])
with c2:
    r = ranges["projects"]
    projects = st.number_input("🛠️ Completed hands-on projects", int(r["min"]), int(r["max"]), int(r["default"]), 1)
    internship = st.radio("💼 Internship experience", ["No", "Yes"], horizontal=True)
    specialization = st.selectbox("🎓 Degree specialization", bundle["specializations"])

if st.button("Predict Readiness", type="primary", use_container_width=True):
    values = {"cgpa": cgpa, "aptitude_score": aptitude, "soft_skills": soft, "projects": int(projects),
              "internship": internship, "specialization": specialization}
    try:
        proba = score(bundle, values)
        feedback = make_feedback(bundle, values, ranges, proba)
    except Exception as exc:
        st.error("Something went wrong while scoring this profile. Please check your inputs and try again.")
        st.caption(f"Technical details: {type(exc).__name__}: {exc}")
        st.stop()

    pct = round(proba * 100)
    ready = proba >= bundle["threshold"]
    label = "Placement Ready" if ready else "Needs Upskilling"
    css = "ready" if ready else "upskill"
    icon = "✅" if ready else "📈"

    st.markdown(
        f"""<div class="score-card {css}">
            <div class="score-label">Readiness score</div>
            <div class="score-value">{icon} {pct}% — {label}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(proba, 0.0), 1.0))
    st.info(f"💡 **Next step:** {feedback}")

st.divider()
st.caption(
    "📘 Educational project: predictions come from a machine-learning model trained on a limited dataset "
    "and do not guarantee placement outcomes. Use them as a conversation starter, not a verdict."
)
