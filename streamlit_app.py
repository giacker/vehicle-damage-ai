"""Vehicle Damage AI — Streamlit web app."""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from pipeline import DamagePipeline

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

DET_PATH = MODELS_DIR / "detection_best.pt"
SEV_PATH = MODELS_DIR / "severity_best.pt"
COST_PATH = MODELS_DIR / "cost_model.pkl"

st.set_page_config(page_title="Vehicle Damage AI", page_icon="icon.png", layout="wide")


@st.cache_resource
def load_pipeline():
    if not (DET_PATH.exists() and SEV_PATH.exists() and COST_PATH.exists()):
        return None
    return DamagePipeline(str(DET_PATH), str(SEV_PATH), str(COST_PATH))


st.title("🚗 AI-Based Vehicle Damage Detection & Cost Estimation")
st.caption(
    "Upload an image of a damaged vehicle. The system detects damage, "
    "classifies severity, and estimates repair cost."
)

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.10, max_value=0.90, value=0.25, step=0.05,
        help="Lower detects more damage but more false positives.",
    )
    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        """
        1. **Detection** — YOLOv8s → damage type + location
        2. **Severity** — ResNet-18 → minor / moderate / severe
        3. **Cost** — Random Forest → INR estimate
        """
    )
    st.divider()
    st.markdown("### Color legend")
    st.markdown("🟢 minor &nbsp;&nbsp; 🟠 moderate &nbsp;&nbsp; 🔴 severe")

pipeline = load_pipeline()
if pipeline is None:
    st.error(
        f"❌ Trained models not found. Expected files:\n\n"
        f"- `{DET_PATH}`\n- `{SEV_PATH}`\n- `{COST_PATH}`\n\n"
        "Train them first via Notebooks 2, 3, 4."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload a vehicle image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read that image. Try a different file.")
    st.stop()

with st.spinner("Analyzing damage..."):
    annotated, findings, total_cost = pipeline.analyze(
        img_bgr, conf_threshold=conf_threshold
    )

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Detected damage")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("Cost estimate")

    if not findings:
        st.success("✅ No damage detected.")
    else:
        st.metric("Total estimated repair cost", f"INR {total_cost:,.0f}")

        by_sev = {"minor": 0, "moderate": 0, "severe": 0}
        for f in findings:
            by_sev[f["severity"]] += f["estimated_cost"]

        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 Minor",    f"INR {by_sev['minor']:,.0f}")
        s2.metric("🟠 Moderate", f"INR {by_sev['moderate']:,.0f}")
        s3.metric("🔴 Severe",   f"INR {by_sev['severe']:,.0f}")

        st.divider()
        st.markdown("**Itemized breakdown**")
        for i, f in enumerate(findings, 1):
            icon = {"minor": "🟢", "moderate": "🟠", "severe": "🔴"}[f["severity"]]
            with st.expander(
                f"{icon} #{i} · {f['damage_type']} · {f['severity']} · "
                f"INR {f['estimated_cost']:,.0f}"
            ):
                st.write(f"**Damage type:** {f['damage_type']}")
                st.write(f"**Severity:** {f['severity']} "
                         f"(confidence: {f['severity_conf']*100:.1f}%)")
                st.write(f"**Detection confidence:** {f['detection_conf']*100:.1f}%")
                st.write(f"**Area:** {f['area_pct']*100:.2f}% of image")
                st.write(f"**Estimated cost:** INR {f['estimated_cost']:,.0f}")

        st.divider()
        st.caption(
            "⚠️ Cost estimates are indicative. Actual repair costs vary by "
            "vehicle model, parts availability, and garage rates."
        )