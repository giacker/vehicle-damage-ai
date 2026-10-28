"""Vehicle Damage AI - Streamlit app with brand pricing, camera, PDF report, insurance."""
from pathlib import Path
from io import BytesIO
import base64
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)

from pipeline import DamagePipeline

# ---- paths ----
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DET_PATH = MODELS_DIR / "detection_best.pt"
SEV_PATH = MODELS_DIR / "severity_best.pt"
COST_PATH = MODELS_DIR / "cost_model.pkl"
ICON_PATH = APP_DIR / "icon.png"

# ---- brand tiers: multiplier applied to base cost ----
# ponytail: flat tiers, upgrade to per-model lookup when real quote data lands
BRANDS = {
    "Economy (Maruti, Hyundai, Tata, Renault)": 1.00,
    "Mid-range (Honda, Toyota, Kia, MG)": 1.40,
    "Premium (Volkswagen, Skoda, Jeep)": 2.00,
    "Luxury (Mercedes, BMW, Audi, Volvo)": 3.50,
}

# ponytail: single national average, upgrade to policy-aware tiers if user actually enters policy
INSURANCE_COPAY_PCT = 0.20
INSURANCE_DEDUCTIBLE = 1000

st.set_page_config(
    page_title="Vehicle Damage AI",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🚗",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    if not (DET_PATH.exists() and SEV_PATH.exists() and COST_PATH.exists()):
        return None
    return DamagePipeline(str(DET_PATH), str(SEV_PATH), str(COST_PATH))


def build_pdf_report(original_bgr, annotated_bgr, findings, total, brand_label, oop):
    """Return PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#0F2942"))
    sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#6B7280"))

    story = [
        Paragraph("Vehicle Damage AI – Repair Estimate", title),
        Paragraph(f"Vehicle class: {brand_label}", sub),
        Spacer(1, 8 * mm),
    ]

    # annotated image, scaled to fit page width
    img_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    img_buf = BytesIO()
    Image.fromarray(img_rgb).save(img_buf, format="JPEG", quality=85)
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=170 * mm, height=170 * mm * annotated_bgr.shape[0] / annotated_bgr.shape[1]))
    story.append(Spacer(1, 6 * mm))

    if not findings:
        story.append(Paragraph("No damage detected.", styles["Normal"]))
    else:
        rows = [["#", "Damage", "Severity", "Area", "Cost (INR)"]]
        for i, f in enumerate(findings, 1):
            rows.append([
                str(i),
                f["damage_type"],
                f["severity"],
                f"{f['area_pct']*100:.1f}%",
                f"{f['estimated_cost']:,.0f}",
            ])
        rows.append(["", "", "", "Total", f"{total:,.0f}"])
        t = Table(rows, colWidths=[12*mm, 60*mm, 28*mm, 20*mm, 30*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2942")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
            ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)

        if oop is not None:
            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph(
                f"<b>With insurance:</b> estimated out-of-pocket "
                f"<b>INR {oop:,.0f}</b> "
                f"({int(INSURANCE_COPAY_PCT*100)}% copay + INR {INSURANCE_DEDUCTIBLE:,} deductible)",
                styles["Normal"],
            ))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "Estimates are indicative. Actual costs vary by garage, parts availability, and insurance policy terms.",
        sub,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ---- header ----
st.title("🚗 AI-Based Vehicle Damage Detection & Cost Estimation")
st.caption("Upload or capture an image → detect damage, classify severity, estimate repair cost.")

with st.sidebar:
    st.header("Vehicle")
    brand_label = st.selectbox("Vehicle class", list(BRANDS.keys()))
    brand_mult = BRANDS[brand_label]

    st.header("Insurance")
    has_insurance = st.toggle("I have insurance", value=False)

    st.header("Detection")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)

    st.divider()
    st.markdown("### Severity legend")
    st.markdown("🟢 minor  🟠 moderate  🔴 severe")

pipeline = load_pipeline()
if pipeline is None:
    st.error("Trained models not found in `models/` folder.")
    st.stop()

# ---- input: upload OR camera ----
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Take photo"])
with tab_upload:
    uploaded = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with tab_camera:
    uploaded = st.camera_input("Point your camera at the damage") or uploaded

if uploaded is None:
    st.info("Upload an image or take a photo to get started.")
    st.stop()

img_bgr = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read that image.")
    st.stop()

with st.spinner("Analyzing..."):
    annotated, findings, raw_total = pipeline.analyze(img_bgr, conf_threshold=conf_threshold)

# apply brand multiplier
for f in findings:
    f["estimated_cost"] *= brand_mult
total = raw_total * brand_mult
oop = total * INSURANCE_COPAY_PCT + INSURANCE_DEDUCTIBLE if (has_insurance and findings) else None

# ---- side-by-side ----
st.subheader("Before / after")
c1, c2 = st.columns(2)
with c1:
    st.caption("Original")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.caption("Detected damage")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

# ---- results panel ----
st.subheader("Estimate")
if not findings:
    st.success("✅ No damage detected.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Total repair cost", f"INR {total:,.0f}")
m2.metric("Vehicle class multiplier", f"{brand_mult:.2f}×")
if oop is not None:
    m3.metric("Your out-of-pocket (with insurance)", f"INR {oop:,.0f}",
              delta=f"-INR {total - oop:,.0f}", delta_color="inverse")

st.markdown("**Itemized breakdown**")
for i, f in enumerate(findings, 1):
    icon = {"minor": "🟢", "moderate": "🟠", "severe": "🔴"}[f["severity"]]
    with st.expander(f"{icon} #{i} · {f['damage_type']} · {f['severity']} · INR {f['estimated_cost']:,.0f}"):
        st.write(f"**Damage type:** {f['damage_type']}")
        st.write(f"**Severity:** {f['severity']} ({f['severity_conf']*100:.1f}% confidence)")
        st.write(f"**Detection confidence:** {f['detection_conf']*100:.1f}%")
        st.write(f"**Area:** {f['area_pct']*100:.2f}% of image")
        st.write(f"**Cost:** INR {f['estimated_cost']:,.0f} (base × {brand_mult:.2f} for vehicle class)")

# ---- PDF download ----
pdf_bytes = build_pdf_report(img_bgr, annotated, findings, total, brand_label, oop)
st.download_button(
    "📄 Download PDF report",
    data=pdf_bytes,
    file_name="damage_report.pdf",
    mime="application/pdf",
    type="primary",
)

st.caption(
    "⚠️ Cost estimates are indicative. Actual repair costs vary by vehicle model, "
    "parts availability, garage rates, and insurance policy terms."
)


def _demo():
    """ponytail: smoke test — brand multiplier applies, insurance math is right."""
    assert BRANDS["Economy (Maruti, Hyundai, Tata, Renault)"] == 1.0
    assert BRANDS["Luxury (Mercedes, BMW, Audi, Volvo)"] == 3.5
    # oop formula
    total = 10000
    oop = total * INSURANCE_COPAY_PCT + INSURANCE_DEDUCTIBLE
    assert oop == 3000, oop
    print("ok")


if __name__ == "__main__":
    _demo()
