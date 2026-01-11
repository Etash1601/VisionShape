import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# ---------------- Image Processing Logic ---------------- #
def process_image(image, threshold_val):
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, threshold_val, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output_img = img_array.copy()
    data = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area < 100:
            continue

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif vertices > 4:
            shape = "Circle / Oval"
        else:
            shape = f"Polygon ({vertices} sides)"

        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(
            output_img,
            f"ID:{i+1}",
            (approx[0][0][0], approx[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        data.append(
            {
                "Object ID": i + 1,
                "Shape Type": shape,
                "Vertices": vertices,
                "Area (px²)": round(area, 2),
                "Perimeter (px)": round(perimeter, 2),
            }
        )

    return output_img, thresh, data


# ---------------- Page Config ---------------- #
st.set_page_config(page_title="Geometric Shape Analyzer", layout="wide")

# ---------------- Light Theme Styling ---------------- #
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
        color: #000000;
    }
    .dashboard-header {
        background: linear-gradient(90deg, #4facfe, #00c6ff);
        padding: 1.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ---------------- #
st.markdown(
    """
    <div class="dashboard-header">
        <h1>📐 Geometric Shape & Contour Analyzer</h1>
        <h3>Computer Vision DA1</h3>
        <p>Interactive Computer Vision Dashboard for Shape Detection & Feature Extraction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])
    threshold = st.slider("🎚️ Binary Threshold", 0, 255, 127)

# ---------------- Main Dashboard ---------------- #
if uploaded_file:
    image = Image.open(uploaded_file)
    processed_img, thresh_img, stats = process_image(image, threshold)
    df = pd.DataFrame(stats)

    # ---------- Metrics ---------- #
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Analysis Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Objects", len(df))
    col2.metric(
        "Average Area (px²)",
        f"{df['Area (px²)'].mean():.1f}" if not df.empty else "0",
    )

    if not df.empty:
        most_common_shape = df["Shape Type"].mode()[0]
        shape_count = df["Shape Type"].value_counts()[most_common_shape]
    else:
        most_common_shape, shape_count = "N/A", 0

    col3.metric("Most Occurred Shape", most_common_shape)
    col4.metric("Occurrences", shape_count)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Visual Outputs ---------- #
    with st.expander("🖼️ Visual Outputs", expanded=True):
        c1, c2 = st.columns(2)
        c1.image(processed_img, caption="Annotated Contours", use_container_width=True)
        c2.image(thresh_img, caption="Binary Threshold Image", use_container_width=True)

    # ---------- Download Images ---------- #
    with st.expander("⬇️ Download Outputs"):
        buf1 = io.BytesIO()
        Image.fromarray(processed_img).save(buf1, format="PNG")
        st.download_button(
            "Download Contour Image",
            buf1.getvalue(),
            "contours.png",
            "image/png",
        )

        buf2 = io.BytesIO()
        Image.fromarray(thresh_img).save(buf2, format="PNG")
        st.download_button(
            "Download Binary Image",
            buf2.getvalue(),
            "binary.png",
            "image/png",
        )

    # ---------- Shape Distribution ---------- #
    if not df.empty:
        with st.expander("📈 Shape Distribution"):
            st.bar_chart(df["Shape Type"].value_counts())

    # ---------- Data Table ---------- #
    with st.expander("📋 Extracted Feature Data", expanded=True):
        st.dataframe(df, use_container_width=True, height=350)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV Report",
            csv,
            "shape_analysis.csv",
            "text/csv",
        )

else:
    st.info("⬅️ Upload an image from the sidebar to start the analysis.")
