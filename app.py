"""Streamlit app: pick laptop specs -> predicted price + market segment.

Run:  streamlit run app.py   (after running the notebook once to create model.joblib)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURES, ppi

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")


@st.cache_resource
def load_artifact():
    return joblib.load(Path(__file__).parent / "model.joblib")


art = load_artifact()
opts = art["options"]

st.title("💻 Laptop Price Predictor")
st.caption(f"Model: {art['model_name']} · test R² {art['test_metrics']['R² (log)']:.2f} · "
           f"typical error ±{art['test_metrics']['MAPE (%)']:.0f}%")

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Brand", opts["Company"], index=opts["Company"].index("Dell"))
    type_name = st.selectbox("Type", opts["TypeName"], index=opts["TypeName"].index("Notebook"))
    ram = st.select_slider("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
    cpu = st.selectbox("Processor", opts["Cpu_brand"], index=opts["Cpu_brand"].index("Intel Core i5"))
    cpu_ghz = st.slider("CPU speed (GHz)", 0.9, 3.6, 2.5, 0.1)
    gpu = st.selectbox("Graphics", opts["Gpu_brand"], index=opts["Gpu_brand"].index("Intel"))
    os_name = st.selectbox("Operating system", opts["OS"], index=opts["OS"].index("Windows"))
with col2:
    inches = st.slider("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    resolution = st.selectbox("Resolution", ["1366x768", "1600x900", "1920x1080", "2304x1440", "2560x1440",
                                             "2560x1600", "2880x1800", "3200x1800", "3840x2160"], index=2)
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS panel")
    ssd = st.select_slider("SSD (GB)", [0, 8, 16, 32, 64, 128, 256, 512, 1024], value=256)
    hdd = st.select_slider("HDD (GB)", [0, 128, 500, 1000, 2000], value=0)
    weight = st.slider("Weight (kg)", 0.7, 4.7, 2.0, 0.05)

x_res, y_res = map(int, resolution.split("x"))
row = pd.DataFrame([{
    "Ram": ram, "Weight": weight, "PPI": ppi(x_res, y_res, inches), "Cpu_GHz": cpu_ghz,
    "SSD": ssd, "HDD": hdd, "Touchscreen": int(touchscreen), "IPS": int(ips),
    "Company": company, "TypeName": type_name, "Cpu_brand": cpu, "Gpu_brand": gpu, "OS": os_name,
}])[FEATURES]

if st.button("Predict price", type="primary"):
    price = float(np.exp(art["model"].predict(row)[0]))
    err = art["test_metrics"]["MAPE (%)"] / 100
    seg_id = int(art["kmeans"].predict(art["seg_scaler"].transform(row[art["seg_features"]]))[0])

    st.metric("Estimated price", f"₹{price:,.0f}")
    st.write(f"Likely range: **₹{price * (1 - err):,.0f} – ₹{price * (1 + err):,.0f}**")
    st.info(f"Market segment (K-Means, from specs only): **{art['segment_names'][seg_id]}**")
    st.caption(f"Screen sharpness: {row['PPI'][0]:.0f} PPI")
