import os
import time
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Enterprise AML Pipeline", layout="wide")

st.title("🛡️ Enterprise AML Detection Pipeline")
st.write("Live visual execution of graph neural networks & tabular fraud detection.")


def render_stage_html(filename):
    # HTML files reside directly in the root workspace directory
    html_path = os.path.join(os.path.dirname(__file__), filename)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=700, scrolling=False)


if st.button("🚀 Run Enterprise Pipeline"):
    st.subheader("Stage 1 & 2: Data Preparation & Ingestion")
    render_stage_html("data_ingestion_flow.html")
    time.sleep(2)

    st.subheader("Stage 3B: GraphSAGE Neural Engine")
    render_stage_html("graphsage_neural_engine.html")
    time.sleep(2)

    st.subheader("Stage 3A: XGBoost Tabular Classification")
    render_stage_html("xgboost_tabular_engine.html")
    time.sleep(2)

    st.subheader("Stage 4: Compliance Fusion Radar")
    render_stage_html("compliance_risk_radar.html")
    time.sleep(2)

    st.success("Pipeline Run Complete!")
