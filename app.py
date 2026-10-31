import os
import time
import webbrowser
import streamlit as st
import streamlit.components.v1 as components


class PipelineUI:
    def __init__(self):
        # Get the absolute path to the root folder where files reside
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.html_folder = self.base_dir

    def _open_html(self, filename):
        """Helper function to open an HTML file in the default browser."""
        file_path = os.path.join(self.html_folder, filename)
        url = f"file://{file_path}"
        webbrowser.open(url)
        print(f"[UI] Broadcasting clip: {filename}")

    def launch_stage_1(self):
        self._open_html("data_ingestion_flow.html")

    def launch_stage_3_xgboost(self):
        self._open_html("xgboost_tabular_engine.html")

    def launch_stage_3_graphsage(self):
        self._open_html("graphsage_neural_engine.html")

    def launch_stage_4(self):
        self._open_html("compliance_risk_radar.html")

    def display_final_results(self, results="N/A"):
        """Displays the real metrics once the backend finishes."""
        print("\n==================================")
        print("  RACE FINISHED: ACTUAL RESULTS  ")
        print("==================================")
        print(f"Final Data: {results}")


class StreamlitPipelineApp:
    def __init__(self):
        self.ui_broadcaster = PipelineUI()

    def render_stage_html(self, filename):
        html_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=700, scrolling=False)
        else:
            st.error(f"HTML component not found: {filename}")

    def run(self):
        st.set_page_config(page_title="Enterprise AML Pipeline", layout="wide")

        st.title("🛡️ Enterprise AML Detection Pipeline")
        st.write(
            "Live visual execution of graph neural networks & tabular fraud detection."
        )

        if st.button("🚀 Run Enterprise Pipeline"):
            # Stage 1 & 2 Combined
            st.subheader(
                "Stage 1 & 2: Data Ingestion, Cleansing, Encoding & Unsupervised Anomaly Scoring")
            self.render_stage_html("data_ingestion_flow.html")
            time.sleep(2)

            # Stage 3.A1 (Spatial Graph Representation Learning)
            st.subheader("Stage 3: Training & Testing")
            self.render_stage_html("graphsage_neural_engine.html")
            time.sleep(2)

            # Stage 3.A2 (Gradient-Boosted Classification)
            st.subheader(
                "Stage 3.B: Gradient-Boosted Classification (XGBoost)")
            self.render_stage_html("xgboost_tabular_engine.html")
            time.sleep(2)

            # Stage 4 (Compliance Risk Fusion & Reporting)
            st.subheader("Stage 4: Compliance Risk Fusion & Reporting")
            self.render_stage_html("compliance_risk_radar.html")
            time.sleep(2)

            st.success("Pipeline Run Complete!")


if __name__ == "__main__":
    pipeline_app = StreamlitPipelineApp()
    pipeline_app.run()
