import os
import webbrowser


class PipelineUI:
    def __init__(self):
        # Get the absolute path to the AN-Fraud root folder
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Point specifically to your new HTML folder
        self.html_folder = os.path.join(self.base_dir, "HTML Files")

    def _open_html(self, filename):
        """Helper function to open an HTML file in the default browser."""
        # Join the folder path with the specific file name
        file_path = os.path.join(self.html_folder, filename)

        # Convert path to a URL format for the browser
        url = f"file://{file_path}"
        webbrowser.open(url)
        print(f"[UI] Broadcasting clip: {filename}")

    def launch_stage_1(self):
        self._open_html("data_ingestion_flow (1).html")

    def launch_stage_3_xgboost(self):
        self._open_html("xgboost_tabular_engine.html")

    def launch_stage_3_graphsage(self):
        self._open_html("graphsage_neural_engine.html")

    def launch_stage_4(self):
        self._open_html("compliance_risk_radar (1).html")

    def display_final_results(self, results="N/A"):
        """Displays the real metrics once the backend finishes."""
        print("\n==================================")
        print("  RACE FINISHED: ACTUAL RESULTS  ")
        print("==================================")
        print(f"Final Data: {results}")
