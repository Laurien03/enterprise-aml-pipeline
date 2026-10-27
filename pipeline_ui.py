import os
import webbrowser


class PipelineUI:
    def __init__(self):
        # Get the absolute path to the root folder where files reside
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # HTML files are now in the root workspace directory
        self.html_folder = self.base_dir

    def _open_html(self, filename):
        """Helper function to open an HTML file in the default browser."""
        file_path = os.path.join(self.html_folder, filename)

        # Convert path to a URL format for the browser
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
