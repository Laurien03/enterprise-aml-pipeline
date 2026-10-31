"""
Enterprise AML Detection Pipeline - Master Connector Script
Author: Laurien Michel Muhire
Description: Connects the backend pipeline engine safely across local and cloud environments.
"""

import sys
import os
from pipeline_engine import AMLPipelineEngine


def main():
    print("🛡️ ==================================================")
    print("   ENTERPRISE AML DETECTION PIPELINE - MASTER CONNECTOR")
    print("================================================== 🛡️")

    # 1. Execute the backend engine (runs data flows, models, and renders analytics)
    print("\n[INFO] Initializing Backend AML Pipeline Engine...")
    data_file = "HI-Small_Trans.csv"
    if not os.path.exists(data_file):
        # Look one level up if executed from a subdirectory
        if os.path.exists(f"../{data_file}"):
            data_file = f"../{data_file}"

    engine = AMLPipelineEngine(data_path=data_file)
    flagged_df, board, graph, node_mapping = engine.run_pipeline()
    print(
        f"\n✅ Pipeline executed successfully! Flagged cases: {len(flagged_df)}")

    # 2. Handle UI prompt safely (Only if running in a true interactive terminal)
    if sys.stdin.isatty():
        print("\n[INFO] Backend execution & analytics complete.")
        while True:
            choice = input(
                "Would you like to launch the Streamlit architecture UI visualizer locally? (y/n): ").strip().lower()

            if choice in ['y', 'yes']:
                print("\n🚀 Launching Streamlit Architecture Dashboard...")
                try:
                    import subprocess
                    subprocess.run(
                        [sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
                except KeyboardInterrupt:
                    print("\n🛑 Streamlit application closed.")
                break
            elif choice in ['n', 'no']:
                print(
                    "\n✅ Skipping UI launch. Pipeline session closed cleanly. Have a great day!")
                break
            else:
                print("⚠️ Invalid input. Please type 'y' for yes or 'n' for no.")
    else:
        print(
            "\n[INFO] Headless/Cloud environment detected. Skipping interactive UI prompt.")
        print("🚀 Note: For web UI deployment, this repository runs natively on Hugging Face Spaces via app.py.")


if __name__ == "__main__":
    main()
