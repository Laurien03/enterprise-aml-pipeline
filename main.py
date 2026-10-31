"""
Enterprise AML Detection Pipeline - Master Connector Script
Author: Laurien Michel Muhire
Description: Connects the backend pipeline engine with the optional Streamlit frontend UI.
"""

import sys
import subprocess
from pipeline_engine import AMLPipelineEngine


def main():
    print("🛡️ ==================================================")
    print("   ENTERPRISE AML DETECTION PIPELINE - MASTER CONNECTOR")
    print("================================================== 🛡️")

    # 1. Execute the backend engine (runs data flows, models, and pops up Matplotlib charts)
    print("\n[INFO] Initializing Backend AML Pipeline Engine...")
    engine = AMLPipelineEngine()
    engine.run_pipeline()

    # 2. Conditional loop / prompt for optional frontend architecture visualizer
    print("\n[INFO] Backend execution & analytics complete.")
    while True:
        choice = input(
            "Would you like to launch the Streamlit architecture UI visualizer? (y/n): ").strip().lower()

        if choice in ['y', 'yes']:
            print("\n🚀 Launching Streamlit Architecture Dashboard...")
            try:
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


if __name__ == "__main__":
    main()
