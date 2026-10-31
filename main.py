from pipeline_engine import AMLPipelineEngine


def main():
    print("========================================")
    print("   STARTING AML BACKEND PIPELINE RUN    ")
    print("========================================")

    # Initialize the engine (pointing to your dataset)
    engine = AMLPipelineEngine(data_path="HI-Small_Trans.csv")

    # Run the full end-to-end pipeline
    flagged_df, board, graph, node_mapping = engine.run_pipeline()

    print("\n========================================")
    print("         PIPELINE EXECUTION SUMMARY      ")
    print("========================================")
    print(f"Total Flagged High-Risk Cases: {len(flagged_df)}")
    print(f"Network Graph Nodes: {graph.number_of_nodes()}")
    print(f"Network Graph Edges: {graph.number_of_edges()}")
    print("========================================")


if __name__ == "__main__":
    main()
