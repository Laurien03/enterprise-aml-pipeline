import networkx as nx


class GraphAnalyzer:

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def build_network(self, reference_df):

        print("🌐 Constructing transaction network graph...")

        self.network_graph = nx.from_pandas_edgelist(
            df=reference_df,

            source='Account_Node_ID',
            # Source node
            target='To_Bank_Node_ID',  # Target node

            create_using=nx.DiGraph()
        )

        print(
            f"🚀 Done! Graph created with {self.network_graph.number_of_nodes():,} nodes and {self.network_graph.number_of_edges():,} edges.")

    def extract_features(self, df, reference_df):

        print("📊 Extracting graph centrality features...")

        # --- Step 1 & 2: Calculate Degrees ---
        in_degree_dict = dict(self.network_graph.in_degree())

        out_degree_dict = dict(self.network_graph.out_degree())

    # --- Step 3: Map Values Back to the DataFrame ---
        df_enriched = df.copy()

        df_enriched['Out_Degree_Source'] = reference_df['Account_Node_ID'].map(
            out_degree_dict).fillna(0).astype(int)

        df_enriched['In_Degree_Target'] = reference_df['To_Bank_Node_ID'].map(
            in_degree_dict).fillna(0).astype(int)
    # --- Step 4: Return Updated DataFrame ---
        return df_enriched
