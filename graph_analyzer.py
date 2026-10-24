import networkx as nx
import pandas as pd
import torch


class GraphAnalyzer:

    def __init__(self):
        self.network_graph = nx.DiGraph()
        self.node_mapping = {}

    def build_network(self, reference_df):

        print("🌐 Constructing transaction network graph...")

        # 1. Mapping string node IDs to contiguous 0-indexed integers for PyTorch/PyG
        uniques_nodes = pd.concat(
            [reference_df['Account_Node_ID'], reference_df['To_Bank_Node_ID']]
        ).unique()

        self.node_mapping = {node_id: idx for idx,
                             node_id in enumerate(uniques_nodes)}

        # 2. Build NetworkX graph using continuous integer node IDs
        mapped_df = pd.DataFrame({
            'source': reference_df['Account_Node_ID'].map(self.node_mapping),
            'target': reference_df['To_Bank_Node_ID'].map(self.node_mapping)
        })

        self.network_graph = nx.from_pandas_edgelist(
            df=mapped_df,

            source='source',
            # Source node

            target='target',  # Target node

            create_using=nx.DiGraph()

        )

        print(
            f"🚀 Done! Graph created with {self.network_graph.number_of_nodes():,} nodes and {self.network_graph.number_of_edges():,} edges.")

        return self.node_mapping

    def extract_features(self, df, reference_df):

        print("📊 Extracting graph centrality features & PyG edge tensor...")

        # 1. Extract PyG edge_index tensor directly from NetworkX edges [2, E]
        edges = list(self.network_graph.edges())
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 2: Calculate Degrees for classical scalar features ---
        in_degree_dict = dict(self.network_graph.in_degree())

        out_degree_dict = dict(self.network_graph.out_degree())

    # --- Step 3: Map Values Back to the DataFrame ---
        df_enriched = df.copy()
        # 3. Map integer mapped IDs back to degrees
        mapped_src = reference_df['Account_Node_ID'].map(self.node_mapping)
        mapped_dst = reference_df['To_Bank_Node_ID'].map(self.node_mapping)

        df_enriched['Out_Degree_Source'] = mapped_src.map(
            out_degree_dict).fillna(0).astype(int)

        df_enriched['In_Degree_Target'] = mapped_dst.map(
            in_degree_dict).fillna(0).astype(int)
    # --- Step 4: Return Updated DataFrame ---
        return edge_index, df_enriched, self.node_mapping
