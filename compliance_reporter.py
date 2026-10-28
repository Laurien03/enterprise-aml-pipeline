import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class ComplianceRiskEngine:
    def __init__(self, xgb_weight=0.7, gnn_weight=0.3):
        # Weighted to lean heavily on XGBoost as the main driver,
        # while keeping GNN as the structural context validator.
        self.xgb_weight = xgb_weight
        self.gnn_weight = gnn_weight

    def fuse_predictions(self, reference_test_df, xgb_probs, gnn_probs):
        print("⚖️ Fusing XGBoost and GraphSAGE predictions...")

        # Create a copy to avoid mutating the original data reference
        result_df = reference_test_df.copy()

        # 1. Storing raw probabilities for auditability
        result_df['XGB_Prob'] = xgb_probs
        result_df['GNN_Prob'] = gnn_probs

        # 2. Weighted Final Risk Score
        result_df['Final_Risk_Score'] = (
            (result_df['XGB_Prob'] * self.xgb_weight) +
            (result_df['GNN_Prob'] * self.gnn_weight)
        )

        # 3. Regulatory Risk Tiers Application
        conditions = [
            (result_df['Final_Risk_Score'] < 0.50),
            (result_df['Final_Risk_Score'] >= 0.50) & (
                result_df['Final_Risk_Score'] < 0.75),
            (result_df['Final_Risk_Score'] >= 0.75) & (
                result_df['Final_Risk_Score'] < 0.90),
            (result_df['Final_Risk_Score'] >= 0.90)
        ]

        tiers = [
            '0:Clean',
            '1:Medium Risk',
            '2:High Risk',
            '3:Critical',
        ]

        result_df['Risk_Tier'] = np.select(
            conditions, tiers, default='0:Clean'
        )

        # FIXED: Changed 'O:Clean' to '0:Clean' to correctly filter out clean accounts
        flagged_accounts = result_df[result_df['Risk_Tier']
                                     != '0:Clean'].copy()

        # Sorting: Critical items right to the top
        flagged_accounts = flagged_accounts.sort_values(
            by='Final_Risk_Score', ascending=False
        )

        print(
            f"🚨 Fusion Complete! Found {len(flagged_accounts)} flagged transactions requiring review.")
        return flagged_accounts


class StringBoard:
    def __init__(self, full_graph, idx_to_node):
        # full_graph is the nx.MultiDiGraph from Stage 2/3 (GraphAnalyzer)
        # idx_to_node is the reverse mapping dictionary from Stage 1
        self.full_graph = full_graph
        self.idx_to_node = idx_to_node

    def generate_investigator_report(self, account_node_id, risk_tier, final_score):
        # Critical tier gets 2 hops to track downstream fund movement; others get 1 hop
        radius = 2 if risk_tier == "3:Critical" else 1

        # Extract localized sub-graph around the suspect account
        sub_graph = nx.ego_graph(
            self.full_graph, account_node_id, radius=radius)

        # Map mathematical node IDs back to human-readable strings
        dict_labels = {node: self.idx_to_node.get(
            node, 'Unknown') for node in sub_graph.nodes()}

        real_account_name = dict_labels.get(
            account_node_id, str(account_node_id))

        print(f"\n--- 🕵️ INVESTIGATIVE AUDIT REPORT ---")
        print(
            f"Target Account : {real_account_name} (Mapped ID: {account_node_id})")
        print(f"Risk Tier      : {risk_tier} (Score: {final_score:.2f})")
        print(f"Network Scope  : {radius}-Hop Radius")
        print(
            f"Connected Nodes: {sub_graph.number_of_nodes() - 1} related entities")
        print(
            f"Total Edges    : {sub_graph.number_of_edges()} transaction pathways")

        # Display direct counterparties
        neighbors = list(sub_graph.neighbors(account_node_id)
                         ) if account_node_id in sub_graph else []
        print("\nDirect Counterparties (1-Hop):")
        for neighbor in neighbors[:5]:
            print(f" - {self.idx_to_node.get(neighbor, 'Unknown')}")
        if len(neighbors) > 5:
            print(f"   ...and {len(neighbors) - 5} more.")

        return sub_graph, dict_labels

    def network_visualization(self, sub_graph, labels, target_node_id):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(sub_graph, seed=42)

        # Draw standard background nodes
        nx.draw_networkx_nodes(
            sub_graph, pos=pos, node_color='lightblue', node_size=500
        )

        # Highlight target fraud node in red
        if target_node_id in sub_graph:
            nx.draw_networkx_nodes(
                sub_graph, pos=pos, nodelist=[target_node_id], node_color='red', node_size=700
            )

        # Draw edges and labels
        nx.draw_networkx_edges(sub_graph, pos=pos, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(
            sub_graph, pos=pos, labels=labels, font_size=8, font_weight='bold'
        )

        plt.title("String Board: Suspicious Transaction Network", fontsize=14)
        plt.axis('off')
        plt.show()
