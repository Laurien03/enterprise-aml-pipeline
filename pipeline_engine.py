
from data_manager import DataManager
from anomaly_detector import AnomalyDetector
from graph_analyzer import GraphAnalyzer
from fraud_classifier import GraphSageClassifier, XGBoostClassifier
from compliance_reporter import ComplianceRiskEngine, StringBoard


class AMLPipelineEngine:
    def __init__(self, data_path="HI-Small_Trans.csv"):
        self.data_path = data_path

    def run_pipeline(self):
        print("🚀 Initializing Enterprise AML Pipeline...")

        # --- STAGE 1: Data Ingestion & Cleansing ---
        print("\n--- Running Stage 1: Data Manager ---")
        df_manager = DataManager(self.data_path)

        # inspection-cleaning
        df_manager.inspection_schema()
        df_manager.clean_data()

        # exploring-converting
        df_manager.data_conversion()
        df_manager.explore_data()

        # feature engneering & enconding
        df_manager.feature_engineering()
        df_manager.engineer_numerical_feature()
        df_manager.encode_clean()

        # Splitting
        X_train, X_cv, X_test, y_train, y_cv, y_test, ref_train, ref_cv, ref_test = df_manager.split_scale()

        # --- STAGE 2: Anomaly Score & Density Feature Augmentation ---
        print("\n--- Running Stage 2: Anomaly Detector ---")
        anomaly_detector = AnomalyDetector()

        X_train_enriched, y_train, ref_train = anomaly_detector.anomalies_scores_extraction(
            X_train, y_train, ref_train)
        X_cv_enriched, y_cv, ref_cv = anomaly_detector.anomalies_scores_extraction(
            X_cv, y_cv, ref_cv)
        X_test_enriched, y_test, ref_test = anomaly_detector.anomalies_scores_extraction(
            X_test, y_test, ref_test)

        # --- STAGE 3. Model Training & Testing ---
        # A.Spatial Graph Representation Learning

        print("\n--- Running Stage 3.1: Graph Analyzer & GraphSAGE ---")

        # --Networkx:Graph_analyzer---
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.build_network(ref_train)

        edge_index_train, X_train_degrees, node_mapping = graph_analyzer.extract_features(
            X_train_enriched, ref_train)
        edge_index_cv, X_cv_degrees, _ = graph_analyzer.extract_features(
            X_cv_enriched, ref_cv)
        edge_index_test, X_test_degrees, _ = graph_analyzer.extract_features(
            X_test_enriched, ref_test)

        # GNN SAGE CLASSIFIER
        in_features = X_train_degrees.select_dtypes(
            include=['number', 'bool']).shape[1]
        gnn_sage = GraphSageClassifier(
            in_channels=in_features, hidden_channels=64, out_channels=32)

        graph_data_train = gnn_sage.torch_conversion(
            X_train_degrees, y_train, edge_index_train)
        gnn_sage.train_model(graph_data=graph_data_train, epochs=100, lr=0.01)

        X_train_final = gnn_sage.extract_embeddings(
            graph_data_train, X_train_degrees, ref_train, node_mapping)

        graph_data_cv = gnn_sage.torch_conversion(
            X_cv_degrees, y_cv, edge_index_cv)
        X_cv_final = gnn_sage.extract_embeddings(
            graph_data_cv, X_cv_degrees, ref_cv, node_mapping)

        graph_data_test = gnn_sage.torch_conversion(
            X_test_degrees, y_test, edge_index_test)
        X_test_final = gnn_sage.extract_embeddings(
            graph_data_test, X_test_degrees, ref_test, node_mapping)

        # B.Gradient-Boosted Classification*
        # ---  XGBoost Classification ---

        print("\n--- Running Stage 3.2: XGBoost Classifier ---")
        xgb_model = XGBoostClassifier()
        xgb_model.train_model(X_train_final, y_train)
        xgb_model.evaluate_model(X_cv_final, y_cv, threshold=0.65)

        # Evaluation on Untouched Test Set
        if hasattr(xgb_model, 'model'):
            y_test_probs = xgb_model.model.predict_proba(X_test_final)[:, 1]
        else:
            y_test_probs = xgb_model.evaluate_model(
                X_test_final, y_test, threshold=0.65)

        # --- STAGE 4: Compliance Fusion & Reporting ---
        print("\n--- Running Stage 4: Compliance Risk Engine ---")
        risk_engine = ComplianceRiskEngine(xgb_weight=0.7, gnn_weight=0.3)
        flagged_df = risk_engine.fuse_predictions(ref_test, y_test_probs)

        board = StringBoard(full_graph=graph_analyzer.graph,
                            idx_to_node=node_mapping)

        print("\n✅ Pipeline Execution Completed Successfully!")
        return flagged_df, board, graph_analyzer.graph, node_mapping
