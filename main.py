import time
from data_manager import DataManager
from anomaly_detector import AnomalyDetector
from graph_analyzer import GraphAnalyzer
from fraud_classifier import XGBoostClassifier, GraphSageClassifier
from compliance_reporter import ComplianceRiskEngine, StringBoard
from pipeline_ui import PipelineUI


def main():
    ui = PipelineUI()
    print("🚀PIPELINE INITIATION....\n")

    # STAGE 1: DATA PREPARATION & FILTERING
    # ==========================================
    ui.launch_stage_1()
    time.sleep(2)

    print("=== STAGE 1: Data Processing ===")  # Instantiation
    data_manager = DataManager("Hi-Small_Trans.csv")
    data_manager.inspection_schema()
    data_manager.clean_data()
    data_manager.data_conversion()
    data_manager.explore_data()
    data_manager.feature_engineering()
    data_manager.engineer_numerical_feature()
    data_manager.encode_clean()

    X_train, X_cv, X_test, y_train, y_cv, y_test, ref_train, ref_cv, ref_test = data_manager.split_scale()

    anomaly_detector = AnomalyDetector()
    X_train_filtred, y_train_filtered, ref_train_filtered = anomaly_detector.filter_anomalies(
        X=X_train, y=y_train, ref_df=ref_train)

    X_cv_filtered, y_cv_filtered, ref_cv_filtered = anomaly_detector.filter_anomalies(
        X=X_cv, y=y_cv, ref_df=ref_cv)
    X_test_filtered, y_test_filtered, ref_test_filtered = anomaly_detector.filter_anomalies(
        X=X_test, y=y_test, ref_df=ref_test)

    graph_analyzer = GraphAnalyzer()
    graph_analyzer.build_network(data_manager.reference_df)
    X_train_enriched = graph_analyzer.extract_features(
        df=X_train_filtred, reference_df=ref_train_filtered)
    X_cv_enriched = graph_analyzer.extract_features(
        df=X_cv_filtered, reference_df=ref_cv_filtered)
    X_test_enriched = graph_analyzer.extract_features(
        df=X_test_filtered, reference_df=ref_test_filtered)

    time.sleep(1.5)  # 1.5-second cooldown before switching channels

    print("\n=== STAGE 3: Training the Detectives ===")
    # STAGE 3A: TRAINING THE DETECTIVES (XGBoost)
    ui.launch_stage_3_xgboost()
    time.sleep(2)
    xgboost_classifier = XGBoostClassifier()
    xgboost_classifier.train_model(X_train_enriched, y_train_filtered)
    xgb_probs = xgboost_classifier.evaluate_model(X_cv_enriched, y_cv_filtered)

    time.sleep(1.5)
    # --- GraphSAGE ---
    print("\n=== STAGE 3B: Spatial Message Passing ===")
    ui.launch_stage_3_graphsage()
    time.sleep(2)
    gnn = GraphSageClassifier(model=None, optimizer=None, df=None,
                              in_channels=X_train_enriched.shape[1], hidden_channels=64)
    train_graph = gnn.pytorch_conversion(
        X_train_enriched, ref_train_filtered, y_train=y_train_filtered)
    gnn.forward()
    gnn.train_model(graph_data=train_graph, epochs=50, lr=0.01)

    cv_graph = gnn.pytorch_conversion(
        X_cv_enriched, ref_cv_filtered, y_train=y_cv_filtered)
    test_graph = gnn.pytorch_conversion(
        X_test_enriched, reference_train=ref_test_filtered, y_train=y_test_filtered)
    gnn_probs = gnn.evaluate_model(cv_graph)

    time.sleep(1.5)

    # STAGE 4: HYBRID FUSION & THE STRING BOARD

    print("\n=== STAGE 4: Regulatory Reporting ===")
    ui.launch_stage_4()
    time.sleep(2)
    compliance_reporter = ComplianceRiskEngine()
    flagged_accounts = compliance_reporter.fuse_predictions(
        reference_test_df=ref_cv_filtered, xgb_probs=xgb_probs, gnn_probs=gnn_probs)
    string_board = StringBoard(
        graph_analyzer.network_graph, idx_to_node=data_manager.idx_to_node)
    time.sleep(1.5)

    # FINAL DISPLAY
    ui.display_final_results(results=flagged_accounts)


if __name__ == "__main__":
    main()
