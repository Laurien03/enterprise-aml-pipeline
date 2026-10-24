# Model Importations
from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1, random_state=42, n_jobs=-1)

    def anomalies_scores_extraction(self, X, y, ref_df):
        print("🌲 Fitting Isolation Forest and extracting anomaly scores...")
        # Model Initiation

        self.model.fit(X)
        X_enriched = X.copy()

        X_enriched['Anomaly_Score'] = -self.model.decision_function(X)

        print("✅ Anomaly_Score attached. 0 rows dropped!\n")

        # full enriched data
        return X_enriched, y, ref_df
