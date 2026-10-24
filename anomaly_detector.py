# Model Importations
from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1, random_state=42, n_jobs=-1)

    def anomalies_detection(self, X, y, ref_df):
        # Model Initiation

        iso_predictions = self.model.fit(X)
        X['Anomaly_Score'] = -self.model.decision_function(X)

        # IsolationForest returns 1 for normal, -1 for anomaly
        # We create a boolean mask to keep only the normal (1) rows
        mask = (iso_predictions == 1)

        # Filtering all three dataframes synchronously to maintain the strict index lock!
        X_filtred = X[mask].copy()
        y_filtered = y.iloc[mask].copy()
        ref_filtered = ref_df.iloc[mask].copy()
        dropped_count = len(X)-len(X_filtred)  # checking dropped accounts

        print(f"🛑 Isolation Forest removed {dropped_count:,} noisy rows.\n")

        # predictions --->filtered data
        return X, y, ref_df
