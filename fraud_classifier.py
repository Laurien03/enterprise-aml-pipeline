import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from graph_analyzer import GraphAnalyzer


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class XGBoostClassifier:

    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train):
        # Dynamic Weighting
        clean_count = len(y_train[y_train == 0])
        fraud_count = len(y_train[y_train == 1])
        dynamic_weight = clean_count / fraud_count
        self.model = xgb.XGBClassifier(n_estimators=100,     # Number of trees to build
                                       max_depth=5,          # Maximum depth of each tree
                                       learning_rate=0.1,     # Step size shrinkage used to prevent overfitting
                                       subsample=0.8,
                                       reg_lambda=1.0,  # L2 regularization to smooth out model weights
                                       scale_pos_weight=dynamic_weight,
                                       # --- Classification Specifics ---
                                       objective='binary:logistic',  # Forces output to be a 0.0 to 1.0 probability
                                       eval_metric='logloss',       # Evaluates error based on probability distance
                                       random_state=42
                                       )

        print("Model in Training...")
        self.model.fit(X_train, y_train)
        print("✅ Model Tuned & Trained.")

    def evaluate_model(self, X_eval, y_eval, threshold=0.25):

        # 1.Computation

        print("Model Evaluating..")
        # Filter numeric features to match training
        X_eval_clean = X_eval.select_dtypes(include=['number', 'bool'])

        # Probabilities
        probabilities = self.model.predict_proba(X_eval_clean)[:, 1]
        # predictions
        predictions = (probabilities >= threshold).astype(int)

        # Accuarcy
        accuracy = accuracy_score(y_eval, predictions)
        # confusion matrix
        matrix = confusion_matrix(y_pred=predictions, y_true=y_eval)
        # classfication report
        model_report = classification_report(y_pred=predictions, y_true=y_eval)

        # 2.Display
        print("📊 --- Model Evaluation Report --- ")
        # Accurcy Score
        print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
        # Confusion Matrix
        print("🧩 Confusion Matrix:")
        print(matrix)

        # Classification Report
        print("\n📋  Classification Report:")
        print(model_report)

        return probabilities


# ------2ND ALGORITHM:GNC SAGE


class GraphSageClassifier(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels=32):
        # initialization of parent nn.Module

        super(GraphSageClassifier, self).__init__()
        self.classfier = nn.Linear(out_channels, 1)

        # Initialize the GraphSAGE Classifier.
        # in_channels: Number of input features per node.
        # hidden_channels: Number of hidden dimensions for the intermediate embeddings.
        # out_channels: Number of output classes (e.g., 2 for Fraud vs. Legitimate).

        # Defining The message Passong Layers
        # 1-hop layer: Aggregates direct neighbors
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # 2-hop layer: Aggregates neighbors' neighbors
        self.conv2 = SAGEConv(hidden_channels, out_channels)

# -------Data Conversion in Tensor

    def torch_conversion(self, enriched_train, reference_train, y_train, edge_index):

        # 🗂️ 1. TRANSLATING NODE FEATURES (x)
        # Selecting only numerical columns (filters out strings, timestamps, object IDs)
        numeric_df = enriched_train.select_dtypes(include=['number', 'bool'])
        # Converting to float32 NumPy array and PyTorch Tensor
        feature_array = numeric_df.to_numpy(dtype=np.float32)
        tensor_features = torch.from_numpy(feature_array)

        # labels (y) into a PyTorch Tensor
        tensor_labels = torch.tensor(y_train.values, dtype=torch.float32)
        # Creation of the edge_index (The Network Connections)

        # Parcking Pytorch Geometric Data Object

        graph_data = Data(x=tensor_features, y=tensor_labels,
                          edge_index=edge_index)

        # Returning the Data() object,

        return graph_data

    def forward(self, X, edge_index):
        # Layer 1: Sample neighbors and aggregate features

        X = self.conv1(X, edge_index)

        X = F.relu(X)

        X = F.dropout(X, p=0.2, training=self.training)

        # Layer 2: Output prediction mapping

        X = self.conv2(X, edge_index)

        # returning the converted Log-softmax Proabilities

        return X

    def train_model(self, graph_data, epochs=100, lr=0.01):

        print(
            f"🧠 Training Graph Neural Network for {epochs} epochs with lr={lr}...")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #  Training Mode Activation
        self.train()

        clean_count = len(graph_data.y[graph_data.y == 0])
        fraud_count = len(graph_data.y[graph_data.y == 1])

        # weights as a float tensor
        class_weights = torch.tensor(
            [1.0, clean_count / fraud_count], dtype=torch.float32)

        pos_weight = torch.tensor(
            [clean_count / fraud_count], dtype=torch.float32)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            out = self(graph_data.x, graph_data.edge_index)

            logits_shape = self.classfier(out).squeeze()

            loss = F.binary_cross_entropy_with_logits(
                logits_shape, graph_data.y, pos_weight=pos_weight)

            # backward pass

            loss.backward()

            # Turning on the optimizer step

            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(
                    f"   Epoch {epoch + 1:03d}/{epochs} | Loss: {loss.item():.4f}")

        print("✅ GNN Model Tuned & Trained.")

     # ----Model Evaluation-----

    def extract_embeddings(self, graph_data, df_enriched, reference_df, node_mapping):
        print("Model Evaluating..")

        # 1.Activate Evaluation Mode ----- #Turnning off dropout protections so predictions are stable and frozen

        self.eval()

        # 2.Running a forward pass to get predictions for the whole graph

        with torch.no_grad():
            node_embeddings = self(
                graph_data.x, graph_data.edge_index).cpu().numpy()

        # 3. Create DataFrame for the 32 embedding dimensions
        emb_cols = [f'sage_emb_{i}' for i in range(node_embeddings.shape[1])]

        # 4. Map transactions back to node integer indices to fetch structural vectors
        mapped_indices = reference_df['Account_Node_ID'].map(
            node_mapping).values

        # 5. Extract corresponding embedding rows and attach to DataFrame
        tx_embeddings = node_embeddings[mapped_indices]
        emb_df = pd.DataFrame(
            tx_embeddings, columns=emb_cols, index=df_enriched.index)

        df_fully_enriched = pd.concat([df_enriched, emb_df], axis=1)
        print(
            f"✅ Successfully attached {len(emb_cols)} GraphSAGE embedding features!")

        return df_fully_enriched
