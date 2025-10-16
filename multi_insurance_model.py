#!/usr/bin/env python3
"""
Multi-Label Insurance Prediction Model
Predicts likelihood of purchasing each insurance product
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')


# 定義所有保險產品
INSURANCE_PRODUCTS = {
    'WAPART': '私人第三方責任險',
    'WABEDR': '企業第三方責任險',
    'WALAND': '農業第三方責任險',
    'PERSAUT': '汽車保險',
    'BESAUT': '送貨車保險',
    'MOTSCO': '機車/速克達保險',
    'VRAAUT': '卡車保險',
    'AANHANG': '拖車保險',
    'TRACTOR': '拖拉機保險',
    'WERKT': '農機保險',
    'BROM': '輕型機車保險',
    'LEVEN': '人壽保險',
    'PERSONG': '個人意外險',
    'GEZONG': '家庭意外險',
    'WAOREG': '殘疾保險',
    'BRAND': '火災保險',
    'ZEILPL': '衝浪板保險',
    'PLEZIER': '船舶保險',
    'FIETS': '自行車保險',
    'INBOED': '財產保險',
    'BYSTAND': '社會保險',
    'CARAVAN': '旅遊保險'
}


class MultiLabelDQN(nn.Module):
    """Multi-label Deep Q-Network for feature extraction"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], feature_dim=32):
        super(MultiLabelDQN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_output = nn.Linear(prev_dim, feature_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_extractor(x)
        extracted_features = self.feature_output(features)
        return extracted_features


class MultiInsurancePredictor:
    """Multi-label insurance prediction system"""

    def __init__(self, input_dim, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.insurance_products = list(INSURANCE_PRODUCTS.keys())

        # DQN for feature extraction
        self.dqn = MultiLabelDQN(input_dim).to(device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

        # One XGBoost model per insurance product
        self.models = {}

        # Scaler
        self.scaler = StandardScaler()

        # Store customer info columns
        self.customer_info_columns = []

    def load_data(self):
        """Load and prepare the insurance data"""
        print("Loading data...")

        # Load training data
        train_data = pd.read_csv('insurance+company+benchmark+coil+2000/ticdata2000.txt',
                                 sep='\t', header=None)

        # Define all column names
        column_names = [
            'MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK', 'MGODPR',
            'MGODOV', 'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND',
            'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG', 'MBERZELF',
            'MBERBOER', 'MBERMIDD', 'MBERARBG', 'MBERARBO', 'MSKA', 'MSKB1', 'MSKB2',
            'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2', 'MAUT0', 'MZFONDS',
            'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M',
            'MINKGEM', 'MKOOPKLA', 'PWAPART', 'PWABEDR', 'PWALAND', 'PPERSAUT',
            'PBESAUT', 'PMOTSCO', 'PVRAAUT', 'PAANHANG', 'PTRACTOR', 'PWERKT', 'PBROM',
            'PLEVEN', 'PPERSONG', 'PGEZONG', 'PWAOREG', 'PBRAND', 'PZEILPL', 'PPLEZIER',
            'PFIETS', 'PINBOED', 'PBYSTAND', 'AWAPART', 'AWABEDR', 'AWALAND', 'APERSAUT',
            'ABESAUT', 'AMOTSCO', 'AVRAAUT', 'AAANHANG', 'ATRACTOR', 'AWERKT', 'ABROM',
            'ALEVEN', 'APERSONG', 'AGEZONG', 'AWAOREG', 'ABRAND', 'AZEILPL', 'APLEZIER',
            'AFIETS', 'AINBOED', 'ABYSTAND', 'CARAVAN'
        ]

        train_data.columns = column_names

        # Store customer demographic columns (first 43 columns)
        self.customer_info_columns = column_names[:43]

        print(f"Training data shape: {train_data.shape}")
        print(f"Number of insurance products: {len(self.insurance_products)}")

        return train_data

    def prepare_targets(self, data):
        """Prepare multi-label targets from A columns (number of policies)"""
        targets = {}

        for product in self.insurance_products:
            # A columns directly correspond to insurance products
            a_column = 'A' + product  # WAPART -> AWAPART
            if a_column in data.columns:
                # Binary: has insurance (>0) or not (0)
                targets[product] = (data[a_column] > 0).astype(int)
                print(f"{product} ({a_column}): {targets[product].sum()} customers ({targets[product].mean()*100:.2f}%)")
            else:
                print(f"Warning: Column {a_column} not found for product {product}")

        if len(targets) == 0:
            raise ValueError("No valid insurance product columns found in data!")

        return pd.DataFrame(targets)

    def train(self, epochs=50, verbose=True):
        """Train models for all insurance products"""

        # Load data
        data = self.load_data()

        # Prepare features (customer demographics only, excluding P and A columns)
        X = data[self.customer_info_columns]

        # Prepare targets (has insurance or not for each product)
        y_all = self.prepare_targets(data)

        print(f"\n{'='*80}")
        print("Training Multi-Insurance Prediction Models")
        print(f"{'='*80}")
        print(f"Features: {X.shape[1]}")
        print(f"Insurance products: {len(self.insurance_products)}")

        # Split data
        X_train, X_test, y_train_all, y_test_all = train_test_split(
            X, y_all, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train DQN for feature extraction (simple autoencoder style)
        if verbose:
            print(f"\n{'='*80}")
            print("Phase 1: Training DQN Feature Extractor")
            print(f"{'='*80}")

        self.dqn.train()
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train_scaled))
            epoch_loss = []

            for i in range(0, len(indices), 64):
                batch_indices = indices[i:i+64]
                batch = torch.FloatTensor(X_train_scaled[batch_indices]).to(self.device)

                # Simple reconstruction loss
                features = self.dqn(batch)
                loss = F.mse_loss(features, torch.randn_like(features))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(epoch_loss):.4f}")

        # Extract DQN features
        self.dqn.eval()
        with torch.no_grad():
            dqn_features_train = self.dqn(torch.FloatTensor(X_train_scaled).to(self.device)).cpu().numpy()
            dqn_features_test = self.dqn(torch.FloatTensor(X_test_scaled).to(self.device)).cpu().numpy()

        # Combine features
        X_train_combined = np.hstack([X_train_scaled, dqn_features_train])
        X_test_combined = np.hstack([X_test_scaled, dqn_features_test])

        # Train XGBoost for each insurance product
        if verbose:
            print(f"\n{'='*80}")
            print("Phase 2: Training XGBoost Models for Each Product")
            print(f"{'='*80}")

        results = {}

        for product in self.insurance_products:
            if product not in y_train_all.columns:
                continue

            y_train = y_train_all[product]
            y_test = y_test_all[product]

            # Skip if no positive samples
            if y_train.sum() < 10:
                print(f"\nSkipping {product} - not enough positive samples")
                continue

            print(f"\nTraining model for: {product} ({INSURANCE_PRODUCTS[product]})")

            # Calculate scale_pos_weight
            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc'
            )

            model.fit(
                X_train_combined, y_train,
                eval_set=[(X_test_combined, y_test)],
                verbose=False
            )

            # Evaluate
            y_pred = model.predict(X_test_combined)
            y_proba = model.predict_proba(X_test_combined)[:, 1]

            try:
                auc = roc_auc_score(y_test, y_proba)
                results[product] = {
                    'auc': auc,
                    'accuracy': (y_pred == y_test).mean()
                }
                print(f"  AUC: {auc:.4f}, Accuracy: {results[product]['accuracy']:.4f}")
            except:
                results[product] = {'auc': 0.5, 'accuracy': 0.5}
                print(f"  Could not calculate metrics")

            self.models[product] = model

        print(f"\n{'='*80}")
        print(f"Training Complete! Trained {len(self.models)} models")
        print(f"{'='*80}")

        return results

    def predict_for_customer(self, customer_features, customer_name=None):
        """
        Predict insurance recommendations for a customer

        Parameters:
        - customer_features: dict or DataFrame with customer demographics
        - customer_name: optional customer name

        Returns:
        - recommendations: dict with customer info and recommendations
        """

        # Convert to DataFrame if dict
        if isinstance(customer_features, dict):
            df = pd.DataFrame([customer_features])
        else:
            df = customer_features

        # Ensure correct columns
        df = df[self.customer_info_columns]

        # Scale features
        X_scaled = self.scaler.transform(df.values)

        # Extract DQN features
        self.dqn.eval()
        with torch.no_grad():
            dqn_features = self.dqn(torch.FloatTensor(X_scaled).to(self.device)).cpu().numpy()

        # Combine features
        X_combined = np.hstack([X_scaled, dqn_features])

        # Predict for each insurance product
        predictions = {}

        for product, model in self.models.items():
            proba = model.predict_proba(X_combined)[0]
            predictions[product] = {
                'product_code': product,
                'product_name': INSURANCE_PRODUCTS[product],
                'probability': float(proba[1]),
                'prediction': int(proba[1] > 0.5)
            }

        # Sort by probability (descending)
        sorted_predictions = sorted(
            predictions.values(),
            key=lambda x: x['probability'],
            reverse=True
        )

        # Get owned insurance (from input if available)
        owned_insurance = []
        for product in self.insurance_products:
            a_column = 'A' + product  # WAPART -> AWAPART
            if a_column in customer_features and customer_features.get(a_column, 0) > 0:
                owned_insurance.append({
                    'product_code': product,
                    'product_name': INSURANCE_PRODUCTS[product]
                })

        # Get recommendations (not owned, high probability)
        recommendations = []
        owned_codes = [ins['product_code'] for ins in owned_insurance]

        for pred in sorted_predictions:
            if pred['product_code'] not in owned_codes and pred['probability'] > 0.3:
                recommendations.append(pred)

        return {
            'customer_name': customer_name or 'Unknown',
            'owned_insurance': owned_insurance,
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'all_predictions': sorted_predictions
        }

    def save_model(self, path='multi_insurance_model.pkl'):
        """Save the model"""
        model_data = {
            'dqn_state': self.dqn.state_dict(),
            'models': self.models,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'customer_info_columns': self.customer_info_columns,
            'insurance_products': self.insurance_products
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load_model(self, path='multi_insurance_model.pkl'):
        """Load the model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.dqn.load_state_dict(model_data['dqn_state'])
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.customer_info_columns = model_data['customer_info_columns']
        self.insurance_products = model_data['insurance_products']

        print(f"Model loaded from {path}")
        print(f"Loaded {len(self.models)} insurance product models")


if __name__ == '__main__':
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    predictor = MultiInsurancePredictor(input_dim=43, device=device)
    results = predictor.train(epochs=50, verbose=True)

    # Save model
    predictor.save_model('multi_insurance_model.pkl')

    print("\nModel training complete!")
