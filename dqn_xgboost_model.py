#!/usr/bin/env python3
"""
DQN + XGBoost Hybrid Model for Insurance Prediction
Combines Deep Q-Network with XGBoost for enhanced prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class DQN(nn.Module):
    """Deep Q-Network for feature extraction and decision making"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=2):
        super(DQN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers (without BatchNorm to avoid single sample issues)
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer for Q-values
        self.q_value_layer = nn.Linear(prev_dim, output_dim)

        # Feature output layer (for XGBoost)
        self.feature_output = nn.Linear(prev_dim, 32)

    def forward(self, x):
        """Forward pass"""
        # Ensure input is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.feature_extractor(x)
        q_values = self.q_value_layer(features)
        extracted_features = self.feature_output(features)
        return q_values, extracted_features



class ReplayMemory:
    """Experience replay memory for DQN training"""

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNXGBoostHybrid:
    """Hybrid model combining DQN and XGBoost"""

    def __init__(self, input_dim, device='cpu'):
        self.device = device
        self.input_dim = input_dim

        # DQN networks
        self.policy_net = DQN(input_dim).to(device)
        self.target_net = DQN(input_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # DQN training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        # XGBoost model
        self.xgb_model = None

        # Scaler
        self.scaler = StandardScaler()

        # Training history
        self.training_history = {
            'dqn_loss': [],
            'xgb_accuracy': [],
            'hybrid_accuracy': []
        }

    def _compute_reward(self, prediction, actual):
        """Compute reward for reinforcement learning"""
        # True Positive: High reward
        if prediction == 1 and actual == 1:
            return 10.0
        # True Negative: Medium reward
        elif prediction == 0 and actual == 0:
            return 1.0
        # False Positive: Small penalty
        elif prediction == 1 and actual == 0:
            return -2.0
        # False Negative: Large penalty (missing potential customer)
        else:
            return -5.0

    def train_dqn_step(self):
        """Perform one training step for DQN"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # Compute Q(s_t, a)
        q_values, _ = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch)
            next_state_values = next_q_values.max(1)[0]
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def extract_dqn_features(self, X):
        """Extract features using trained DQN"""
        self.policy_net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, features = self.policy_net(X_tensor)
            return features.cpu().numpy()

    def train(self, X_train, y_train, X_val, y_val, dqn_episodes=100, verbose=True):
        """Train the hybrid model"""

        if verbose:
            print("="*80)
            print("DQN + XGBoost Hybrid Model Training")
            print("="*80)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Phase 1: Train DQN with reinforcement learning
        if verbose:
            print("\nPhase 1: Training DQN...")

        for episode in range(dqn_episodes):
            # Shuffle training data
            indices = np.random.permutation(len(X_train_scaled))
            episode_loss = []

            for idx in indices:
                state = X_train_scaled[idx]
                actual = y_train.iloc[idx] if isinstance(y_train, pd.Series) else y_train[idx]

                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values, _ = self.policy_net(state_tensor)
                        action = q_values.max(1)[1].item()

                # Compute reward
                reward = self._compute_reward(action, actual)

                # Store transition
                self.memory.push(state, action, reward, state, True)

                # Train DQN
                loss = self.train_dqn_step()
                if loss is not None:
                    episode_loss.append(loss)

            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Log progress
            if verbose and (episode + 1) % 10 == 0:
                avg_loss = np.mean(episode_loss) if episode_loss else 0
                print(f"Episode {episode + 1}/{dqn_episodes} - Loss: {avg_loss:.4f} - Epsilon: {self.epsilon:.4f}")
                self.training_history['dqn_loss'].append(avg_loss)

        # Phase 2: Extract features from DQN
        if verbose:
            print("\nPhase 2: Extracting DQN features...")

        dqn_features_train = self.extract_dqn_features(X_train_scaled)
        dqn_features_val = self.extract_dqn_features(X_val_scaled)

        # Combine original features with DQN features
        X_train_combined = np.hstack([X_train_scaled, dqn_features_train])
        X_val_combined = np.hstack([X_val_scaled, dqn_features_val])

        if verbose:
            print(f"Original features: {X_train_scaled.shape[1]}")
            print(f"DQN features: {dqn_features_train.shape[1]}")
            print(f"Combined features: {X_train_combined.shape[1]}")

        # Phase 3: Train XGBoost on combined features
        if verbose:
            print("\nPhase 3: Training XGBoost...")

        # Calculate scale_pos_weight for imbalanced data
        y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
        scale_pos_weight = (y_train_array == 0).sum() / (y_train_array == 1).sum()

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20
        )

        self.xgb_model.fit(
            X_train_combined, y_train,
            eval_set=[(X_val_combined, y_val)],
            verbose=False
        )

        # Evaluate
        train_pred = self.xgb_model.predict(X_train_combined)
        val_pred = self.xgb_model.predict(X_val_combined)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        if verbose:
            print(f"\nTraining Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")

        self.training_history['xgb_accuracy'].append(val_acc)

        # Phase 4: Hybrid prediction evaluation
        if verbose:
            print("\nPhase 4: Hybrid Model Evaluation...")

        val_proba = self.xgb_model.predict_proba(X_val_combined)[:, 1]
        roc_auc = roc_auc_score(y_val, val_proba)

        if verbose:
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, val_pred))

        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'roc_auc': roc_auc
        }

    def predict(self, X):
        """Make predictions"""
        # Scale
        X_scaled = self.scaler.transform(X)

        # Extract DQN features
        dqn_features = self.extract_dqn_features(X_scaled)

        # Combine features
        X_combined = np.hstack([X_scaled, dqn_features])

        # Predict with XGBoost
        return self.xgb_model.predict(X_combined)

    def predict_proba(self, X):
        """Predict probabilities"""
        # Scale
        X_scaled = self.scaler.transform(X)

        # Extract DQN features
        dqn_features = self.extract_dqn_features(X_scaled)

        # Combine features
        X_combined = np.hstack([X_scaled, dqn_features])

        # Predict with XGBoost
        return self.xgb_model.predict_proba(X_combined)

    def save_model(self, path='dqn_xgboost_model.pkl'):
        """Save the hybrid model"""
        model_data = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'training_history': self.training_history
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load_model(self, path='dqn_xgboost_model.pkl'):
        """Load the hybrid model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.policy_net.load_state_dict(model_data['policy_net_state'])
        self.target_net.load_state_dict(model_data['target_net_state'])
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.training_history = model_data['training_history']

        print(f"Model loaded from {path}")


def get_feature_importance(model, feature_names):
    """Get feature importance from XGBoost"""
    if model.xgb_model is None:
        return None

    importance = model.xgb_model.feature_importances_

    # Create feature names for combined features
    dqn_feature_names = [f'DQN_Feature_{i}' for i in range(32)]
    all_feature_names = list(feature_names) + dqn_feature_names

    feature_importance = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return feature_importance
