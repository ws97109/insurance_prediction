#!/usr/bin/env python3
"""
Train DQN + XGBoost Hybrid Model
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from dqn_xgboost_model import DQNXGBoostHybrid, get_feature_importance
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and prepare the insurance data"""
    print("Loading data...")

    # Load training data
    train_data = pd.read_csv('insurance+company+benchmark+coil+2000/ticdata2000.txt',
                             sep='\t', header=None)

    # Define column names
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

    print(f"Training data shape: {train_data.shape}")
    print(f"Target distribution:\n{train_data['CARAVAN'].value_counts()}")
    print(f"Positive class ratio: {train_data['CARAVAN'].mean():.4f}")

    return train_data, column_names


def train_model():
    """Train the hybrid model"""

    # Load data
    train_data, column_names = load_data()

    # Separate features and target
    X = train_data.drop('CARAVAN', axis=1)
    y = train_data['CARAVAN']
    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nData split:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Initialize model
    print("\nInitializing DQN + XGBoost Hybrid Model...")
    model = DQNXGBoostHybrid(input_dim=X_train.shape[1], device=device)

    # Train model
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)

    metrics = model.train(
        X_train.values, y_train,
        X_val.values, y_val,
        dqn_episodes=50,  # Can increase for better performance
        verbose=True
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)

    test_pred = model.predict(X_test.values)
    test_proba = model.predict_proba(X_test.values)[:, 1]

    test_acc = (test_pred == y_test).mean()
    test_roc_auc = roc_auc_score(y_test, test_proba)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    # Feature importance
    print("\n" + "="*80)
    print("Feature Importance Analysis")
    print("="*80)

    feature_importance = get_feature_importance(model, feature_names)
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    # Save model
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metrics': {
            'train_accuracy': metrics['train_accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'test_accuracy': test_acc,
            'val_roc_auc': metrics['roc_auc'],
            'test_roc_auc': test_roc_auc
        },
        'feature_importance': feature_importance
    }

    import pickle
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Model saved to 'trained_model.pkl'")

    # Visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    create_visualizations(y_test, test_pred, test_proba, feature_importance, cm, model)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nFinal Metrics:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"\nModel Type: DQN + XGBoost Hybrid")
    print(f"Total Features: {len(feature_names)} original + 32 DQN features")

    return model, model_data


def create_visualizations(y_test, test_pred, test_proba, feature_importance, cm, model):
    """Create visualization plots"""

    plt.figure(figsize=(20, 12))

    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 2. ROC Curve
    plt.subplot(2, 3, 2)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    roc_auc = roc_auc_score(y_test, test_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. Feature Importance (Top 20)
    plt.subplot(2, 3, 3)
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # 4. Prediction Distribution
    plt.subplot(2, 3, 4)
    plt.hist(test_proba[y_test == 0], bins=50, alpha=0.5, label='Will Not Buy', color='blue')
    plt.hist(test_proba[y_test == 1], bins=50, alpha=0.5, label='Will Buy', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # 5. DQN Training Loss (if available)
    plt.subplot(2, 3, 5)
    if model.training_history['dqn_loss']:
        plt.plot(model.training_history['dqn_loss'], linewidth=2)
        plt.xlabel('Episode (x10)')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No DQN Loss History', ha='center', va='center')

    # 6. Feature Type Distribution
    plt.subplot(2, 3, 6)
    original_features = feature_importance[~feature_importance['Feature'].str.startswith('DQN_')]
    dqn_features = feature_importance[feature_importance['Feature'].str.startswith('DQN_')]

    importance_by_type = [
        original_features['Importance'].sum(),
        dqn_features['Importance'].sum()
    ]
    labels = [f'Original Features\n({len(original_features)})',
              f'DQN Features\n({len(dqn_features)})']

    plt.pie(importance_by_type, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Importance by Type', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dqn_xgboost_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'dqn_xgboost_analysis.png'")


if __name__ == '__main__':
    train_model()
