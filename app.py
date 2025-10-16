#!/usr/bin/env python3
"""
DQN + XGBoost Insurance Prediction Web Application
Provides REST API and web interface for insurance predictions
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime
import torch

app = Flask(__name__)
CORS(app)

# Load the trained model
print("Loading DQN + XGBoost hybrid model...")
with open('trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
metrics = model_data['metrics']
feature_importance = model_data['feature_importance']

print("Model loaded successfully!")
print(f"Model metrics: {metrics}")

@app.route('/')
def home():
    """Home page with web interface"""
    return render_template('index.html',
                         feature_names=feature_names,
                         metrics=metrics)

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get model information"""
    return jsonify({
        'status': 'success',
        'model_type': 'DQN + XGBoost Hybrid',
        'model_architecture': {
            'dqn': 'Deep Q-Network (3 layers: 256-128-64)',
            'xgboost': 'XGBoost Classifier',
            'feature_extraction': 'DQN extracts 32 features from 85 original features',
            'total_features': 117
        },
        'features_count': len(feature_names),
        'feature_names': feature_names,
        'metrics': metrics,
        'description': 'Hybrid model combining Deep Q-Network with XGBoost for enhanced insurance prediction'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict for a single customer
    Input: JSON with feature values
    Output: Prediction and probability
    """
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Check if all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing features: {list(missing_features)}'
            }), 400

        # Ensure correct order of features
        df = df[feature_names]

        # Make prediction
        prediction = model.predict(df.values)[0]
        probability = model.predict_proba(df.values)[0]

        result = {
            'status': 'success',
            'prediction': int(prediction),
            'prediction_label': 'Will Buy' if prediction == 1 else 'Will Not Buy',
            'probability': {
                'will_not_buy': float(probability[0]),
                'will_buy': float(probability[1])
            },
            'confidence': float(max(probability)),
            'model_type': 'DQN + XGBoost Hybrid',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple customers
    Input: CSV file or JSON array
    Output: Predictions for all customers
    """
    try:
        # Check if file upload or JSON
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400

            # Read CSV file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.txt'):
                df = pd.read_csv(file, sep='\t', header=None)
                df.columns = feature_names
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'File must be CSV or TXT format'
                }), 400

        else:
            # JSON array
            data = request.get_json()
            if not isinstance(data, list):
                return jsonify({
                    'status': 'error',
                    'message': 'Data must be an array of objects'
                }), 400
            df = pd.DataFrame(data)

        # Validate features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing features: {list(missing_features)}'
            }), 400

        # Ensure correct order of features
        df = df[feature_names]

        # Make predictions
        predictions = model.predict(df.values)
        probabilities = model.predict_proba(df.values)

        # Prepare results
        results = []
        for i in range(len(predictions)):
            results.append({
                'index': i,
                'prediction': int(predictions[i]),
                'prediction_label': 'Will Buy' if predictions[i] == 1 else 'Will Not Buy',
                'probability_will_not_buy': float(probabilities[i][0]),
                'probability_will_buy': float(probabilities[i][1]),
                'confidence': float(max(probabilities[i]))
            })

        # Calculate summary statistics
        total_predictions = len(predictions)
        will_buy_count = int(predictions.sum())
        will_not_buy_count = total_predictions - will_buy_count
        avg_probability = float(probabilities[:, 1].mean())

        response = {
            'status': 'success',
            'model_type': 'DQN + XGBoost Hybrid',
            'total_records': total_predictions,
            'summary': {
                'will_buy': will_buy_count,
                'will_not_buy': will_not_buy_count,
                'will_buy_percentage': (will_buy_count / total_predictions) * 100,
                'average_buy_probability': avg_probability
            },
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict_batch/download', methods=['POST'])
def predict_batch_download():
    """
    Predict for multiple customers and return CSV file
    """
    try:
        # Get predictions using the batch predict logic
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.txt'):
                df = pd.read_csv(file, sep='\t', header=None)
                df.columns = feature_names
        else:
            data = request.get_json()
            df = pd.DataFrame(data)

        # Ensure correct order of features
        df_features = df[feature_names]

        # Make predictions
        predictions = model.predict(df_features.values)
        probabilities = model.predict_proba(df_features.values)

        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Prediction'] = predictions
        result_df['Prediction_Label'] = ['Will Buy' if p == 1 else 'Will Not Buy' for p in predictions]
        result_df['Probability_Will_Not_Buy'] = probabilities[:, 0]
        result_df['Probability_Will_Buy'] = probabilities[:, 1]
        result_df['Confidence'] = probabilities.max(axis=1)
        result_df['Model_Type'] = 'DQN + XGBoost Hybrid'

        # Convert to CSV
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        # Create BytesIO object for sending file
        mem = io.BytesIO()
        mem.write(output.getvalue().encode())
        mem.seek(0)

        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'dqn_xgb_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the model"""
    top_n = request.args.get('top_n', default=20, type=int)

    importance_data = feature_importance.head(top_n).to_dict('records')

    return jsonify({
        'status': 'success',
        'top_features': importance_data,
        'description': 'Feature importance from XGBoost model (includes original + DQN features)'
    })

@app.route('/api/example', methods=['GET'])
def get_example():
    """Get example input data"""
    example_data = {feature: 0 for feature in feature_names}
    return jsonify({
        'status': 'success',
        'example': example_data,
        'description': 'Example input with all features set to 0. Replace with actual values.'
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("\n" + "="*80)
    print("DQN + XGBoost Insurance Prediction API Server")
    print("="*80)
    print(f"Model: DQN + XGBoost Hybrid")
    print(f"Features: {len(feature_names)} original + 32 DQN features = 117 total")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print("\nAPI Endpoints:")
    print("- GET  /                           - Web interface")
    print("- GET  /api/info                   - Model information")
    print("- POST /api/predict                - Single prediction (JSON)")
    print("- POST /api/predict_batch          - Batch predictions (JSON/CSV)")
    print("- POST /api/predict_batch/download - Batch predictions (download CSV)")
    print("- GET  /api/feature_importance     - Feature importance")
    print("- GET  /api/example                - Get example input")
    print("\nStarting server on http://localhost:8080")
    print("="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8080)
