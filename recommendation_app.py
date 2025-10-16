#!/usr/bin/env python3
"""
Insurance Cross-Selling Recommendation API
Provides personalized insurance recommendations based on customer profile
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import torch

app = Flask(__name__)
CORS(app)

# Load the multi-insurance model
print("Loading Multi-Insurance Recommendation Model...")
try:
    with open('multi_insurance_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    from multi_insurance_model import MultiInsurancePredictor, INSURANCE_PRODUCTS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = MultiInsurancePredictor(input_dim=43, device=device)
    predictor.load_model('multi_insurance_model.pkl')

    print("Multi-Insurance Model loaded successfully!")
    print(f"Loaded {len(predictor.models)} insurance product models")

except FileNotFoundError:
    print("Model not found. Please train the model first by running:")
    print("python multi_insurance_model.py")
    predictor = None


@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'status': 'success',
        'message': 'Insurance Cross-Selling Recommendation API',
        'endpoints': {
            '/api/recommend': 'POST - Get insurance recommendations for a customer',
            '/api/recommend_batch': 'POST - Get recommendations for multiple customers',
            '/api/products': 'GET - List all insurance products',
            '/api/example': 'GET - Get example customer data'
        }
    })


@app.route('/api/products', methods=['GET'])
def get_products():
    """Get list of all insurance products"""
    products = [
        {'code': code, 'name': name}
        for code, name in INSURANCE_PRODUCTS.items()
    ]

    return jsonify({
        'status': 'success',
        'total_products': len(products),
        'products': products
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Get insurance recommendations for a single customer

    Input JSON:
    {
        "customer_name": "張三",  // optional
        "customer_features": {
            "MOSTYPE": 1,
            "MAANTHUI": 1,
            ... (43 demographic features)
        },
        "owned_insurance": {  // optional, current insurance holdings
            "AWAPART": 1,
            "APERSAUT": 2,
            ...
        }
    }

    Output:
    {
        "customer_name": "張三",
        "owned_insurance": [...],
        "recommendations": [
            {
                "product_code": "LEVEN",
                "product_name": "人壽保險",
                "probability": 0.85,
                "confidence": "high"
            },
            ...
        ]
    }
    """

    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please train the model first.'
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Extract customer info
        customer_name = data.get('customer_name', 'Unknown Customer')
        customer_features = data.get('customer_features', {})
        owned_insurance_data = data.get('owned_insurance', {})

        # Merge owned insurance into customer features
        full_features = {**customer_features, **owned_insurance_data}

        # Validate required features
        missing_features = set(predictor.customer_info_columns) - set(full_features.keys())
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing required features: {list(missing_features)[:10]}...',
                'required_features': predictor.customer_info_columns
            }), 400

        # Get recommendations
        result = predictor.predict_for_customer(full_features, customer_name)

        # Add confidence levels
        for rec in result['recommendations']:
            prob = rec['probability']
            if prob >= 0.7:
                rec['confidence'] = 'high'
            elif prob >= 0.5:
                rec['confidence'] = 'medium'
            else:
                rec['confidence'] = 'low'

        # Format response
        response = {
            'status': 'success',
            'customer_name': result['customer_name'],
            'owned_insurance_count': len(result['owned_insurance']),
            'owned_insurance': result['owned_insurance'],
            'recommendations': result['recommendations'],
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/recommend_batch', methods=['POST'])
def recommend_batch():
    """
    Get recommendations for multiple customers

    Input: CSV file or JSON array with customer data
    Output: Recommendations for each customer
    """

    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please train the model first.'
        }), 500

    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400

            # Read CSV/TXT file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.txt'):
                df = pd.read_csv(file, sep='\t')
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'File must be CSV or TXT format'
                }), 400

        else:
            # Handle JSON array
            data = request.get_json()
            if not isinstance(data, list):
                return jsonify({
                    'status': 'error',
                    'message': 'Data must be an array of customer objects'
                }), 400
            df = pd.DataFrame(data)

        # Check for customer name column
        name_column = None
        for col in ['customer_name', 'name', 'NAME', 'CUSTOMER_NAME']:
            if col in df.columns:
                name_column = col
                break

        # Process each customer
        results = []

        for idx, row in df.iterrows():
            customer_data = row.to_dict()
            customer_name = customer_data.get(name_column, f'Customer_{idx+1}') if name_column else f'Customer_{idx+1}'

            try:
                result = predictor.predict_for_customer(customer_data, customer_name)

                # Add confidence levels
                for rec in result['recommendations']:
                    prob = rec['probability']
                    if prob >= 0.7:
                        rec['confidence'] = 'high'
                    elif prob >= 0.5:
                        rec['confidence'] = 'medium'
                    else:
                        rec['confidence'] = 'low'

                results.append({
                    'customer_name': result['customer_name'],
                    'owned_insurance_count': len(result['owned_insurance']),
                    'owned_insurance': result['owned_insurance'],
                    'top_recommendations': result['recommendations'][:3]  # Top 3
                })

            except Exception as e:
                results.append({
                    'customer_name': customer_name,
                    'error': str(e)
                })

        # Summary statistics
        total_customers = len(results)
        successful_predictions = len([r for r in results if 'error' not in r])

        return jsonify({
            'status': 'success',
            'total_customers': total_customers,
            'successful_predictions': successful_predictions,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/example', methods=['GET'])
def get_example():
    """Get example customer data"""

    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500

    # Example customer data
    example = {
        'customer_name': '張三',
        'customer_features': {col: 0 for col in predictor.customer_info_columns},
        'owned_insurance': {
            'AWAPART': 1,  # Has private third party insurance
            'APERSAUT': 1   # Has car insurance
        }
    }

    # Add some realistic values
    example['customer_features'].update({
        'MOSTYPE': 8,      # Middle class families
        'MAANTHUI': 1,     # 1 house
        'MGEMOMV': 3,      # Average household size
        'MGEMLEEF': 3,     # Age 40-50
        'MOSHOOFD': 3,     # Average Family
        'MHKOOP': 5,       # Home owner
        'MAUT1': 5,        # 1 car
        'MINKGEM': 4,      # Average income
    })

    return jsonify({
        'status': 'success',
        'example': example,
        'description': 'Example customer with demographics and current insurance holdings'
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'models_count': len(predictor.models) if predictor else 0
    })


if __name__ == '__main__':
    if predictor is None:
        print("\n" + "="*80)
        print("WARNING: Model not loaded!")
        print("="*80)
        print("Please train the model first by running:")
        print("  python multi_insurance_model.py")
        print("\nThen restart this API server.")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("Insurance Cross-Selling Recommendation API")
        print("="*80)
        print(f"Model: Multi-Insurance Predictor")
        print(f"Insurance Products: {len(INSURANCE_PRODUCTS)}")
        print(f"Trained Models: {len(predictor.models)}")
        print("\nAPI Endpoints:")
        print("- GET  /                        - API information")
        print("- GET  /api/products            - List all insurance products")
        print("- POST /api/recommend           - Get recommendations for one customer")
        print("- POST /api/recommend_batch     - Get recommendations for multiple customers")
        print("- GET  /api/example             - Get example customer data")
        print("- GET  /api/health              - Health check")
        print("\nStarting server on http://localhost:5000")
        print("="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
