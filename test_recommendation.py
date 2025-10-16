#!/usr/bin/env python3
"""
Test the insurance recommendation system
"""

import pickle
import pandas as pd
import torch
from multi_insurance_model import MultiInsurancePredictor, INSURANCE_PRODUCTS

def test_recommendation():
    """Test the recommendation system with sample data"""

    print("="*80)
    print("Testing Insurance Cross-Selling Recommendation System")
    print("="*80)

    # Load model
    print("\n1. Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = MultiInsurancePredictor(input_dim=43, device=device)
    predictor.load_model('multi_insurance_model.pkl')

    # Create sample customer data
    print("\n2. Creating sample customer data...")
    customer_data = {col: 0 for col in predictor.customer_info_columns}

    # Add realistic customer profile
    customer_data.update({
        'MOSTYPE': 8,      # Middle class families
        'MAANTHUI': 1,     # 1 house
        'MGEMOMV': 3,      # Average household size 3
        'MGEMLEEF': 3,     # Age 40-50
        'MOSHOOFD': 3,     # Average Family
        'MGODRK': 2,       # Some catholic
        'MGODPR': 3,       # Some protestant
        'MRELGE': 5,       # Mostly married
        'MFWEKIND': 4,     # Family with children
        'MOPLMIDD': 4,     # Medium education
        'MBERMIDD': 3,     # Middle management
        'MSKB1': 4,        # Social class B1
        'MHKOOP': 5,       # Home owner
        'MAUT1': 5,        # 1 car
        'MZFONDS': 4,      # National health service
        'MINK4575': 4,     # Income 45-75K
        'MINKGEM': 4,      # Average income
        'MKOOPKLA': 5,     # Purchasing power class 5
    })

    # Add existing insurance (customer already has car insurance and fire insurance)
    customer_data['APERSAUT'] = 1  # Has car insurance
    customer_data['ABRAND'] = 1    # Has fire insurance

    print("\n3. Customer Profile:")
    print(f"   Customer Type: Middle class family")
    print(f"   Age: 40-50 years")
    print(f"   Home: Owner")
    print(f"   Car: 1 car")
    print(f"   Income: 45-75K")
    print(f"   Existing Insurance:")
    print(f"     - PERSAUT (汽車保險)")
    print(f"     - BRAND (火災保險)")

    # Get recommendations
    print("\n4. Getting recommendations...")
    result = predictor.predict_for_customer(customer_data, customer_name="測試客戶 - 張三")

    # Display results
    print("\n" + "="*80)
    print("RECOMMENDATION RESULTS")
    print("="*80)

    print(f"\n客戶名稱: {result['customer_name']}")

    print(f"\n已購買保險 ({len(result['owned_insurance'])} 項):")
    for ins in result['owned_insurance']:
        print(f"  ✓ [{ins['product_code']}] {ins['product_name']}")

    print(f"\n推薦購買的保險 (前 5 項):")
    if len(result['recommendations']) == 0:
        print("  (無推薦項目)")
    else:
        for i, rec in enumerate(result['recommendations'], 1):
            confidence = rec['probability']
            bar = '█' * int(confidence * 20)
            confidence_level = "高" if confidence >= 0.7 else "中" if confidence >= 0.5 else "低"
            print(f"\n  {i}. [{rec['product_code']}] {rec['product_name']}")
            print(f"     購買機率: {confidence:.1%} [{bar:<20}] (信心度: {confidence_level})")

    print("\n" + "="*80)

    # Test with another customer
    print("\n\nTesting with another customer profile...")
    print("="*80)

    customer_data2 = {col: 0 for col in predictor.customer_info_columns}
    customer_data2.update({
        'MOSTYPE': 12,     # Affluent young families
        'MAANTHUI': 1,
        'MGEMOMV': 4,      # Larger household
        'MGEMLEEF': 2,     # Age 30-40
        'MOSHOOFD': 1,     # Successful hedonists
        'MHKOOP': 6,       # Strong home owner
        'MAUT2': 5,        # 2 cars
        'MINK7512': 4,     # Income 75-122K
        'MINKGEM': 5,      # High average income
        'MKOOPKLA': 7,     # High purchasing power
    })

    # Only has basic insurance
    customer_data2['AWAPART'] = 1  # Has third party insurance

    print("\nCustomer Profile:")
    print("   Customer Type: Affluent young family")
    print("   Age: 30-40 years")
    print("   Home: Owner")
    print("   Cars: 2 cars")
    print("   Income: 75-122K (High)")
    print("   Existing Insurance:")
    print("     - WAPART (私人第三方責任險)")

    result2 = predictor.predict_for_customer(customer_data2, customer_name="測試客戶 - 李四")

    print(f"\n客戶名稱: {result2['customer_name']}")
    print(f"\n推薦購買的保險 (前 5 項):")
    for i, rec in enumerate(result2['recommendations'], 1):
        confidence = rec['probability']
        bar = '█' * int(confidence * 20)
        confidence_level = "高" if confidence >= 0.7 else "中" if confidence >= 0.5 else "低"
        print(f"\n  {i}. [{rec['product_code']}] {rec['product_name']}")
        print(f"     購買機率: {confidence:.1%} [{bar:<20}] (信心度: {confidence_level})")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Start the API server: python recommendation_app.py")
    print("2. Test the API endpoints as shown in RECOMMENDATION_GUIDE.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_recommendation()
