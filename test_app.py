#!/usr/bin/env python3
"""
Simple test script to verify the supercar price predictor components work correctly.
Run this before using the main app to ensure everything is set up properly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ joblib imported successfully")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ tensorflow imported successfully")
        print(f"   TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ tensorflow import failed: {e}")
        return False
    
    return True

def test_utils():
    """Test that the utils modules can be imported and work correctly."""
    print("\n🧪 Testing utils modules...")
    
    try:
        from utils.features import engineer, safe_div, parse_last_service_date
        print("✅ utils.features imported successfully")
    except ImportError as e:
        print(f"❌ utils.features import failed: {e}")
        return False
    
    try:
        from utils.infer import load_artifacts, predict_df, predict_single_row
        print("✅ utils.infer imported successfully")
    except ImportError as e:
        print(f"❌ utils.infer import failed: {e}")
        return False
    
    return True

def test_artifacts():
    """Test that model artifacts can be loaded."""
    print("\n🧪 Testing artifact loading...")
    
    try:
        from utils.infer import load_artifacts
        preprocessor, model, num_cols, cat_cols = load_artifacts()
        print("✅ Model artifacts loaded successfully")
        print(f"   Preprocessor type: {type(preprocessor).__name__}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Numeric columns: {len(num_cols)}")
        print(f"   Categorical columns: {len(cat_cols)}")
        return True
    except Exception as e:
        print(f"❌ Artifact loading failed: {e}")
        return False

def test_prediction():
    """Test that a simple prediction works."""
    print("\n🧪 Testing prediction functionality...")
    
    try:
        from utils.infer import predict_single_row
        
        # Test car data (using real data format)
        test_car = {
            'year': 2020,
            'brand': 'Ferrari',
            'color': 'Red',
            'carbon_fiber_body': 0,
            'engine_config': 'V8',
            'horsepower': 500,
            'torque': 400,
            'weight_kg': 1500,
            'zero_to_60_s': 3.5,
            'top_speed_mph': 200,
            'num_doors': 2,
            'transmission': 'automatic',
            'drivetrain': 'RWD',
            'market_region': 'North America',
            'mileage': 10000,
            'num_owners': 1,
            'interior_material': 'leather',
            'brake_type': 'carbon-ceramic',
            'tire_brand': 'Pirelli',
            'aero_package': 0,
            'limited_edition': 0,
            'has_warranty': 1,
            'last_service_date': '2024-12-01',
            'service_history': 'authorized',
            'non_original_parts': 0,
            'model': 'F8 Tributo',
            'warranty_years': 3,
            'damage': 0,
            'damage_cost': 0,
            'damage_type': 'none'
        }
        
        # Make prediction
        price = predict_single_row(test_car)
        print(f"✅ Prediction successful!")
        print(f"   Test car: {test_car['year']} {test_car['brand']} {test_car['model']}")
        print(f"   Predicted price: ${price:,.2f}")
        
        # Basic validation
        if price > 1000:
            print("✅ Price is above minimum threshold ($1000)")
        else:
            print("⚠️  Price is below expected minimum")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚗 Supercar Price Predictor 2025 - Component Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('preprocessor.joblib'):
        print("❌ Error: preprocessor.joblib not found in current directory")
        print("   Please run this script from the supercar-price-app directory")
        return False
    
    if not os.path.exists('supercar_mlp.keras'):
        print("❌ Error: supercar_mlp.keras not found in current directory")
        print("   Please run this script from the supercar-price-app directory")
        return False
    
    if not os.path.exists('feature_columns.json'):
        print("❌ Error: feature_columns.json not found in current directory")
        print("   Please run this script from the supercar-price-app directory")
        return False
    
    print("✅ Model artifacts found in current directory")
    
    # Run tests
    tests = [
        test_imports,
        test_utils,
        test_artifacts,
        test_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 50)
    if passed == total:
        print(f"🎉 All {total} tests passed! The app is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app_streamlit.py")
        print("   2. Or use CLI: python predict_batch.py --input data.csv --output predictions.csv")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 