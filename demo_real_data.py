#!/usr/bin/env python3
"""
Demo script showing how to use the real supercar data
This script demonstrates loading and analyzing the real CSV files
"""

import pandas as pd
import numpy as np
from utils.infer import load_artifacts, predict_df

def analyze_real_data():
    """Analyze the real supercar data files."""
    print("🚗 Real Supercar Data Analysis")
    print("=" * 50)
    
    # Load test data
    print("\n📊 Test Data Analysis (supercars_test.csv)")
    test_df = pd.read_csv('supercars_test.csv')
    print(f"   Rows: {len(test_df):,}")
    print(f"   Columns: {len(test_df.columns)}")
    print(f"   Brands: {test_df['brand'].nunique()}")
    print(f"   Models: {test_df['model'].nunique()}")
    print(f"   Year range: {test_df['year'].min()} - {test_df['year'].max()}")
    print(f"   Horsepower range: {test_df['horsepower'].min():.0f} - {test_df['horsepower'].max():.0f}")
    
    # Show brand distribution
    print(f"\n   Brand distribution:")
    brand_counts = test_df['brand'].value_counts()
    for brand, count in brand_counts.head(5).items():
        print(f"     {brand}: {count}")
    
    # Load training data
    print("\n📊 Training Data Analysis (supercars_train.csv)")
    train_df = pd.read_csv('supercars_train.csv')
    print(f"   Rows: {len(train_df):,}")
    print(f"   Columns: {len(train_df.columns)}")
    print(f"   Price range: ${train_df['price'].min():,.2f} - ${train_df['price'].max():,.2f}")
    print(f"   Average price: ${train_df['price'].mean():,.2f}")
    print(f"   Median price: ${train_df['price'].median():,.2f}")
    
    # Show price distribution by brand
    print(f"\n   Average price by brand:")
    brand_prices = train_df.groupby('brand')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for brand, row in brand_prices.head(5).iterrows():
        print(f"     {brand}: ${row['mean']:,.2f} ({row['count']} cars)")
    
    return test_df, train_df

def demo_prediction():
    """Demonstrate prediction on real test data."""
    print("\n🔮 Prediction Demo")
    print("=" * 50)
    
    try:
        # Load model artifacts
        print("Loading model artifacts...")
        preprocessor, model, num_cols, cat_cols = load_artifacts()
        print("✅ Model loaded successfully")
        
        # Load test data
        test_df = pd.read_csv('supercars_test.csv')
        print(f"📊 Loaded {len(test_df)} test cars")
        
        # Make predictions on first 5 cars
        print("\n🎯 Making predictions on first 5 cars...")
        sample_df = test_df.head(5)
        
        predictions_df = predict_df(sample_df, preprocessor, model, num_cols, cat_cols)
        
        # Display results
        print("\n📋 Prediction Results:")
        for i, (_, car) in enumerate(sample_df.iterrows()):
            predicted_price = predictions_df.iloc[i]['price']
            print(f"   {i+1}. {car['year']} {car['brand']} {car['model']}")
            print(f"      Color: {car['color']}, HP: {car['horsepower']}")
            print(f"      Predicted Price: ${predicted_price:,.2f}")
            print()
        
        # Show summary statistics
        print(f"📈 Prediction Summary:")
        print(f"   Min predicted price: ${predictions_df['price'].min():,.2f}")
        print(f"   Max predicted price: ${predictions_df['price'].max():,.2f}")
        print(f"   Mean predicted price: ${predictions_df['price'].mean():,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

def main():
    """Main demo function."""
    print("🚗 Supercar Price Predictor 2025 - Real Data Demo")
    print("=" * 60)
    
    # Analyze real data
    test_df, train_df = analyze_real_data()
    
    # Demo prediction
    success = demo_prediction()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Run Streamlit app: streamlit run app_streamlit.py")
        print("   2. Use batch CLI: python predict_batch.py --input supercars_test.csv --output predictions.csv")
        print("   3. Explore the data in the CSV files")
    else:
        print("\n❌ Demo failed. Please check the setup.")

if __name__ == "__main__":
    main() 