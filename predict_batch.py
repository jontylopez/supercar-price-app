#!/usr/bin/env python3
"""
Batch prediction CLI for Supercar Price Predictor 2025
Usage: python predict_batch.py --input path/to/supercars_test.csv --output predictions.csv
"""

import argparse
import sys
import pandas as pd
from utils.infer import load_artifacts, predict_df

def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction for supercar prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_batch.py --input supercars_test.csv --output predictions.csv
  python predict_batch.py -i data/test.csv -o results/predictions.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path for predictions'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print(f"🚗 Loading model artifacts...")
        
        # Load artifacts
        preprocessor, model, num_cols, cat_cols = load_artifacts()
        
        if args.verbose:
            print(f"✅ Model loaded successfully")
            print(f"📊 Expected columns: {len(num_cols + cat_cols)} total")
            print(f"   Numeric: {len(num_cols)}")
            print(f"   Categorical: {len(cat_cols)}")
        
        # Read input CSV
        if args.verbose:
            print(f"📁 Reading input file: {args.input}")
        
        df = pd.read_csv(args.input)
        print(f"📊 Loaded {len(df)} cars from {args.input}")
        
        # Check for ID column
        has_id = 'id' in df.columns
        if has_id:
            print(f"🆔 Found ID column in input data")
        else:
            print(f"⚠️  No ID column found, will generate sequential IDs")
        
        # Make predictions
        if args.verbose:
            print(f"🔮 Making predictions...")
        
        predictions_df = predict_df(df, preprocessor, model, num_cols, cat_cols)
        
        # Create output DataFrame
        if has_id:
            output_df = pd.DataFrame({
                'ID': df['id'],
                'price': predictions_df['price']
            })
        else:
            output_df = pd.DataFrame({
                'ID': range(len(df)),
                'price': predictions_df['price']
            })
        
        # Save predictions
        if args.verbose:
            print(f"💾 Saving predictions to: {args.output}")
        
        output_df.to_csv(args.output, index=False)
        
        # Summary statistics
        print(f"✅ Predictions saved to {args.output}")
        print(f"📈 Price Summary:")
        print(f"   Count: {len(output_df)}")
        print(f"   Min: ${output_df['price'].min():,.2f}")
        print(f"   Max: ${output_df['price'].max():,.2f}")
        print(f"   Mean: ${output_df['price'].mean():,.2f}")
        print(f"   Median: ${output_df['price'].median():,.2f}")
        
        if args.verbose:
            print(f"\n🎯 First 5 predictions:")
            print(output_df.head().to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 