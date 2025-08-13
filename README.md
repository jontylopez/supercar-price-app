# 🚗 Supercar Price Predictor 2025

A lightweight prediction app for estimating supercar market values using a trained ML model.

## 🏗️ Project Structure

```
supercar-price-app/
├─ app_streamlit.py           # Streamlit UI (single car + batch CSV)
├─ predict_batch.py           # CLI for batch predictions
├─ utils/
│  ├─ features.py             # Feature engineering functions
│  └─ infer.py                # Model loading and prediction functions
├─ requirements.txt           # Dependencies
└─ README.md                  # This file
```

## 📋 Prerequisites

- Python 3.10+
- The following model artifacts must be in the project root:
  - `preprocessor.joblib` (sklearn ColumnTransformer)
  - `supercar_mlp.keras` (Keras model)
  - `feature_columns.json` (feature column definitions)

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Run Batch CLI

```bash
python predict_batch.py --input supercars_test.csv --output predictions.csv
```

### 5. Demo Real Data

```bash
python demo_real_data.py
```

**Note:** The project includes real supercar data:
- `supercars_train.csv` - Training data with prices (2002 cars)
- `supercars_test.csv` - Test data for predictions (502 cars)

## 🎯 Features

### Streamlit App

**Tab 1: Single Car Prediction**
- Form inputs for all car specifications
- Real-time price prediction
- Car summary display
- Handles missing data gracefully

**Tab 2: Batch CSV Prediction**
- Upload CSV files for bulk predictions
- Preview first 20 predictions
- Price distribution visualization
- Download predictions as CSV

### Batch CLI

- Command-line interface for automation
- Supports verbose output with `--verbose` flag
- Generates ID column if missing
- Comprehensive error handling

## 📊 Expected Input Schema

The model expects these columns (order doesn't matter):

**Numeric Features:**
- `year`, `horsepower`, `torque`, `weight_kg`, `zero_to_60_s`, `top_speed_mph`
- `num_doors`, `mileage`, `num_owners`, `warranty_years`
- `damage_cost`, `damage`

**Categorical Features:**
- `brand` (McLaren, Ferrari, Lamborghini, Bugatti, Aston Martin, Pagani, Koenigsegg)
- `color` (Black, White, Red, Blue, Green, Yellow, Orange, Silver)
- `engine_config` (V8, V10, V12, W16, Hybrid, Electric)
- `transmission` (automatic, manual, dual-clutch)
- `drivetrain` (RWD, AWD)
- `market_region` (North America, Europe, Asia, Middle East)
- `interior_material` (leather, alcantara, suede, carbon_fiber)
- `brake_type` (steel, carbon-ceramic)
- `tire_brand` (Pirelli, Michelin, Bridgestone, Continental, Goodyear)
- `service_history` (authorized, independent, none)
- `model`, `damage_type`

**Boolean Features:**
- `carbon_fiber_body`, `aero_package`, `limited_edition`
- `has_warranty`, `non_original_parts`

**Date Features:**
- `last_service_date` (YYYY-MM-DD format)

## 🔧 Configuration

### Model Artifacts

Place these files in the project root:
- `preprocessor.joblib` - Feature preprocessing pipeline
- `supercar_mlp.keras` - Trained neural network model
- `feature_columns.json` - Feature column definitions

### Environment Variables

No environment variables required. The app automatically detects and loads artifacts from the current directory.

## 🧪 Testing

### Smoke Test

Test the prediction function with a simple example:

```python
from utils.infer import predict_single_row

# Test car data
test_car = {
    'year': 2023,
    'brand': 'McLaren',
    'color': 'Black',
    'carbon_fiber_body': 0,
    'engine_config': 'V8',
    'horsepower': 1000,
    'torque': 800,
    'weight_kg': 1500,
    'zero_to_60_s': 3.0,
    'top_speed_mph': 250,
    'num_doors': 2,
    'transmission': 'automatic',
    'drivetrain': 'RWD',
    'market_region': 'North America',
    'mileage': 15000,
    'num_owners': 2,
    'interior_material': 'alcantara',
    'brake_type': 'carbon-ceramic',
    'tire_brand': 'Pirelli',
    'aero_package': 0,
    'limited_edition': 0,
    'has_warranty': 1,
    'last_service_date': '2024-12-01',
    'service_history': 'authorized',
    'non_original_parts': 0,
    'model': '600LT',
    'warranty_years': 2,
    'damage': 0,
    'damage_cost': 0,
    'damage_type': 'none'
}

# Test prediction
price = predict_single_row(test_car)
print(f"Predicted price: ${price:,.2f}")
assert price > 1000, "Price should be greater than $1000"
```

## 🚨 Troubleshooting

### Common Issues

1. **Model artifacts not found**
   - Ensure `preprocessor.joblib`, `supercar_mlp.keras`, and `feature_columns.json` are in the project root
   - Check file permissions

2. **Import errors**
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Memory issues**
   - The app uses CPU-only TensorFlow by default
   - For large datasets, consider using the CLI instead of Streamlit

4. **Missing columns in CSV**
   - The app will warn about missing columns
   - Default values will be used for missing features

### Performance Tips

- Use `@st.cache_resource` for artifact loading (already implemented)
- For large batch predictions, use the CLI instead of Streamlit
- The model is optimized for CPU inference

## 📝 License

This project is part of the "Predict Supercars Prices 2025" assignment.

## 🤝 Contributing

This is a standalone prediction app. For model training or modifications, refer to the main project repository. 