import streamlit as st
import pandas as pd
import numpy as np
from utils.infer import load_artifacts, predict_df, predict_single_row
import io

# Page config
st.set_page_config(
    page_title="Supercar Price Predictor 2025",
    page_icon="🚗",
    layout="wide"
)

# Cache the artifacts loading
@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts()
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

# Load artifacts
preprocessor, model, num_cols, cat_cols = get_artifacts()

if preprocessor is None:
    st.error("Failed to load model artifacts. Please ensure preprocessor.joblib, supercar_mlp.keras, and feature_columns.json are in the project root.")
    st.stop()

# Define all expected columns
ALL_COLUMNS = [
    'year', 'brand', 'color', 'carbon_fiber_body', 'engine_config', 'horsepower', 
    'torque', 'weight_kg', 'zero_to_60_s', 'top_speed_mph', 'num_doors', 
    'transmission', 'drivetrain', 'market_region', 'mileage', 'num_owners', 
    'interior_material', 'brake_type', 'tire_brand', 'aero_package', 
    'limited_edition', 'has_warranty', 'last_service_date', 'service_history', 
    'non_original_parts', 'model', 'warranty_years', 'damage', 'damage_cost', 
    'damage_type'
]

# Default values for form (based on real data)
DEFAULT_VALUES = {
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

def main():
    st.title("🚗 Supercar Price Predictor 2025")
    st.markdown("Predict the market value of supercars using our trained ML model")
    
    tab1, tab2 = st.tabs(["Single Car Prediction", "Batch CSV Prediction"])
    
    with tab1:
        st.header("Single Car Price Prediction")
        st.markdown("Fill in the car details below to get a price prediction")
        
        # Create form
        with st.form("car_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                year = st.number_input("Year", min_value=1900, max_value=2025, value=DEFAULT_VALUES['year'])
                brand = st.selectbox("Brand", ['McLaren', 'Ferrari', 'Lamborghini', 'Bugatti', 'Aston Martin', 'Pagani', 'Koenigsegg'], index=0)
                model_name = st.text_input("Model", value=DEFAULT_VALUES['model'])
                color = st.selectbox("Color", ['Black', 'White', 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Silver'], index=1)
                engine_config = st.selectbox("Engine Configuration", ['V8', 'V10', 'V12', 'W16', 'Hybrid', 'Electric'], index=0)
                
                st.subheader("Performance")
                horsepower = st.number_input("Horsepower", min_value=0, value=DEFAULT_VALUES['horsepower'])
                torque = st.number_input("Torque (lb-ft)", min_value=0, value=DEFAULT_VALUES['torque'])
                weight_kg = st.number_input("Weight (kg)", min_value=0, value=DEFAULT_VALUES['weight_kg'])
                zero_to_60_s = st.number_input("0-60 mph (seconds)", min_value=0.1, value=DEFAULT_VALUES['zero_to_60_s'], step=0.1)
                top_speed_mph = st.number_input("Top Speed (mph)", min_value=0, value=DEFAULT_VALUES['top_speed_mph'])
                
                st.subheader("Specifications")
                num_doors = st.number_input("Number of Doors", min_value=1, max_value=5, value=DEFAULT_VALUES['num_doors'])
                transmission = st.selectbox("Transmission", ['automatic', 'manual', 'dual-clutch'], index=0)
                drivetrain = st.selectbox("Drivetrain", ['RWD', 'AWD'], index=0)
                market_region = st.selectbox("Market Region", ['North America', 'Europe', 'Asia', 'Middle East'], index=0)
                
            with col2:
                st.subheader("Condition & History")
                mileage = st.number_input("Mileage", min_value=0, value=DEFAULT_VALUES['mileage'])
                num_owners = st.number_input("Number of Previous Owners", min_value=1, value=DEFAULT_VALUES['num_owners'])
                interior_material = st.selectbox("Interior Material", ['leather', 'alcantara', 'suede', 'carbon_fiber'], index=1)
                brake_type = st.selectbox("Brake Type", ['steel', 'carbon-ceramic'], index=1)
                tire_brand = st.selectbox("Tire Brand", ['Pirelli', 'Michelin', 'Bridgestone', 'Continental', 'Goodyear'], index=0)
                
                st.subheader("Features & Options")
                carbon_fiber_body = st.checkbox("Carbon Fiber Body", value=bool(DEFAULT_VALUES['carbon_fiber_body']))
                aero_package = st.checkbox("Aero Package", value=bool(DEFAULT_VALUES['aero_package']))
                limited_edition = st.checkbox("Limited Edition", value=bool(DEFAULT_VALUES['limited_edition']))
                has_warranty = st.checkbox("Has Warranty", value=bool(DEFAULT_VALUES['has_warranty']))
                non_original_parts = st.checkbox("Non-Original Parts", value=bool(DEFAULT_VALUES['non_original_parts']))
                
                st.subheader("Warranty & Service")
                warranty_years = st.number_input("Warranty Years", min_value=0, value=DEFAULT_VALUES['warranty_years'])
                last_service_date = st.text_input("Last Service Date (YYYY-MM-DD)", value=DEFAULT_VALUES['last_service_date'])
                service_history = st.selectbox("Service History", ['authorized', 'independent', 'none'], index=0)
                
                st.subheader("Damage Information")
                damage = st.checkbox("Has Damage", value=bool(DEFAULT_VALUES['damage']))
                damage_cost = st.number_input("Damage Cost ($)", min_value=0, value=DEFAULT_VALUES['damage_cost'])
                damage_type = st.selectbox("Damage Type", ['none', 'minor', 'moderate', 'major'], index=0)
            
            submitted = st.form_submit_button("🚀 Predict Price")
            
            if submitted:
                # Build the car data dictionary
                car_data = {
                    'year': year,
                    'brand': brand,
                    'color': color,
                    'carbon_fiber_body': int(carbon_fiber_body),
                    'engine_config': engine_config,
                    'horsepower': horsepower,
                    'torque': torque,
                    'weight_kg': weight_kg,
                    'zero_to_60_s': zero_to_60_s,
                    'top_speed_mph': top_speed_mph,
                    'num_doors': num_doors,
                    'transmission': transmission,
                    'drivetrain': drivetrain,
                    'market_region': market_region,
                    'mileage': mileage,
                    'num_owners': num_owners,
                    'interior_material': interior_material,
                    'brake_type': brake_type,
                    'tire_brand': tire_brand,
                    'aero_package': int(aero_package),
                    'limited_edition': int(limited_edition),
                    'has_warranty': int(has_warranty),
                    'last_service_date': last_service_date,
                    'service_history': service_history,
                    'non_original_parts': int(non_original_parts),
                    'model': model_name,
                    'warranty_years': warranty_years,
                    'damage': int(damage),
                    'damage_cost': damage_cost,
                    'damage_type': damage_type
                }
                
                try:
                    # Make prediction
                    predicted_price = predict_single_row(car_data, preprocessor, model, num_cols, cat_cols)
                    
                    # Display result
                    st.success("🎯 Prediction Complete!")
                    st.metric(
                        label="Estimated Price",
                        value=f"${predicted_price:,.2f}",
                        delta=None
                    )
                    
                    # Show car summary
                    st.subheader("Car Summary")
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.write(f"**Brand:** {brand}")
                        st.write(f"**Model:** {model_name}")
                        st.write(f"**Year:** {year}")
                        st.write(f"**Color:** {color}")
                        st.write(f"**Horsepower:** {horsepower:,}")
                    with summary_col2:
                        st.write(f"**Mileage:** {mileage:,} miles")
                        st.write(f"**Transmission:** {transmission}")
                        st.write(f"**Drivetrain:** {drivetrain}")
                        st.write(f"**Weight:** {weight_kg:,} kg")
                        st.write(f"**0-60:** {zero_to_60_s}s")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    with tab2:
        st.header("Batch CSV Prediction")
        st.markdown("Upload a CSV file with multiple cars to get batch predictions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain the same columns as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} cars from CSV")
                
                # Check for required columns
                missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
                if missing_cols:
                    st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
                    st.info("The model will use default values for missing columns")
                
                # Show sample data
                st.subheader("Sample Data (First 5 rows)")
                st.dataframe(df.head(), use_container_width=True)
                
                # Make predictions
                if st.button("🚀 Predict All Prices"):
                    with st.spinner("Making predictions..."):
                        try:
                            # Add missing columns with defaults
                            for col in ALL_COLUMNS:
                                if col not in df.columns:
                                    if col in DEFAULT_VALUES:
                                        df[col] = DEFAULT_VALUES[col]
                                    else:
                                        df[col] = 0
                            
                            # Make predictions
                            predictions_df = predict_df(df, preprocessor, model, num_cols, cat_cols)
                            
                            # Create output DataFrame
                            if 'id' in df.columns:
                                output_df = pd.DataFrame({
                                    'ID': df['id'],
                                    'price': predictions_df['price']
                                })
                            else:
                                output_df = pd.DataFrame({
                                    'ID': range(len(df)),
                                    'price': predictions_df['price']
                                })
                            
                            st.success(f"✅ Predictions complete for {len(df)} cars!")
                            
                            # Show first 20 predictions
                            st.subheader("First 20 Predictions")
                            st.dataframe(output_df.head(20), use_container_width=True)
                            
                            # Show histogram
                            st.subheader("Price Distribution")
                            st.bar_chart(output_df['price'].value_counts(bins=20))
                            
                            # Download button
                            csv_buffer = io.StringIO()
                            output_df.to_csv(csv_buffer, index=False)
                            csv_str = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download predictions.csv",
                                data=csv_str,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {e}")
                            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    main() 