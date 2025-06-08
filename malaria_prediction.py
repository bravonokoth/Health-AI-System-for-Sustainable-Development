# Week 2 Assignment: AI for Sustainable Development
# Theme: Predicting Malaria Outbreaks for SDG 3 (Good Health and Well-Being)
# Repository: https://github.com/bravonokoth/Health-AI-System-for-Sustainable-Development
# Team: [AI FOR SOFTWARE ENGINEERS GROUP 69]
# Description: This script implements a supervised learning model to predict malaria cases
# using features like temperature, rainfall, population density, and healthcare access.

# Import libraries
import kagglehub
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# --- Load Kaggle Dataset ---
kaggle_path = kagglehub.dataset_download("lydia70/malaria-in-africa")
kaggle_files = os.listdir(kaggle_path)
print("Kaggle dataset path:", kaggle_path)
print("Kaggle files:", kaggle_files)

# Copy Kaggle CSV to data/
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
kaggle_csv = os.path.join(kaggle_path, 'DatasetAfricaMalaria.csv')
target_csv = os.path.join(data_dir, 'malaria_cases.csv')
shutil.copy(kaggle_csv, target_csv)
print(f"Copied Kaggle CSV to {target_csv}")

# Load Kaggle
malaria_data = pd.read_csv(target_csv)

# Inspect Kaggle
print("\nMalaria Dataset Preview:")
print(malaria_data.head())
print("\nMalaria Columns:", list(malaria_data.columns))
print("\nMalaria Missing Values:")
print(malaria_data.isnull().sum())

# --- Load World Bank Health Expenditure ---
try:
    wb_data = pd.read_csv('data/health_expenditure.csv', skiprows=4)
    wb_data_melted = pd.melt(
        wb_data,
        id_vars=['Country Name', 'Country Code'],
        value_vars=[str(year) for year in range(2010, 2023)],
        var_name='Year',
        value_name='Health_Expenditure'
    )
    wb_data_melted['Year'] = wb_data_melted['Year'].astype(int)

    print("\nWorld Bank Health Expenditure Preview:")
    print(wb_data_melted.head())
    print("\nWorld Bank Columns:", list(wb_data_melted.columns))
    print("\nWorld Bank Missing Values:")
    print(wb_data_melted.isnull().sum())
except FileNotFoundError:
    print("Error: data/health_expenditure.csv not found. Download from https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS")

# --- Load World Bank Climate Data ---
try:
    climate_data = pd.read_csv('data/climate_data.csv')
    print("\nWorld Bank Climate Data Preview:")
    print(climate_data.head())
    print("\nWorld Bank Climate Columns:", list(climate_data.columns))
    print("\nWorld Bank Climate Missing Values:")
    print(climate_data.isnull().sum())
except FileNotFoundError:
    print("Error: data/climate_data.csv not found. Process API_19_DS2_en_csv_v2_4644.csv or download from https://climateknowledgeportal.worldbank.org/download-data")
    climate_data = None

# --- Merge Datasets ---
if climate_data is not None:
    # Merge Kaggle, World Bank Health, and World Bank Climate
    merged_data = malaria_data.merge(
        wb_data_melted[['Country Code', 'Year', 'Health_Expenditure']],
        on=['Country Code', 'Year'],
        how='left'
    ).merge(
        climate_data[['Country Code', 'Year', 'Temperature', 'Precipitation']],
        on=['Country Code', 'Year'],
        how='left'
    )

    # Filter to 2010-2022
    merged_data = merged_data[merged_data['Year'].between(2010, 2022)]

    # Inspect
    print("\nMerged Dataset Preview:")
    print(merged_data[['Country Code', 'Year', 'Malaria cases reported', 'Health_Expenditure', 'Temperature', 'Precipitation']].head())
    print("\nMerged Columns:", list(merged_data.columns))
    print("\nMerged Missing Values:")
    print(merged_data.isnull().sum())

    merged_data.to_csv('data/merged_data.csv', index=False)
else:
    print("No merged dataset created due to missing climate data.")
    merged_data = None

# --- Scikit-learn Modeling ---
if merged_data is not None and not merged_data.empty:
    # Handle missing values
    merged_data = merged_data.dropna(subset=['Malaria cases reported'])  # Drop missing target
    numerical_cols = [
        'Health_Expenditure', 'Temperature', 'Precipitation',
        'Incidence of malaria (per 1,000 population at risk)',
        'People using at least basic drinking water services (% of population)',
        'People using at least basic sanitation services (% of population)'
    ]
    numerical_cols = [col for col in numerical_cols if col in merged_data.columns]
    merged_data[numerical_cols] = merged_data[numerical_cols].fillna(merged_data[numerical_cols].median())

    # Drop sparse columns (>50% missing)
    sparse_cols = [col for col in merged_data.columns if merged_data[col].isnull().sum() > len(merged_data) * 0.5]
    merged_data = merged_data.drop(columns=sparse_cols)

    # Select features and target
    features = numerical_cols
    X = merged_data[features]
    y = merged_data['Malaria cases reported']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print("\nFeature Importance:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    # Save model
    import joblib
    joblib.dump(model, 'data/malaria_model.pkl')
    print("Model saved to data/malaria_model.pkl")
else:
    print("No merged data available for modeling.")

# Document sources
print("\nData Sources:")
print("- Kaggle: https://www.kaggle.com/datasets/lydia70/malaria-in-africa")
print("- World Bank Health: https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS")
print("- World Bank Climate: https://climateknowledgeportal.worldbank.org/download-data")