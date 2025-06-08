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

# --- Load World Bank Data ---
try:
    wb_data = pd.read_csv('data/health_expenditure.csv', skiprows=4)
    # Reshape to long format
    wb_data_melted = pd.melt(
        wb_data,
        id_vars=['Country Name', 'Country Code'],
        value_vars=[str(year) for year in range(2010, 2023)],
        var_name='Year',
        value_name='Health_Expenditure'
    )
    wb_data_melted['Year'] = wb_data_melted['Year'].astype(int)

    # Inspect World Bank
    print("\nWorld Bank Dataset Preview:")
    print(wb_data_melted.head())
    print("\nWorld Bank Columns:", list(wb_data_melted.columns))
    print("\nWorld Bank Missing Values:")
    print(wb_data_melted.isnull().sum())
except FileNotFoundError:
    print("Error: data/health_expenditure.csv not found. Please download from https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS")

# --- Placeholder for NOAA GSOM ---
# Add NOAA query code after running find_stations.py

# Document sources
print("\nData Sources:")
print("- Kaggle: https://www.kaggle.com/datasets/lydia70/malaria-in-africa")
print("- World Bank: https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS")
print("- NOAA GSOM: https://www.ncei.noaa.gov/data/global-summary-of-the-month/")