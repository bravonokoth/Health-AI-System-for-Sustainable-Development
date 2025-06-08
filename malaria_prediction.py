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

# Set random seed
np.random.seed(42)

# --- Load Kaggle Dataset ---
kaggle_path = kagglehub.dataset_download("lydia70/malaria-in-africa")
print("Kaggle path:", kaggle_path)
kaggle_files = os.listdir(kaggle_path)
print("Kaggle files:", kaggle_files)

csv_file = os.path.join(kaggle_path, 'Dataset_malaria.csv')  # Update filename
malaria_data = pd.read_csv(csv_file)

# --- Load World Bank Data ---
wb_data = pd.read_csv('data/health_expenditure.csv', skiprows=4)

# Reshape
wb_data_melted = pd.melt(
    wb_data,
    id_vars=['Country Name', 'Country Code'],
    value_vars=[str(year) for year in range(2010, 2023)],
    var_name='Year',
    value_name='Health_Expenditure'
)
wb_data_melted['Year'] = wb_data_melted['Year'].astype(int)

# --- Placeholder for NOAA GSOM ---
# Add Step 2 code here after running find_stations.py

# Inspect Kaggle
print("\nMalaria Dataset Preview:")
print(malaria_data.head())
print("\nMalaria Columns:", list(malaria_data.columns))