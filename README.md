# Health-AI-System-for-Sustainable-Development
## Overview
Predicts malaria outbreaks for SDG 3.

## Project Structure

Health-AI-System-for-Sustainable-Development/
├── data/
│   ├── malaria_cases.csv
│   ├── health_expenditure.csv
│   ├── africa_stations_*.csv  # Pending
│   ├── climate_data.csv  # Optional fallback
├── malaria_prediction.py
├── find_stations.py
├── README.md

## Setup
- Install: `pip install kagglehub pandas numpy requests`
- Run: `python malaria_prediction.py`

## Dataset
- Sources:
  - Kaggle: https://www.kaggle.com/datasets/lydia70/malaria-in-africa
  - World Bank: https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS
  - NOAA GSOM: https://www.ncei.noaa.gov/data/global-summary-of-the-month/

  ## Outputs
- `data/climate_data.csv`: Temperature and precipitation
- `data/merged_data.csv`: Merged dataset
- `data/malaria_model.pkl`: Random Forest model