# Week 2 Assignment: AI for Sustainable Development
# Theme: Predicting Malaria Outbreaks for SDG 3 (Good Health and Well-Being)
# Repository: https://github.com/bravonokoth/Health-AI-System-for-Sustainable-Development
# Team: [AI FOR SOFTWARE ENGINEERS GROUP 69]
# Description: This script implements a supervised learning model to predict malaria cases
# using features like temperature, rainfall, population density, and healthcare access.

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("Environment Setup Complete: Ready for SDG 3 Malaria Prediction")