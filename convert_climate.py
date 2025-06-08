import pandas as pd

# Load CSV
df = pd.read_csv('data/climate_data_raw.csv', skiprows=4)

# Filter temperature and precipitation
temp_df = df[df['Indicator Name'].str.contains('Temperature', case=False, na=False)]
precip_df = df[df['Indicator Name'].str.contains('Precipitation', case=False, na=False)]

# Melt years to long format
temp_melted = pd.melt(
    temp_df,
    id_vars=['Country Code'],
    value_vars=[str(year) for year in range(2010, 2023)],
    var_name='Year',
    value_name='Temperature'
)
precip_melted = pd.melt(
    precip_df,
    id_vars=['Country Code'],
    value_vars=[str(year) for year in range(2010, 2023)],
    var_name='Year',
    value_name='Precipitation'
)

# Merge
climate_data = temp_melted.merge(
    precip_melted[['Country Code', 'Year', 'Precipitation']],
    on=['Country Code', 'Year'],
    how='inner'
)
climate_data['Year'] = climate_data['Year'].astype(int)

# Save
climate_data.to_csv('data/climate_data.csv', index=False)
print("Saved to data/climate_data.csv")
print(climate_data.head())