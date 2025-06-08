# find_stations.py
# Searches for NOAA GSOM weather stations in Africa for malaria prediction (SDG 3)
# Output: data/africa_stations.csv

import requests
import pandas as pd
import time

# NOAA API token
TOKEN = 'pYdxrAWVswYOOyLzUuYXwzuQUAHWDWoS'

# API base URL for stations
station_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/stations'

# Headers
headers = {'token': TOKEN}

# Parameters
station_params = {
    'datasetid': 'GSOM',
    'extent': '37,-17,-35,51',  # Africa: North=37°, West=-17°, South=-35°, East=51°
    'limit': 1000
}

# Initialize empty DataFrame
all_stations = pd.DataFrame()

# Handle pagination
offset = 1
while True:
    station_params['offset'] = offset
    response = requests.get(station_url, headers=headers, params=station_params)
    
    if response.status_code == 200:
        stations = response.json()
        if 'results' in stations and stations['results']:
            station_df = pd.DataFrame(stations['results'])
            all_stations = pd.concat([all_stations, station_df], ignore_index=True)
            offset += 1000
            time.sleep(0.2)  # Avoid rate limit
        else:
            break
    else:
        print(f"Error: {response.status_code} - {response.text}")
        break

# Save and inspect
if not all_stations.empty:
    print("\nAvailable Stations in Africa:")
    print(all_stations[['id', 'name', 'latitude', 'longitude']].head(10))
    all_stations.to_csv('data/africa_stations.csv', index=False)
    print(f"Saved {len(all_stations)} stations to data/africa_stations.csv")
else:
    print("No stations found. Try adjusting extent or dataset.")