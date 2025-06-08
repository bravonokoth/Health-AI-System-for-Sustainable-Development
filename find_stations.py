# find_stations.py
# Searches for NOAA weather stations in Africa for malaria prediction (SDG 3)
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

# African countries (FIPS codes)
african_countries = [
    'FIPS:KE',  # Kenya
    'FIPS:NG',  # Nigeria
    'FIPS:UG',  # Uganda
    'FIPS:ZA',  # South Africa
    'FIPS:GH'   # Ghana
]

# Configurations to try
configs = [
    {'datasetid': 'GSOM', 'extent': '40,-25,-40,60', 'desc': 'Africa GSOM', 'startdate': '2010-01-01', 'enddate': '2022-12-31'},
    {'datasetid': 'GHCND', 'extent': '40,-25,-40,60', 'desc': 'Africa GHCND', 'startdate': '2010-01-01', 'enddate': '2022-12-31'},
    {'datasetid': 'GHCND', 'locationcategoryid': 'ST', 'desc': 'African Countries GHCND', 'startdate': '2010-01-01', 'enddate': '2022-12-31'},
]
for country in african_countries:
    configs.append({'datasetid': 'GHCND', 'locationid': country, 'desc': f'{country} GHCND', 'startdate': '2010-01-01', 'enddate': '2022-12-31'})

for config in configs:
    print(f"\nTrying: {config['desc']} (dataset: {config['datasetid']})")
    # Parameters
    station_params = {
        'datasetid': config['datasetid'],
        'limit': 1000,
        'startdate': config['startdate'],
        'enddate': config['enddate']
    }
    if 'extent' in config:
        station_params['extent'] = config['extent']
    if 'locationid' in config:
        station_params['locationid'] = config['locationid']
    if 'locationcategoryid' in config:
        station_params['locationcategoryid'] = config['locationcategoryid']

    # Initialize empty DataFrame
    all_stations = pd.DataFrame()

    # Handle pagination
    offset = 1
    while True:
        station_params['offset'] = offset
        try:
            response = requests.get(station_url, headers=headers, params=station_params)
            response.raise_for_status()
            stations = response.json()
            results = stations.get('results', [])
            print(f"Offset {offset}: {len(results)} stations found")
            print(f"Raw response sample: {results[:2] if results else 'Empty'}")
            
            if results:
                station_df = pd.DataFrame(results)
                all_stations = pd.concat([all_stations, station_df], ignore_index=True)
                offset += 1000
                time.sleep(0.2)
            else:
                print("No more results.")
                break
        except requests.RequestException as e:
            print(f"Error: {e} - {response.text if 'response' in locals() else ''}")
            break

    # Save and inspect
    if not all_stations.empty:
        print(f"\nAvailable Stations ({config['desc']}):")
        print(all_stations[['id', 'name', 'latitude', 'longitude']].head(10))
        csv_name = f"data/africa_stations_{config['datasetid'].lower()}_{config['desc'].lower().replace(' ', '_').replace('fips:', '')}.csv"
        all_stations.to_csv(csv_name, index=False)
        print(f"Saved {len(all_stations)} stations to {csv_name}")
        break
    else:
        print(f"No stations found for {config['desc']}.")

print("\nIf no stations were found, consider World Bank climate data: https://climateknowledgeportal.worldbank.org/download-data")