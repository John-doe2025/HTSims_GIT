import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_and_save_solar_data(latitude, longitude, days, output_filename="solar_flux_data.csv"):
    print(f"--- Starting Data Fetch for {days} days at ({latitude}, {longitude}) ---")
    base_url = "https://api.open-meteo.com/v1/forecast"
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    params = {"latitude": latitude, "longitude": longitude, "start_date": start_date, "end_date": end_date, "hourly": "shortwave_radiation", "timezone": "auto"}
    print(f"Querying Open-Meteo API for solar flux from {start_date} to {end_date}...")
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        print("API call successful. Processing data...")
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        start_time_obj = df['time'].iloc[0]
        df['time_seconds'] = (df['time'] - start_time_obj).dt.total_seconds()
        df.rename(columns={"shortwave_radiation": "flux_w_m2"}, inplace=True)
        output_df = df[['time_seconds', 'flux_w_m2']]
        output_df.to_csv(output_filename, index=False)
        print(f"\nSuccess! Data saved to '{output_filename}'")
    except Exception as e:
        print(f"Error: An exception occurred. {e}")

if __name__ == "__main__":
    LATITUDE = 34.1478; LONGITUDE = -118.1445; DAYS_TO_FETCH = 15
    fetch_and_save_solar_data(LATITUDE, LONGITUDE, DAYS_TO_FETCH)
