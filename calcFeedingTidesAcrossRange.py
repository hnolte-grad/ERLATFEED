import pandas as pd
import numpy as np
import requests

def fetch_noaa_tidal_data2(station, start_date, end_date):
    """
    Fetch tidal data from NOAA API.

    Parameters:
    - station: NOAA station ID (e.g., '9414290' for San Francisco).
    - start_date: Start date in 'YYYY-MM-DD' format.
    - end_date: End date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame with columns ['DT', 'tidalheight'].
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # NOAA API allows a maximum range of 4 days for one-minute water level data
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='4D').tolist()
    if date_ranges[-1] != pd.to_datetime(end_date):
        date_ranges.append(pd.to_datetime(end_date) + pd.Timedelta(days=1))

    all_data = []

    for i in range(len(date_ranges) - 1):
        begin_date = date_ranges[i].strftime('%Y%m%d')
        next_date = (date_ranges[i + 1] - pd.Timedelta(days=1)).strftime('%Y%m%d')

        params = {
            'begin_date': begin_date,
            'end_date': next_date,
            'station': station,
            'product': 'one_minute_water_level',
            'datum': 'MLLW',
            'units': 'metric',
            'time_zone': 'gmt',
            'format': 'json'
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

        try:
            data = response.json().get('data', [])
            if not data:
                raise ValueError("No tidal data returned. Check station ID and date range.")

            all_data.extend(data)

        except Exception as e:
            raise Exception(f"Failed to parse NOAA API response: {e}")

    # Filter out records with invalid or empty tidal heights
    cleaned_data = [item for item in all_data if item['v'].strip() != '']

    tidal_data = pd.DataFrame({
        'DT': pd.to_datetime([item['t'] for item in cleaned_data]),
        'tidalheight': [float(item['v']) * 3.28084 for item in cleaned_data]  # Convert meters to feet
    })
    return tidal_data
def fetch_noaa_tidal_data(station, start_date, end_date):
    """
    Fetch tidal data from NOAA API.

    Parameters:
    - station: NOAA station ID (e.g., '9414290' for San Francisco).
    - start_date: Start date in 'YYYY-MM-DD' format.
    - end_date: End date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame with columns ['DT', 'tidalheight'].
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # NOAA API allows a maximum range of 4 days for one-minute water level data
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='4D').tolist()
    if date_ranges[-1] != pd.to_datetime(end_date):
        date_ranges.append(pd.to_datetime(end_date) + pd.Timedelta(days=1))

    all_data = []

    for i in range(len(date_ranges) - 1):
        begin_date = date_ranges[i].strftime('%Y%m%d')
        next_date = (date_ranges[i + 1] - pd.Timedelta(days=1)).strftime('%Y%m%d')

        for product in ['one_minute_water_level', 'water_level', 'predictions', 'hourly_height']:
            params = {
                'begin_date': begin_date,
                'end_date': next_date,
                'station': station,
                'product': product,
                'datum': 'MLLW',
                'units': 'metric',
                'time_zone': 'gmt',
                'format': 'json'
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Error fetching data with product '{product}': {response.status_code} - {response.text}")
                continue

            try:
                data = response.json().get('data', [])
                if data:
                    all_data.extend(data)
                    break
            except Exception as e:
                print(f"Failed to parse NOAA API response with product '{product}': {e}")
                continue

        else:
            raise Exception("Failed to fetch tidal data with available products.")

    # Filter out records with invalid or empty tidal heights
    cleaned_data = [item for item in all_data if item['v'].strip() != '']

    tidal_data = pd.DataFrame({
        'DT': pd.to_datetime([item['t'] for item in cleaned_data]),
        'tidalheight': [float(item['v']) * 3.28084 for item in cleaned_data]  # Convert meters to feet
    })
    return tidal_data
def calculate_tidal_hours(station, start_date, end_date, threshold):
    """
    Calculate the daily total hours the tidal levels were above and below a given threshold.

    Parameters:
    - station: NOAA station ID (e.g., '9414290' for San Francisco).
    - start_date: Start date in 'YYYY-MM-DD' format.
    - end_date: End date in 'YYYY-MM-DD' format.
    - threshold: Tidal height threshold.

    Returns:
    - result_df: DataFrame with daily total hours above and below the threshold.
    - mean_hours_above: Mean daily hours above the threshold.
    - std_hours_above: Standard deviation of daily hours above the threshold.
    """
    # Fetch tidal data from NOAA
    tidal_data = fetch_noaa_tidal_data(station, start_date, end_date)

    # Ensure data is sorted by time
    tidal_data.sort_values('DT', inplace=True)

    # Validate interval calculation
    tidal_data['interval_seconds'] = tidal_data['DT'].diff().dt.total_seconds().fillna(60)

    # Create flags for whether the tidal height exceeds or is below the threshold
    tidal_data['aboveThreshold'] = tidal_data['tidalheight'] > threshold

    # Determine tidal direction ('in' or 'out')
    tidal_data['tidedirection'] = np.where(tidal_data['aboveThreshold'], 'in', 'out')

    # Identify tidal changes
    tidal_data['tideChange'] = tidal_data['tidedirection'].ne(tidal_data['tidedirection'].shift())

    # Save indices where the tide direction changes
    change_indices = tidal_data.index[tidal_data['tideChange']].tolist()

    # Group by date
    tidal_data['date'] = tidal_data['DT'].dt.normalize()

    # Calculate hours above and below threshold
    daily_hours = tidal_data.groupby(['date', 'aboveThreshold'])['interval_seconds'].sum().unstack(fill_value=0) / 3600

    # Ensure consistent column names
    daily_hours.columns = ['hours_below_threshold', 'hours_above_threshold']

    # Ensure all days in range are included
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    result_df = pd.DataFrame({'date': all_dates})

    # Merge with calculated hours
    result_df = result_df.merge(daily_hours, how='left', left_on='date', right_index=True).fillna(0)

    # Calculate mean and standard deviation for hours above the threshold
    mean_hours_above = result_df['hours_above_threshold'].mean()
    std_hours_above = result_df['hours_above_threshold'].std()

    # Save results to CSV
    output_filename = f"feed-tides-hours_{start_date}_to_{end_date}.csv"
    result_df.to_csv(output_filename, index=False)
    print(f"Saved daily tidal hours to {output_filename}")

    return result_df, mean_hours_above, std_hours_above, change_indices
def calculate_monthly_tidal_stats(result_df, year):
    """
    Calculate mean and standard deviation of hours above the threshold for each month and the year.

    Parameters:
    - result_df: DataFrame with daily tidal hours.
    - year: Year of the data (for filename output).

    Saves:
    - CSV with mean and standard deviation for each month and the whole year.
    """
    result_df['month'] = result_df['date'].dt.month

    monthly_stats = result_df.groupby('month')['hours_above_threshold'].agg(['mean', 'std']).reset_index()
    yearly_mean = result_df['hours_above_threshold'].mean()
    yearly_std = result_df['hours_above_threshold'].std()

    yearly_row = pd.DataFrame({'month': [f'Yearly{year}'], 'mean': [yearly_mean], 'std': [yearly_std]})

    monthly_stats = pd.concat([monthly_stats, yearly_row], ignore_index=True)
    # Insert blank rows where total hours > 24
    overfilled_days = result_df[
        (result_df['hours_above_threshold'] + result_df['hours_below_threshold']) > 24
    ]

    if not overfilled_days.empty:
        blank_rows = pd.DataFrame({
            'date': overfilled_days['date'],
            'hours_below_threshold': np.nan,
            'hours_above_threshold': np.nan
        })
        result_df = pd.concat([result_df, blank_rows], ignore_index=True)

        # Optional: sort again by date
        result_df.sort_values('date', inplace=True)
        result_df.reset_index(drop=True, inplace=True)


    output_filename = f"mean-feed-tides_{year}.csv"
    monthly_stats.to_csv(output_filename, index=False)
    print(f"Saved monthly and yearly tidal stats to {output_filename}")

# Example usage:

result, mean, std, changes = calculate_tidal_hours('9444900', '2014-01-01', '2014-12-31', threshold=4.5)
calculate_monthly_tidal_stats(result, '2014')