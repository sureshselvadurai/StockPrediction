import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import yfinance as yf
import csv


def write_dict_to_csv(my_dict, file_path):
    # Extract keys and values from the dictionary
    keys = list(my_dict.keys())
    values = list(my_dict.values())

    # Write the dictionary to the CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerow(values)


def concatenate_dataframes(changed_rows, not_common_df):
    """
    Concatenate two DataFrames while handling empty or all-NA entries.

    Parameters:
    - changed_rows: DataFrame
    - not_common_df: DataFrame

    Returns:
    - changes: DataFrame
    """

    changed_rows = changed_rows.fillna("NA")
    not_common_df = not_common_df.fillna("NA")

    if not changed_rows.empty and not not_common_df.empty:
        # Both DataFrames are non-empty, concatenate them
        changes = pd.concat([changed_rows, not_common_df])
    elif not changed_rows.empty:
        # Only changed_rows is non-empty, no need to concatenate
        changes = changed_rows.copy()
    elif not not_common_df.empty:
        # Only not_common_df is non-empty, no need to concatenate
        changes = not_common_df.copy()
    else:
        # Both DataFrames are empty
        changes = pd.DataFrame(columns=changed_rows.columns)

    return changes


def generate_close_timestamps(start_date, end_date_str, step_hours=24):
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S%z')

    if isinstance(end_date_str, datetime):
        end_date_str = end_date_str.strftime('%Y-%m-%d %H:%M:%S%z')

    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S%z')

    current_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S%z')
    timestamps = []

    while current_date <= end_date:
        timestamps.append(current_date.strftime('%Y-%m-%d %H:%M:%S%z'))
        current_date += timedelta(hours=step_hours)

    return timestamps


def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        data_x.append(a)
        data_y.append(dataset[i + look_back, -1])
    return np.array(data_x), np.array(data_y)


def str_list(features, target=False):
    out = []
    features = [item[0] for item in features]
    features = [item[0] for item in features] if target else features
    for feature in features:
        out.append(str(feature).replace("\n", ""))
    return out


def move_column_to_last(df, col):
    if col in df.columns:
        return df[list(df.columns.difference([col])) + [col]]
    else:
        return df


def get_stock_data_yf(end_date, start_date, symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data


def clear_folder(folder_path):
    try:
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                clear_folder(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def merge_csv_files(directory_path):
    # Get a list of all CSV files in the specified directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    # Check if there are any CSV files in the directory
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Create an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Iterate through each CSV file and merge its data into the main DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, df], ignore_index=True)

    # Write the merged data to a new CSV file
    merged_data.to_csv(f'{directory_path}/report/report.csv', index=False)


def round_psi(value):
    value = float(value)
    rounded_value = round(value, 2)
    return '' if rounded_value < 0.01 else rounded_value
