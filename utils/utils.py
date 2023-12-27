import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


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
