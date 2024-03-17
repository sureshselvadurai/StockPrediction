import pandas as pd
import os

def generate_report():
    data = pd.read_csv("data_output/model/report/report.csv")
    data = data[data["Tag"] == "Prediction"]

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


if __name__ == '__main__':
    merge_csv_files("data_output/report")

