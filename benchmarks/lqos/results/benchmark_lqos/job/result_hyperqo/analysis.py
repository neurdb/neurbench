import pandas as pd
import re


def custom_sort_key(query_ident):
    # Split the query_ident into two parts: numeric and alphabetic
    match = re.match(r'(\D+)(\d+)(\D*)', query_ident)
    if match:
        prefix, number, suffix = match.groups()
        return (prefix, int(number), suffix)
    else:
        return (query_ident, 0, '')


def sort_and_display(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Print available columns to check if execution_time exists
    print("Available columns:", df.columns)

    # Filter only rows where query_ident starts with "test_"
    df_filtered = df[df["query_ident"].str.startswith("test_")]

    # Sort based on the custom key
    df_sorted = df_filtered.sort_values(by='query_ident', key=lambda col: col.map(custom_sort_key))

    # Create a new DataFrame
    new_df = pd.DataFrame()

    # Add 'name' column from sorted 'query_ident'
    new_df["name"] = df_sorted["query_ident"]
    new_df["execution_time"] = df_sorted["hinter_latency"] * 1000
    # new_df["pg_execution_time"] = df_sorted["pg_latency"] * 1000

    return new_df


# Example usage
file_path = 'query_shifts_05/q_test_join_merge.test'  # Replace with your actual file path
sorted_df = sort_and_display(file_path)

# If you want to save the sorted output to a new CSV file
sorted_df.to_csv('sorted_execution_times.csv', index=False)
