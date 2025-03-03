import pandas as pd
import argparse
import re


def sort_sql_ids(sql_id):
    # Split the SQL ID into numeric and alphabetic parts for proper sorting
    match = re.match(r"(\d+)([a-z]+)", sql_id)
    if match:
        return int(match.group(1)), match.group(2)
    return sql_id


def process_file(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file, header=None, sep=';')
    df.columns = ['query_ident', 'inference_time', 'planning_time', 'execution_time']

    # Sort based on SQL ID in the order you want (numeric part first, then alphabetic)
    df['sort_key'] = df['query_ident'].apply(sort_sql_ids)
    df_sorted = df.sort_values('sort_key').drop(columns=['sort_key'])

    # Write the sorted results to the output CSV file
    df_sorted[['query_ident', 'execution_time']].to_csv(output_file, index=False, header=['name', 'execution_time'])


if __name__ == "__main__":
    # Setup argparse to accept input and output file arguments
    parser = argparse.ArgumentParser(description='Process SQL data and output key-value pairs')
    parser.add_argument('--input_file', type=str,
                        default="query_shift_05/2025_02_26__170712__Neo_DB2_SMALL_SET_JOIN_SHIFT_Train05__plan_and_execute.txt")
    parser.add_argument('--output_file', type=str, default="sorted_execution_times.csv")
    args = parser.parse_args()

    # Process the input file and write to the output file
    process_file(args.input_file, args.output_file)
