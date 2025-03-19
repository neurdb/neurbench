import os
import json
import argparse


def read_sql_files(directory):
    sql_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".sql"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                sql_content = file.read()
                query_id = "train_" + os.path.splitext(filename)[0]
                sql_data.append([sql_content, query_id, [-1, False]])
    return sql_data


def write_json(output_path, data):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Read SQL files and output JSON.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing the .sql files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")

    # Parse the arguments
    args = parser.parse_args()

    # Process the SQL files and write to JSON
    sql_data = read_sql_files(args.input_folder)
    write_json(args.output_json, sql_data)
    print(f"JSON file has been created at: {args.output_json}")


if __name__ == "__main__":
    main()
