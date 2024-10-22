import os
import sys
import re


def extract_sql_queries(input_dir, output_file):
    # Compile regex to find files starting with 'test_' followed by an id and ending with '.sql'
    regex = re.compile(r'^test_query_(\d+)\.sql$')

    # List to store extracted queries
    queries = []

    # Traverse directory
    for filename in os.listdir(input_dir):
        # Check if the filename matches the required pattern
        match = regex.match(filename)
        if match:
            id = match.group(1)  # Extract the id
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                sql_query = file.read().strip()
                queries.append(f"q{id}#####{sql_query}")

    # Write all queries to the specified output file
    with open(output_file, 'w') as file:
        for query in queries:
            file.write(query + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    extract_sql_queries(input_dir, output_file)
    print(f"Queries extracted and saved to {output_file}")
