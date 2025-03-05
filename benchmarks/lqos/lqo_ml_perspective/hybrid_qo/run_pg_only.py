import psycopg2
import os
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run EXPLAIN ANALYZE on SQL queries.")
parser.add_argument("--dbname", type=str, required=True, help="Name of the database to connect to")
args = parser.parse_args()

# Database configuration
PORT = 5432
HOST = "localhost"
USER = "postgres"
PASSWORD = "postgres"
DB = args.dbname
CONNECTION_STR = f"dbname={DB} user={USER} password={PASSWORD} host={HOST} port={PORT}"
TIMEOUT = 30000000

# File with SQL queries
SQL_FILE_PATH = "imdb_q_test_0.txt"  # Replace with the path to your SQL file

# Dictionary to store execution times for each query
execution_times = {}

try:
    # Connect to PostgreSQL database
    connection = psycopg2.connect(CONNECTION_STR)
    connection.autocommit = True
    cursor = connection.cursor()

    # Read the SQL queries from the file
    with open(SQL_FILE_PATH, 'r') as file:
        lines = file.readlines()
    print(lines)
    NUM_EXECUTIONS = 1
    # Extract and execute each query from q1 to q22
    for line in lines:
        query_id, query = line.split("#####")
        query_id = query_id.strip()

        if query_id.startswith("q") and 1 <= int(query_id[1:]) <= 22:
            try:
                for i in range(NUM_EXECUTIONS):
                    explain_query = f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {query}"
                    cursor.execute(explain_query)
                    explain_result = cursor.fetchall()[0][0]
                    execution_time = explain_result[0].get("Execution Time", "N/A")
                    if i == 2:
                        # Store the result in the dictionary
                        execution_times[query_id] = execution_time
            except:
                execution_times[query_id] = "failed"

                # Sort the queries by their ID (q1 to q22)
    for query_id in sorted(execution_times.keys(), key=lambda x: int(x[1:])):
        print(f"Execution Time for {query_id}: {execution_times[query_id]} ms")

except psycopg2.Error as e:
    print("Error: Unable to connect or execute the SQL query")
    print(e)

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
