import re

# Define the input log file and SQL range
input_file = 'test__bao_data0'  # Replace with your actual log file path
output_file = 'execution_times_output.txt'

# Dictionary to store execution times for each SQL file (for '2,')
execution_times = {}

# Regular expression to match lines that contain ', 2,' and capture the SQL file and execution time
# The prefix can be anything (including no hint, SET ..., or x)
log_pattern = re.compile(r".*, 2, .*, (/.*?q_test_0/(\d+)\.sql), .*, ([\d.]+), Bao")

# Read the input log file
with open(input_file, 'r') as file:
    for line in file:
        match = log_pattern.match(line.strip())
        if match:
            sql_file = match.group(1)  # Full SQL file path
            sql_number = int(match.group(2))  # Extract SQL file number (e.g., 1, 2, 3, etc.)
            execution_time = float(match.group(3))  # Extract execution time (in milliseconds)
            execution_times[sql_number] = (sql_file, execution_time)

# Sort by SQL file number and write the results
with open(output_file, 'w') as output:
    for sql_number in sorted(execution_times.keys()):
        sql_file, execution_time = execution_times[sql_number]
        output.write(f"{sql_file}: {execution_time:.3f} ms\n")

print(f"Execution times for lines with '2,' extracted and written to {output_file} in order.")
