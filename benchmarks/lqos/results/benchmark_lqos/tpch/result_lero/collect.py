import re


def extract_id_and_execution_time(input_file, output_file):
    results = []

    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                # Extract the id
                id_match = re.match(r'q(\d+)#####', line)
                if not id_match:
                    print(f"Skipping line (ID not found): {line.strip()}")
                    continue
                id = int(id_match.group(1))

                # Extract the execution time in milliseconds
                execution_time_match = re.search(r'"Execution Time": (\d+\.?\d*)', line)
                if not execution_time_match:
                    print(f"Skipping line (Execution Time not found): {line.strip()}")
                    continue
                execution_time_ms = float(execution_time_match.group(1))
                execution_time_s = execution_time_ms

                # Store the result as a tuple
                results.append((id, execution_time_s))

            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {e}")

    # Sort the results by id
    results.sort(key=lambda x: x[0])

    # Write the sorted results to the output file
    with open(output_file, 'w') as outfile:
        for id1 in range(1, 23):
            is_write = False
            for id2, execution_time_s in results:
                if id2 == id1:
                    outfile.write(f'q{id1}##### {execution_time_s:.3f}\n')
                    is_write = True
                    break
            if not is_write:
                outfile.write(f'q{id1}##### failed\n')

if __name__ == '__main__':
    input_file = 'query_shifts_03/q_test_predicates_merge.test'  # Change to your input file path
    output_file = 'output.txt'  # Change to your desired output file path
    extract_id_and_execution_time(input_file, output_file)
    print(f"Results have been written to {output_file}")
