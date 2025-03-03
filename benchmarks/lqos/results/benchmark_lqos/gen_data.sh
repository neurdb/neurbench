#!/bin/bash

# Define a log file
LOG_FILE="process.log"

# Add a note before the first command and log the date/time
echo "Starting first command at $(date)" | tee -a $LOG_FILE
python  -u dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_0/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_1/lineitem.tbl -b 5 -D 0.05 -s 1 >> $LOG_FILE 2>&1

# Add a note before the second command
echo "Starting second command at $(date)" | tee -a $LOG_FILE
python  -u dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_1/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_2/lineitem.tbl -b 5 -D 0.05 -s 1 >> $LOG_FILE 2>&1

# Add a note before the third command
echo "Starting third command at $(date)" | tee -a $LOG_FILE
python -u  dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_2/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_3/lineitem.tbl -b 5 -D 0.05 -s 1 >> $LOG_FILE 2>&1

# Final completion message
echo "All commands completed at $(date)" | tee -a $LOG_FILE
