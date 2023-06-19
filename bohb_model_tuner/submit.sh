#!/bin/bash

# Check if the directory parameter is provided
if [ -z "$1" ]; then
    echo "Please provide the directory parameter."
    exit 1
fi

directory="$1"

# Run the first instance of the Python script
python3 cnn_bohb_tuning9.py  &> "$directory/out1" &
sleep 10  # Add a 10-second delay

# Run the subsequent instances with different arguments
for i in {2..41}; do
    python3 cnn_bohb_tuning9.py --worker &> "$directory/out$i" &
    sleep 3  # Add a 3-second delay
done

