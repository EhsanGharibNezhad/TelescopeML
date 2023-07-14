#!/bin/bash

# How to Run:
# 1. check the name of the *.py bohb codes
# 2. make sure to have the dir ready for the output
# 3. import the bohb enviroemnt: conda activate bohb
# 4. run: ./submit.sh <DIR_PATH>
# 5. wait until all workers get submitted (~about 3 mins)
# 6. go grap a coffee :)

# Check if the directory parameter is provided

# conda activate bohb

if [ -z "$1" ]; then
    echo "Please provide the directory parameter."
    exit 1
fi

directory="$1"

# Run the first instance of the Python script
python3 Tune_CNN_hyperparameters_BOHB.py  &> "$directory/out1" &
sleep 10  # Add a 10-second delay

# Run the subsequent instances with different arguments
for i in {2..41}; do
    python3 Tune_CNN_hyperparameters_BOHB.py --worker &> "$directory/out$i" &
    sleep 3  # Add a 3-second delay
done

