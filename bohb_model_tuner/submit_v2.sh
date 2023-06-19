#!/bin/bash

python << EOF
import concurrent.futures
import time

# Check if the directory parameter is provided
#if [ -z "$1" ]; then
#    echo "Please provide the directory parameter."
#    exit 1
#fi

directory="$1"


# Run the first instance of the Python script
python3 cnn_bohb_tuning8.py &> "$directory/out1" &

#start = time.perf_counter()



def do_something(seconds):
    #for i in {2..41}; do
    python3 cnn_bohb_tuning8.py --worker &> "$directory/out$i" &
    #sleep 3  # Add a 3-second delay

    #print(f'Sleeping {seconds} second(s)...')
    #time.sleep(seconds)
    #return f'Done Sleeping...{seconds}'

# Specify the number of CPUs to be used
num_cpus=42  # Set the desired number of CPUs

with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)

#finish = time.perf_counter()

#print(f'Finished in {round(finish-start, 2)} second(s)')
EOF





