#!/bin/sh

# Define the set of strings
items=('perov' 'carbon' 'mp_20' 'mpts_52')

# Loop over the set of strings
for item in "${items[@]}"; do
    echo "Processing $item"
    python scripts/data_setup.py data="$item"
    # Add your processing logic here
done