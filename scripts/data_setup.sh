#!/bin/bash

items=('perov' 'carbon' 'mp_20' 'mpts_52')
# items=('perov' 'carbon' 'mp_20' 'mpts_52' 'alex_mp_20')

for item in "${items[@]}"; do
    echo "Processing $item"
    uv run scripts/data_setup.py data="$item"
done
