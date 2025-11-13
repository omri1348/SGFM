#!/bin/bash

items=('perov' 'carbon' 'mp_20' 'mpts_52')

if [ -d "data/alex_mp_20" ] && compgen -G "data/alex_mp_20/*.csv" > /dev/null; then
    # .csv files found, add alex_mp_20 to the list
    items+=("alex_mp_20")
fi

for item in "${items[@]}"; do
    echo "Processing $item"
    uv run scripts/data_setup.py data="$item"
done
