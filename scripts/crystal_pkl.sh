#!/bin/bash

items=('mp_20')

if [ -d "data/alex_mp_20" ] && compgen -G "data/alex_mp_20/*.csv" > /dev/null; then
    # .csv files found, add alex_mp_20 to the list
    items+=("alex_mp_20")
fi

for item in "${items[@]}"; do
    echo "Generating crystal pkls for $item"
    uv run scripts/crystal_pkl.py "$item"
done
