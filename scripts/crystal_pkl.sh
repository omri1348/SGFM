#!/bin/sh

items=('mp_20')
# items=('mp_20' 'alex_mp_20')

for item in "${items[@]}"; do
    echo "Generating crystal pkls for $item"
    uv run scripts/crystal_pkl.py data="$item"
done

