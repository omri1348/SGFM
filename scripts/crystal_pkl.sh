#!/bin/sh

echo "Generating crystal pkls"
uv run scripts/crystal_pkl.py mp_20
# uv run scripts/crystal_pkl.py alex_mp_20

items=('mp_20')
# items=('mp_20' 'alex_mp_20')

for item in "${items[@]}"; do
    echo "Generating crystal pkls for $item"
    uv run scripts/crystal_pkl.py data="$item"
done

