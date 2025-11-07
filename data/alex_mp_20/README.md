# Alex-MP-20

Alex-MP-20 is a large dataset and is not bundled with our release. However, it is not complex to download and process.

1. You need to download the files from [Mattergen](https://github.com/microsoft/mattergen?tab=readme-ov-file#train-mattergen-yourself). Save the csvs into `${MATTERGEN_ALEX_MP_20_DIR}`.

2. We split the data using the same split as [OMatG](https://github.com/FERMat-ML/OMatG?tab=readme-ov-file#included-datasets). Those indices are included with this repository in the file `data/alex_mp_20/test_ids.txt`.
```
# run commands from the SGFM root directory
mkdir data/alex_mp_20/from_mattergen
cp ${MATTERGEN_ALEX_MP_20_DIR}/train.csv ${MATTERGEN_ALEX_MP_20_DIR}/val.csv data/alex_mp_20/from_mattergen
uv run data/alex_mp_20/split.py
```
3. In the file `README.md`, there is a section called "Data Setup." Each of those scripts have a line that you can uncomment so `alex_mp_20` is included in the processing. Uncomment those lines and rerun the scripts!
