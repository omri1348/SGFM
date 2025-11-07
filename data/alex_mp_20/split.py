from pathlib import Path
import pandas as pd


train_path = Path(__file__).parent / "from_mattergen/train.csv"
val_path = Path(__file__).parent / "from_mattergen/val.csv"
test_id_path = Path(__file__).parent / "test_ids.txt"

with open(test_id_path, 'r') as file:
    test_ids = [line.strip() for line in file]

train_df = pd.read_csv(train_path, index_col=0)
val_df = pd.read_csv(val_path, index_col=0)

in_test = train_df["material_id"].isin(test_ids)
in_train = ~train_df["material_id"].isin(test_ids)

test_df = train_df[in_test]
new_train_df = train_df[in_train]

new_train_df.to_csv(Path(__file__).parent / "train.csv", index=True)
val_df.to_csv(Path(__file__).parent / "val.csv", index=True)
test_df.to_csv(Path(__file__).parent / "test.csv", index=True)
