from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pandas as pd


# Columns with single tensors: (name, as_tensor)
TENSOR_COLUMNS = [
    ("anchors", False),
    ("G", True),
    ("G_inv", True),
    ("Point_G", True),
    ("ks", False),
    ("G_inv_permutation", True),
    ("G_permutation", True),
]

# graph_arrays tuple indices
GRAPH_ARRAY_KEYS = [
    "frac_coords", "atom_types", "lengths", "angles",
    "edge_indices", "to_jimages", "num_atoms",
]

# Map numpy dtype to string and back
_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "int16": np.int16,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def flatten_array(x: Any) -> tuple[list | None, list | None, str | None]:
    """Flatten tensor/array to 1D list and return (data, shape, dtype)."""
    if x is None:
        return None, None, None
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist(), list(x.shape), str(x.dtype)
    if isinstance(x, (int, float, np.integer, np.floating)):
        arr = np.asarray(x)
        return arr.ravel().tolist(), list(arr.shape), str(arr.dtype)
    return None, None, None


def unflatten_array(
    data: list | None,
    shape: list | None,
    dtype: str | None = None,
    as_tensor: bool = True,
) -> Any:
    """Reconstruct array from flattened data, shape, and dtype."""
    if data is None or shape is None:
        return None
    if dtype and dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}' in parquet data. Add it to _DTYPE_MAP.")
    np_dtype = _DTYPE_MAP.get(dtype) if dtype else None
    arr = np.array(data, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr) if as_tensor else arr


def convert_for_parquet(records: list[dict]) -> list[dict]:
    """Convert tensor columns to flattened lists with shapes and dtypes."""
    out = []
    for rec in records:
        new_rec = {}
        for k, v in rec.items():
            if k == "graph_arrays" and v is not None:
                for i, gk in enumerate(GRAPH_ARRAY_KEYS):
                    data, shape, dtype = flatten_array(v[i])
                    new_rec[f"graph_{gk}_data"] = data
                    new_rec[f"graph_{gk}_shape"] = shape
                    new_rec[f"graph_{gk}_dtype"] = dtype
            elif k == "wyckoff_ops" and v is not None:
                shapes = []
                dtypes = []
                all_data = []
                for tensor in v:
                    arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
                    shapes.append(list(arr.shape))
                    dtypes.append(str(arr.dtype))
                    all_data.extend(arr.ravel().tolist())
                new_rec["wyckoff_ops_data"] = all_data
                new_rec["wyckoff_ops_shapes"] = shapes
                new_rec["wyckoff_ops_dtypes"] = dtypes
            elif any(k == name for name, _ in TENSOR_COLUMNS):
                data, shape, dtype = flatten_array(v)
                new_rec[f"{k}_data"] = data
                new_rec[f"{k}_shape"] = shape
                new_rec[f"{k}_dtype"] = dtype
            else:
                new_rec[k] = v
        out.append(new_rec)
    return out


def convert_from_parquet(df: pd.DataFrame) -> list[dict]:
    """Reconstruct tensors from flattened data and return list of dicts."""
    df = df.copy()

    # Reconstruct tensor columns
    for name, as_tensor in TENSOR_COLUMNS:
        data_col = f"{name}_data"
        shape_col = f"{name}_shape"
        dtype_col = f"{name}_dtype"
        if data_col in df.columns:
            df[name] = [
                unflatten_array(d, s, dt, as_tensor)
                for d, s, dt in zip(df[data_col], df[shape_col], df.get(dtype_col, [None] * len(df)))
            ]
            drop_cols = [data_col, shape_col]
            if dtype_col in df.columns:
                drop_cols.append(dtype_col)
            df = df.drop(columns=drop_cols)

    # Reconstruct wyckoff_ops
    if "wyckoff_ops_data" in df.columns:
        def rebuild_wyckoff(data, shapes, dtypes):
            if data is None or shapes is None:
                return None
            if dtypes is None:
                dtypes = [None] * len(shapes)
            tensors = []
            offset = 0
            for shape, dtype in zip(shapes, dtypes):
                size = int(np.prod(shape))
                np_dtype = _DTYPE_MAP.get(dtype) if dtype else None
                arr = np.array(data[offset:offset + size], dtype=np_dtype).reshape(shape)
                tensors.append(torch.from_numpy(arr))
                offset += size
            return tensors

        dtypes_col = df.get("wyckoff_ops_dtypes", pd.Series([None] * len(df)))
        df["wyckoff_ops"] = [
            rebuild_wyckoff(d, s, dt)
            for d, s, dt in zip(df["wyckoff_ops_data"], df["wyckoff_ops_shapes"], dtypes_col)
        ]
        drop_cols = ["wyckoff_ops_data", "wyckoff_ops_shapes"]
        if "wyckoff_ops_dtypes" in df.columns:
            drop_cols.append("wyckoff_ops_dtypes")
        df = df.drop(columns=drop_cols)

    # Reconstruct graph_arrays
    graph_cols = [f"graph_{gk}_data" for gk in GRAPH_ARRAY_KEYS]
    if all(c in df.columns for c in graph_cols):
        def rebuild_graph(row):
            arrays = []
            for gk in GRAPH_ARRAY_KEYS:
                data = row[f"graph_{gk}_data"]
                shape = row[f"graph_{gk}_shape"]
                dtype = row.get(f"graph_{gk}_dtype")
                if data is None:
                    arrays.append(None)
                    continue
                np_dtype = _DTYPE_MAP.get(dtype) if dtype else None
                arrays.append(np.array(data, dtype=np_dtype).reshape(shape))
            return tuple(arrays)

        df["graph_arrays"] = df.apply(rebuild_graph, axis=1)
        for gk in GRAPH_ARRAY_KEYS:
            drop_cols = [f"graph_{gk}_data", f"graph_{gk}_shape"]
            if f"graph_{gk}_dtype" in df.columns:
                drop_cols.append(f"graph_{gk}_dtype")
            df = df.drop(columns=drop_cols)

    return df.to_dict("records")


def save_parquet(records: list[dict], path: Path | str) -> None:
    """Save records to parquet with proper array handling."""
    converted = convert_for_parquet(records)
    table = pa.Table.from_pylist(converted)
    pq.write_table(table, str(path))


def load_parquet(path: Path | str) -> list[dict]:
    """Load parquet file and convert arrays back to list of dicts."""
    table = pq.read_table(str(path))
    df = table.to_pandas()
    return convert_from_parquet(df)
