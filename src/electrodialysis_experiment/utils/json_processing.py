import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


CASE_RE = re.compile(r"^case_(\d+)$")


def predicted_values_to_arrays(
    data_or_path: Union[str, Path, list],
    *,
    predicted_key: str = "predicted_values",
    sample_id_key: str = "ti_param_sample_id",
    element_keys: Optional[List[str]] = None,
    fill_value: float = np.nan,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """
    Convert predicted_values in JSON to element-wise NumPy arrays.

    Returns:
        arrays_by_element : dict[str, np.ndarray]  -> (n_samples, n_cases)
        sample_ids        : list[str]               row order
        case_names        : list[str]               column order
    """

    # Load JSON if a path is given
    if isinstance(data_or_path, (str, Path)):
        with open(data_or_path, "r") as f:
            data = json.load(f)
    else:
        data = data_or_path

    # Sample IDs (row order)
    sample_ids = [rec[sample_id_key] for rec in data]
    n = len(sample_ids)

    # Collect and sort case names numerically
    case_set = set()
    for rec in data:
        case_set.update(rec[predicted_key].keys())

    def case_sort_key(name: str):
        m = CASE_RE.match(name)
        return int(m.group(1)) if m else name

    case_names = sorted(case_set, key=case_sort_key)
    m = len(case_names)

    # Infer elements if not provided
    if element_keys is None:
        elem_set = set()
        for rec in data:
            for case_dict in rec[predicted_key].values():
                elem_set.update(case_dict.keys())
        element_keys = sorted(elem_set)

    # Allocate arrays
    arrays = {elem: np.full((n, m), fill_value, dtype=float) for elem in element_keys}

    case_index = {c: j for j, c in enumerate(case_names)}

    # Fill arrays
    for i, rec in enumerate(data):
        for c, case_dict in rec[predicted_key].items():
            j = case_index[c]
            for elem in element_keys:
                if elem in case_dict:
                    arrays[elem][i, j] = case_dict[elem]

    return arrays, sample_ids, case_names
