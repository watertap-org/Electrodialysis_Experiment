import pandas as pd
import re
from dataclasses import dataclass
from typing import Literal, Sequence, Optional

ComponentKind = Literal["variable", "parameter", "expression", "constraint"]

_KIND_TO_KEY = {
    "variable": "variables",
    "parameter": "parameters",
    "expression": "expressions",
    "constraint": "constraints",
}

@dataclass(frozen=True)
class SearchResult:
    """
    Efficient container for search output.

    - matches: the filtered rows (always includes component + index + other columns)
    - by_component: grouped view {component_name: sub-DataFrame}
    - kind: what table was searched
    - comp_keywords: comp_keywords used (AND semantics)
    - index_keywords: index_keywords used (AND semantics)
    """
    kind: ComponentKind
    comp_keywords: tuple[str, ...]
    matches: pd.DataFrame
    by_component: dict[str, pd.DataFrame]
    index_keywords: Optional[tuple[str, ...]] = None

    @property
    def n_rows(self) -> int:
        return int(len(self.matches))

    @property
    def n_components(self) -> int:
        return int(len(self.by_component))

    @property
    def case(self) -> int:
        """
        0: no matches
        1: exactly one row
        2: multiple rows, single component (same component name, different index)
        3: multiple rows across multiple components and/or indices
        """
        if self.n_rows == 0:
            return 0
        if self.n_rows == 1:
            return 1
        if self.n_components == 1:
            return 2
        return 3

def _table_for_kind(kind: ComponentKind) -> str:
    try:
        return _KIND_TO_KEY[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {sorted(_KIND_TO_KEY)}; got {kind!r}")

def search_components(
    h5_path: str,
    kind: ComponentKind,
    comp_keywords: Sequence[str],
    index_keywords: Sequence[str]=None,
    *,
    match_case: bool = False,
    regex: bool = False,
) -> SearchResult:
    """
    Search snapshot entries by AND-ing multiple comp_keywords against the 'component' column.

    Semantics:
      - A row matches if its component string contains ALL comp_keywords (AND) and ALL index_keywords (AND).
      - comp_keywords can be any length (>=1 recommended).
      - index_keywords can be any length.
      - If regex=False: comp_keywords and index_keywords are treated as literal substrings.
      - If regex=True: each keyword is a regex pattern, AND-ed.

    Returns:
      SearchResult with:
        - matches DataFrame
        - by_component dict (component -> DataFrame)
        - case classification (0/1/2/3)
    """
    if not comp_keywords:
        raise ValueError("comp_keywords must be a non-empty sequence of strings")

    # Load the relevant table
    table = _table_for_kind(kind)
    df = load_mod_hdf(h5_path, table)

    if "component" not in df.columns or "index" not in df.columns:
        raise ValueError(f"Table {table!r} missing required columns 'component' and/or 'index'")

    comp = df["component"].astype(str)
    

    # Build AND mask across all comp_keywords
    mask = pd.Series(True, index=df.index)

    if regex:
        # AND regex patterns
        for kw in comp_keywords:
            mask &= comp.str.contains(kw, case=match_case, regex=True, na=False)
    else:
        # AND literal substring matches
        for kw in comp_keywords:
            pattern = re.escape(kw)
            mask &= comp.str.contains(pattern, case=match_case, regex=True, na=False)

    out = df.loc[mask]

    if index_keywords:
        idx = out["index"].astype(str)
        # Build AND mask across all index_keywords
        idx_mask = pd.Series(True, index=out.index)

        if regex:
            # AND regex patterns
            for kw in index_keywords:
                idx_mask &= idx.str.contains(kw, case=match_case, regex=True, na=False)
        else:
            # AND literal substring matches
            for kw in index_keywords:
                pattern = re.escape(kw)
                idx_mask &= idx.str.contains(pattern, case=match_case, regex=True, na=False)

        out = out.loc[idx_mask]
    

    # Build grouped view for case 2
    by_comp = {name: g.reset_index(drop=True) for name, g in out.groupby("component", sort=False)}

    return SearchResult(
        kind=kind,
        comp_keywords=tuple(comp_keywords),
        matches=out.reset_index(drop=True),
        by_component=by_comp,
    )


def list_hdf_keys(h5_path: str):
    """Show what tables exist in the HDF5 file."""
    with pd.HDFStore(h5_path, mode="r") as store:
        return store.keys()

def load_mod_hdf(h5_path: str, key: str):
    """
    Load one table: 'variables', 'parameters', 'constraints', 'expressions'
    """
    k = key if key.startswith("/") else f"/{key}"
    return pd.read_hdf(h5_path, key=k)



