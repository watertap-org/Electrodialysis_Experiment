# variable_setting.py
# -----------------------------------------------------------------------------
# Apply parameter updates to a Pyomo/IDAES model from a YAML file or dict.
# Supports:
#   - scalarVar + value (+ mode: fix|set|unfix)
#   - indexedVar + items: [{index: ..., value: ...}] (+ mode)
#
# YAML example:
# InitialValue:         
#   variables:
#     - scalarVar: fs.feed.properties[0].pressure
#       value: 101325
#
#     - indexedVar: fs.EDstack.membrane_thickness
#       items:
#         - index: ["cem"]
#           value: 5.7e-4
#         - index: ["aem"]
#           value: 6.35e-4
#
# Usage:
#   from variable_setting import apply_param_updates_from_yaml
#   apply_param_updates_from_yaml(model.m, "path/to/file.yml")
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, List,  Union
import re
import yaml
from pydantic import  ValidationError

from electrodialysis_experiment.configs.fixed_var_schema import VarAssignment, VarValueConfig


_INDEX_RE = re.compile(
    r"""
    (?P<attr>[A-Za-z_]\w*)
    (?P<indexers>(\[
        (?:
          "[^"]*" | '[^']*' | [^()\[\]'",\s]+
          (?:\s*,\s*(?:"[^"]*"|'[^']*'|[^()\[\]'",\s]+))*
        )
    \])*)$
    """,
    re.VERBOSE,
)

def _convert_index_token(tok: str):
    tok = tok.strip()
    if (tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"')):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok

def _apply_indexing(obj: Any, indexers: str) -> Any:
    """
    Apply chains like [0]["Liq"]["Na_+"] or ["Liq","Na_+"] to `obj`.
    """
    while indexers:
        lb = indexers.find('[')
        rb = indexers.find(']', lb + 1)
        if lb == -1 or rb == -1:
            raise ValueError(f"Malformed indexers: {indexers}")
        raw_inside = indexers[lb + 1:rb]
        indexers = indexers[rb + 1:]

        # split by comma at top-level (no nested structures expected)
        parts = [p.strip()] if ',' not in raw_inside else [p.strip() for p in raw_inside.split(',')]
        idx = (
            tuple(_convert_index_token(p) for p in parts)
            if len(parts) > 1
            else _convert_index_token(parts[0])
        )
        obj = obj[idx]
    return obj

def resolve_path(m, path: str):
    """
    Resolve a dotted path with optional bracket indices against model `m`.
    Example: "fs.feed.properties[0].flow_vol_phase['Liq']"
    """
    obj = m
    for piece in path.split('.'):
        if not piece:
            continue
        mobj = _INDEX_RE.match(piece)
        if not mobj:
            obj = getattr(obj, piece)
            continue
        attr = mobj.group('attr')
        indexers = mobj.group('indexers')
        obj = getattr(obj, attr)
        if indexers:
            obj = _apply_indexing(obj, indexers)
    return obj


def _fix_var(var, val):
    # Works for Pyomo Var and (mutable) Param; fallback to set_value if needed.
    try:
        var.fix(val)
    except Exception:
        try:
            var.set_value(val)
        except Exception as e:
            raise TypeError(f"Cannot fix/set value on {getattr(var, 'name', var)}: {e}")

def _set_var(var, val):
    # Prefer set_value, fallback to fix if unavailable.
    try:
        var.set_value(val)
    except Exception:
        var.fix(val)

def _unfix_var(var):
    try:
        var.unfix()
    except Exception as e:
        raise TypeError(f"Cannot unfix {getattr(var, 'name', var)}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Apply assignments
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_index(idx: Union[str, int, float, List[Union[str, int, float]]]):
    if isinstance(idx, list):
        return tuple(idx)
    return idx

def _apply_scalar_assignment(m, assign: VarAssignment):
    var = resolve_path(m, assign.scalarVar)  # may already include [indices]
    mode = assign.mode.lower()
    if mode == "fix":
        _fix_var(var, assign.value)
    elif mode == "set":
        _set_var(var, assign.value)
    elif mode == "unfix":
        _unfix_var(var)
    else:
        raise ValueError(f"Unknown mode '{assign.mode}' for scalarVar {assign.scalarVar}")

def _apply_indexed_assignment(m, assign: VarAssignment):
    indexed = resolve_path(m, assign.indexedVar)  # parent IndexedVar/Param
    mode = assign.mode.lower()
    for it in assign.items or []:
        idx = _normalize_index(it.index)
        element = indexed[idx]
        if mode == "fix":
            _fix_var(element, it.value)
        elif mode == "set":
            _set_var(element, it.value)
        elif mode == "unfix":
            _unfix_var(element)
        else:
            raise ValueError(f"Unknown mode '{assign.mode}' on indexedVar {assign.indexedVar} for index {idx}")

def apply_var_value_updates(m, cfg: VarValueConfig):
    """
    Apply all variable updates in cfg to model `m`.
    """
    for assign in cfg.variables:
        if assign.scalarVar and (assign.value is not None):
            _apply_scalar_assignment(m, assign)
        elif assign.indexedVar and assign.items:
            _apply_indexed_assignment(m, assign)
        elif assign.scalarVar and assign.mode.lower() == "unfix":
            # allow unfix on scalarVar without value
            _apply_scalar_assignment(m, assign)
        else:
            raise ValueError(
                "Invalid VarAssignment: need either (scalarVar+value) or (indexedVar+items), "
                "or (scalarVar+mode=='unfix')."
            )

# ──────────────────────────────────────────────────────────────────────────────
# YAML loader
# ──────────────────────────────────────────────────────────────────────────────

def apply_param_updates_from_yaml(m, yaml_path: str, key: str = "InitialValue"):
    """
    Load YAML, validate against ParamUpdateConfig, then apply to model `m`.

    By default expects:
      InitialValue:
        variables:
          - scalarVar: ...
            value: ...
            mode: fix|set|unfix
          - indexedVar: ...
            items: [{index: ..., value: ...}]
            mode: fix|set|unfix
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get(key, {})
    try:
        cfg = VarValueConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Variable value YAML failed validation: {e}")
    apply_var_value_updates(m, cfg)
