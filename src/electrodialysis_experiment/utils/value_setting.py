#################################################################################
# Electrodialysis_Experiment is part of the WaterTAP software platform.
#
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
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
#       mode: fix
#     - scalarVar: fs.EDstack.cell_pair_num
#       value: 10
#       mode: set
#       lb: 1
#       ub: 100
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

from typing import Any, List, Union
import re
import yaml
from pydantic import ValidationError
from pyomo.core.base.var import Var, _VarData
from pyomo.environ import value

from electrodialysis_experiment.schema.config.comp_value_schema import (
    VarAssignment,
    VarValueConfig,
)
import idaes.logger as idaeslogger

_log = idaeslogger.getLogger(__name__)

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
    # quoted string
    if (tok.startswith("'") and tok.endswith("'")) or (
        tok.startswith('"') and tok.endswith('"')
    ):
        return tok[1:-1]
    # int
    try:
        return int(tok)
    except ValueError:
        pass
    # float
    try:
        return float(tok)
    except ValueError:
        pass
    # bare token (e.g., aem, Liq)
    return tok


def _iter_matching_elements(indexed, idx_pattern):
    """
    Yield (key, element) pairs from an IndexedComponent that match idx_pattern.
    idx_pattern can be:
      - "*"                  => match all keys
      - scalar (e.g. "cem")  => prefix match on first axis
      - tuple/list (e.g. ["cem","*"]) with "*" wildcards per position
    """
    for key in indexed:
        k_tuple = key if isinstance(key, tuple) else (key,)

        if idx_pattern == "*" or idx_pattern is None:
            yield key, indexed[key]
            continue

        p_tuple = (
            (idx_pattern,)
            if not isinstance(idx_pattern, (list, tuple))
            else tuple(idx_pattern)
        )

        if len(p_tuple) > len(k_tuple):
            continue

        ok = True
        for p, a in zip(p_tuple, k_tuple):
            if p == "*" or p == a:
                continue
            ok = False
            break
        if ok:
            yield key, indexed[key]


def _apply_indexing(obj, indexers: str):
    """
    Apply chains like:
      [0], ["Liq"], ["Liq","Na_+"], [0,"Liq"], [0]["Liq"]
    to `obj`, returning the indexed object.
    """
    s = indexers
    while s:
        lb = s.find("[")
        rb = s.find("]", lb + 1)
        if lb == -1 or rb == -1:
            raise ValueError(f"Malformed indexers: {s}")

        raw_inside = s[lb + 1 : rb].strip()
        # split tokens (no nesting inside expected)
        parts = (
            [raw_inside]
            if "," not in raw_inside
            else [p.strip() for p in raw_inside.split(",")]
        )

        # convert tokens -> python types (quote-strip, int/float)
        tokens = [_convert_index_token(p) for p in parts] if raw_inside else []
        idx = (
            tuple(tokens) if len(tokens) > 1 else (tokens[0] if tokens else slice(None))
        )

        obj = obj[idx]

        # move to any trailing indexers, e.g. ]["Liq"]["Na_+"]
        s = s[rb + 1 :]
    return obj


def resolve_path(m, path: str):
    """
    Resolve a dotted path with optional bracket indices against model `m`.
    Example: "fs.feed.properties[0].flow_vol_phase['Liq']"
    """
    obj = m
    for piece in path.split("."):
        if not piece:
            continue
        mobj = _INDEX_RE.match(piece)
        if not mobj:
            obj = getattr(obj, piece)
            continue
        attr = mobj.group("attr")
        indexers = mobj.group("indexers")
        obj = getattr(obj, attr)
        if indexers:
            obj = _apply_indexing(obj, indexers)
    return obj


def _fix_var(var, val):
    try:
        var.fix(val)
    except Exception as e:
        raise TypeError(f"Cannot fix value on {getattr(var, 'name', var)}: {e}")


def _set_var(var, val):
    try:

        var.set_value(val)
    except Exception as e:
        raise TypeError(f"Cannot set value on {getattr(var, 'name', var)}: {e}")


def _unfix_var(var):
    try:
        var.unfix()
    except Exception as e:
        raise TypeError(f"Cannot unfix {getattr(var, 'name', var)}: {e}")


def _set_lb(var, lb):
    try:
        var.setlb(lb)
    except Exception as e:
        raise TypeError(f"Cannot set lower bound on {getattr(var, 'name', var)}: {e}")


def _set_ub(var, ub):
    try:
        var.setub(ub)
    except Exception as e:
        raise TypeError(f"Cannot set upper bound on {getattr(var, 'name', var)}: {e}")


def _normalize_index(idx: Union[str, int, float, List[Union[str, int, float]]]):
    if isinstance(idx, list):
        return tuple(idx)
    return idx


def _apply_scalar_assignment(m, assign: VarAssignment):
    var = resolve_path(m, assign.scalarVar)  # may already include [indices]
    if not isinstance(var, (Var, _VarData)):
        raise TypeError(f"Expected {var} type as Pyomo Var or VarData, got {type(var)}")
    if var.is_indexed():
        raise TypeError(
            f"Resolved {assign.scalarVar} to an indexed component; expected scalar. Got {var} with indices {list(var._index_set)}"
        )
    mode = assign.mode.lower()
    if mode == "fix":
        _fix_var(var, assign.value)
        _log.info(
            f"Fixed {assign.scalarVar} to {assign.value}; is_fixed={var.is_fixed()}"
        )
    elif mode == "set":
        print(var.is_fixed())
        _set_var(var, assign.value)
        _log.info(
            f"Set {assign.scalarVar} to {assign.value}; is_fixed={var.is_fixed()}"
        )
        print(var.is_fixed())
        if assign.lb is not None:
            _set_lb(var, assign.lb)
            _log.info(f"Set lower bound of {assign.scalarVar} to {assign.lb}")
        if assign.ub is not None:
            _set_ub(var, assign.ub)
            _log.info(f"Set upper bound of {assign.scalarVar} to {assign.ub}")
    elif mode == "unfix":
        _unfix_var(var)
        _log.info(f"Unfixed {assign.scalarVar}; is_fixed={var.is_fixed()}")
    else:
        raise ValueError(
            f"Unknown mode '{assign.mode}' for scalarVar {assign.scalarVar}"
        )


def _apply_indexed_assignment(m, assign: VarAssignment):
    indexed = resolve_path(m, assign.indexedVar)  # parent IndexedVar/Param
    mode = assign.mode.lower()

    # Whole-variable broadcast: value present, no items
    if (assign.items is None or len(assign.items) == 0) and (assign.value is not None):
        for key in indexed:
            elem = indexed[key]
            if mode == "fix":
                _fix_var(elem, assign.value)
                _log.info(
                    f"Fixed {assign.indexedVar}[{key}] to {assign.value}; is_fixed={elem.is_fixed()}"
                )
            elif mode == "set":
                _set_var(elem, assign.value)
                _log.info(
                    f"Set {assign.indexedVar}[{key}] to {assign.value}; is_fixed={elem.is_fixed()}"
                )
                if assign.lb is not None:
                    _set_lb(elem, assign.lb)
                    _log.info(
                        f"Set lower bound of {assign.indexedVar}[{key}] to {assign.lb}"
                    )
                if assign.ub is not None:
                    _set_ub(elem, assign.ub)
                    _log.info(
                        f"Set upper bound of {assign.indexedVar}[{key}] to {assign.ub}"
                    )
            elif mode == "unfix":
                _unfix_var(elem)
                _log.info(
                    f"Unfixed {assign.indexedVar}[{key}]; is_fixed={elem.is_fixed()}"
                )
            else:
                raise ValueError(
                    f"Unknown mode '{assign.mode}' on indexedVar {assign.indexedVar}"
                )
        return

    # Item-by-item (with wildcard/default broadcast if index omitted)
    for it in assign.items or []:
        # If user omitted index, treat it as "*" (apply to all)
        idx_pattern = _normalize_index(getattr(it, "index", None) or "*")
        matched_any = False
        for _, element in _iter_matching_elements(indexed, idx_pattern):
            matched_any = True
            if mode == "fix":
                _fix_var(element, it.value)
            elif mode == "set":
                _set_var(element, it.value)
                if assign.lb is not None:
                    _set_lb(element, assign.lb)
                if assign.ub is not None:
                    _set_ub(element, assign.ub)
            elif mode == "unfix":
                _unfix_var(element)
            else:
                raise ValueError(
                    f"Unknown mode '{assign.mode}' on indexedVar {assign.indexedVar}"
                )
        if not matched_any:
            raise KeyError(
                f"No indices matched pattern {idx_pattern!r} for {assign.indexedVar}"
            )


def apply_value_updates(m, cfg: VarValueConfig):
    """
    Apply all variable updates in cfg to model `m`.
    """
    for assign in cfg.variables:
        if assign.scalarVar and (assign.value is not None):
            _apply_scalar_assignment(m, assign)
        elif assign.indexedVar and (
            (assign.items is not None) or (assign.value is not None)
        ):
            _apply_indexed_assignment(m, assign)
        elif assign.scalarVar and assign.mode.lower() == "unfix":
            _apply_scalar_assignment(m, assign)
        elif assign.indexedVar and assign.mode.lower() == "unfix":
            _apply_indexed_assignment(m, assign)
        else:
            raise ValueError(
                "Invalid VarAssignment: need either (scalarVar+value) or "
                "(indexedVar+items/value), or (scalarVar|indexedVar + mode=='unfix')."
            )


def apply_value_updates_from_yaml(m, yaml_path: str, key: str = "InitialValue"):
    """
    Load YAML, validate against ParamUpdateConfig, then apply to model m.
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
    apply_value_updates(m, cfg)
