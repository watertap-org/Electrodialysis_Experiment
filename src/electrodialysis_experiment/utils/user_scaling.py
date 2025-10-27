from typing import Any
import re
import yaml
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslogger
from electrodialysis_experiment.schema.config.scaling_schema import ScalingConfig

_log = idaeslogger.getIdaesLogger(__name__)

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
    if (tok.startswith("'") and tok.endswith("'")) or (
        tok.startswith('"') and tok.endswith('"')
    ):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _apply_indexing(obj: Any, indexers: str) -> Any:
    # indexers looks like: [0]["Liq"]["Na_+"] or ["Liq"]
    while indexers:
        lb = indexers.find("[")
        rb = indexers.find("]", lb + 1)
        raw_inside = indexers[lb + 1 : rb]
        indexers = indexers[rb + 1 :]
        # split by comma at top-level (no nesting expected)
        parts = (
            [p.strip() for p in raw_inside.split(",")]
            if "," in raw_inside
            else [raw_inside.strip()]
        )
        idx = (
            tuple(_convert_index_token(p) for p in parts)
            if len(parts) > 1
            else _convert_index_token(parts[0])
        )
        obj = obj[idx]
    return obj


def resolve_path(m, path: str):
    """
    Resolve a dotted path with optional [] indices against model m.
    Example: "fs.feed.properties[0].flow_vol_phase['Liq']"
    """
    obj = m
    for piece in path.split("."):
        if not piece:
            continue
        mobj = _INDEX_RE.match(piece)
        if not mobj:
            # simple attribute, no indexers
            obj = getattr(obj, piece)
            continue
        attr = mobj.group("attr")
        indexers = mobj.group("indexers")
        obj = getattr(obj, attr)
        if indexers:
            obj = _apply_indexing(obj, indexers)
    return obj


def apply_scaling_from_yaml(m, yaml_path: str):
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("scaling", {})
    cfg = ScalingConfig(**data)
    apply_scaling(m, cfg)


def apply_scaling(m, cfg: ScalingConfig):
    # default scaling for solution properties, e.g. flow_mol_phase_comp
    for ps in cfg.properties:
        idx = tuple(ps.index) if ps.index is not None else None
        if idx is None:
            m.fs.properties.set_default_scaling(ps.var, ps.factor)
            _log.info(f"Set default scaling for {ps.var} to {ps.factor}")
        else:
            m.fs.properties.set_default_scaling(ps.var, ps.factor, index=tuple(idx))
            _log.info(f"Set default scaling for {ps.var}{idx} to {ps.factor}")

    # scaling for component variables
    for cs in cfg.components:
        comp = resolve_path(m, cs.target)
        iscale.set_scaling_factor(comp, cs.factor)
        _log.info(f"Set scaling for {cs.target} to {cs.factor}")

    # scaling for constraints
    for cts in cfg.constraints:
        cons = resolve_path(m, cts.target)
        iscale.constraint_scaling_transform(cons, cts.factor)
        _log.info(f"Set scaling for {cts.target} to {cts.factor}")

def check_badly_scaled_vars(model):
    found = False
    for var, val in iscale.badly_scaled_var_generator(model, small=1e-4, large=1e4):
        print(f"Badly scaled var: {var}, value: {val}")
        found = True
    if not found:
        print("No badly scaled variables found.")