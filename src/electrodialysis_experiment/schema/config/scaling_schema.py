from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Dict, List, Optional, Tuple, Union

Number = Union[int, float]


class SolutionDefaultScaling(BaseModel):
    var: str  # e.g. "flow_mol_phase_comp"
    factor: Number
    index: Optional[List[Union[str, int]]] = None  # e.g. ["Liq", "Na_+"]


class ComponentScaling(BaseModel):
    """For iscale.set_scaling_factor(target_component, factor)."""

    target: str  # dotted path w/ indices, e.g. "fs.EDstack.cell_width"
    factor: Number


class ConstraintScaling(BaseModel):
    """
    For iscale.constraint_scaling_transform(constraint, multiplier * get_scaling_factor(ref)).
    """

    target: str  # e.g. "fs.eq_electrodialysis_equal_flow"
    factor: Number  # e.g. 10


class ScalingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    properties: List[SolutionDefaultScaling] = Field(default_factory=list)
    components: List[ComponentScaling] = Field(default_factory=list)
    constraints: List[ConstraintScaling] = Field(default_factory=list)
