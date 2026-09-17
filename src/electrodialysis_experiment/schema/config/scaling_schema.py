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
