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
from __future__ import annotations
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict

Number = Union[int, float]


class IndexedVarItem(BaseModel):
    """
    One (index, value) pair for an indexed variable.
    index can be:
      - scalar (0, "Liq", "aem")
      - a YAML list for tuples: ["Liq", "Na_+"]
    """

    index: Union[str, int, float, List[Union[str, int, float]]]
    value: Number


class VarAssignment(BaseModel):
    """
    A single variable assignment. You can use either:
      1) target + value (target may include indices explicitly)
      2) base + items     (base points to an IndexedVar; items provide per-index values)
      3) mode: "fix" (default), "set", or "unfix"
    """

    model_config = ConfigDict(extra="forbid")

    # Form 1: target path possibly with indices
    scalarVar: Optional[str] = None
    value: Optional[Number] = None

    # Form 2: base path (IndexedVar) + items
    indexedVar: Optional[str] = None
    items: Optional[List[IndexedVarItem]] = None

    # Behavior:
    #  - fix: var.fix(value)
    #  - set: var.set_value(value) (for mutable/unfixed Params/Vars)
    #  - unfix: var.unfix()
    mode: str = Field(default="fix")
    lb: Optional[Number] = None
    ub: Optional[Number] = None


class VarValueConfig(BaseModel):
    """
    Top-level variable value  config: a list of assignments.
    """

    model_config = ConfigDict(extra="forbid")

    variables: List[VarAssignment] = Field(default_factory=list)
