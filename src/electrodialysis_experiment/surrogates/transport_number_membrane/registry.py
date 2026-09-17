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
from typing import Callable, Dict
import pyomo.environ as pyo

# A surrogate builder is a function that mutates a Block (adds Vars/Constraints)
SurrogateFn = Callable[[pyo.Block], None]
SurrogateInitFn = Callable[[pyo.Block, dict, dict, str], None]

SURROGATE_REGISTRY: Dict[str, SurrogateFn] = {}
SURROGATE_INITIALIZE: Dict[str, SurrogateInitFn] = {}
SURROGATE_IDENTITY: Dict[str, SurrogateFn] = {}


def register_transport_number_surrogate(name: str):
    """Decorator to register a transport number surrogate builder under a string key."""

    def deco(fn: SurrogateFn) -> SurrogateFn:
        if name in SURROGATE_REGISTRY:
            raise KeyError(f"Transport number surrogate '{name}' already registered")
        SURROGATE_REGISTRY[name] = fn
        return fn

    return deco


def register_transport_number_surrogate_init(name: str):
    """Decorator to register a transport number surrogate initialization function under a string key."""

    def deco(fn: SurrogateInitFn) -> SurrogateInitFn:
        if name in SURROGATE_INITIALIZE:
            raise KeyError(
                f"Transport number surrogate init '{name}' already registered"
            )
        SURROGATE_INITIALIZE[name] = fn
        return fn

    return deco


def register_transport_number_surrogate_fn_identity(name: str):
    """Decorator to register a transport number surrogate function identity under a string key."""

    def deco(fn: SurrogateFn) -> SurrogateFn:
        if name in SURROGATE_IDENTITY:
            raise KeyError(f"Transport number surrogate '{name}' already registered")
        SURROGATE_IDENTITY[name] = fn
        return fn

    return deco
