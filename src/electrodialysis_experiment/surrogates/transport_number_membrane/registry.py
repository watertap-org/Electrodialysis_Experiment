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
