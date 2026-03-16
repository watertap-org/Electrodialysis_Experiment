import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    BeforeValidator,
)
from typing import Dict, List, Optional, Tuple, Annotated, TypeVar
from electrodialysis_experiment.processes.base import (
    ElectricalOperationMode,
    PressureDropMethod,
    FrictionFactorMethod,
    HydraulicDiameterMethod,
    LimitingCurrentDensityMethod,
)
from electrodialysis_experiment.processes.solution import (
    ElectricalMobilityCalculation,
    EquivalentConductivityCalculation,
    TransportNumberCalculation,
)
from enum import Enum


# Helper to parse enum values case-insensitively from YAML
def _enum_by_name(enum_type: type[Enum]):
    def _parse(v):
        if isinstance(v, str):
            for k, member in enum_type.__members__.items():
                if k.lower() == v.lower():
                    return member
        return v

    return BeforeValidator(_parse)


subenum = TypeVar("subenum", bound=Enum)


def validate_enum(cls: type[subenum]) -> subenum:
    return Annotated[cls, _enum_by_name(cls)]


class ProcessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    build_costing: bool = False
    tee: bool = True
    output_dir: Optional[str] = None


class IPOPTconfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver_name: str = "ipopt-watertap"
    tol: Optional[float] = None
    max_iter: Optional[int] = None
    linear_solver: Optional[str] = None
    bound_push: Optional[float] = None
    mu_strategy: Optional[str] = None
    nlp_scaling_method: Optional[str] = None


class EDStackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dynamic: bool = Field(default=False)  # ED_base requires False
    has_holdup: bool = Field(default=False)  # ED_base requires False
    has_pressure_change: bool = True
    pressure_drop_method: validate_enum(PressureDropMethod) = (
        PressureDropMethod.experimental
    )
    friction_factor_method: validate_enum(FrictionFactorMethod) = (
        FrictionFactorMethod.fixed
    )
    hydraulic_diameter_method: validate_enum(HydraulicDiameterMethod) = (
        HydraulicDiameterMethod.conventional
    )
    operation_mode: validate_enum(ElectricalOperationMode) = (
        ElectricalOperationMode.Constant_Voltage
    )
    limiting_current_density_method: validate_enum(LimitingCurrentDensityMethod) = (
        LimitingCurrentDensityMethod.InitialValue
    )
    limiting_current_density_data: float = 500
    has_nonohmic_potential_membrane: bool = True
    has_Nernst_diffusion_layer: bool = True
    is_isothermal: bool = True
    property_package_args: dict = Field(default_factory=dict)

    # Discretization
    transformation_method: str = "dae.finite_difference"
    transformation_scheme: str = "BACKWARD"
    finite_elements: int = 10
    collocation_points: int = 2


class IonConfig(BaseModel):
    """
    Ion & transport data passed to MCASParameterBlock(**kwargs).
    Users can supply either tuple-keyed dicts directly, or
    YAML-friendly nested dicts (the *_yaml fields).
    """

    model_config = ConfigDict(extra="forbid")

    # Required basics
    solute_list: List[str]
    mw_data: Dict[str, float]
    charge: Dict[str, int]

    # Tuple-keyed forms (what MCAS expects)
    diffusivity_data: Optional[Dict[Tuple[str, str], float]] = None
    elec_mobility_data: Optional[Dict[Tuple[str, str], float]] = None
    trans_num_data: Optional[Dict[Tuple[str, str], float]] = None

    @field_validator(
        "diffusivity_data", "elec_mobility_data", "trans_num_data", mode="before"
    )
    @classmethod
    def _normalize_dict(cls, v):
        #  already tuple-keyed
        if isinstance(v, dict) and all(isinstance(k, tuple) for k in v):
            return v

        # nested keys
        if isinstance(v, dict) and all(isinstance(val, dict) for val in v.values()):
            out = {}
            for phase, comps in v.items():
                for sp, val in comps.items():
                    out[(phase, sp)] = float(val)
            return out

        # Anything else (None, wrong type, etc.)
        return v


class SolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    electrical_mobility_calculation: validate_enum(ElectricalMobilityCalculation) = (
        ElectricalMobilityCalculation.none
    )
    equivalent_conductivity_calculation: validate_enum(
        EquivalentConductivityCalculation
    ) = EquivalentConductivityCalculation.ElectricalMobility
    transport_number_calculation: validate_enum(TransportNumberCalculation) = (
        TransportNumberCalculation.ElectricalMobility
    )
    equiv_conductivity_phase_data: Optional[Dict[str, float]] = None


class OneStageSinglePassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ed_stack: EDStackConfig = Field(default_factory=EDStackConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    ipopt: IPOPTconfig = Field(default_factory=IPOPTconfig)
    ion: IonConfig
    solution: SolutionConfig = Field(default_factory=SolutionConfig)

class FourStageSinglePassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ed_stack_1: EDStackConfig = Field(default_factory=EDStackConfig)
    ed_stack_2: EDStackConfig = Field(default_factory=EDStackConfig)
    ed_stack_3: EDStackConfig = Field(default_factory=EDStackConfig)
    ed_stack_4: EDStackConfig = Field(default_factory=EDStackConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    ipopt: IPOPTconfig = Field(default_factory=IPOPTconfig)
    ion: IonConfig
    solution: SolutionConfig = Field(default_factory=SolutionConfig)


class KStageSinglePassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_stages: int = Field(default=1, ge=1)
    ed_stack: EDStackConfig = Field(default_factory=EDStackConfig)
    ed_stacks: Dict[int, EDStackConfig] = Field(default_factory=dict)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    ipopt: IPOPTconfig = Field(default_factory=IPOPTconfig)
    ion: IonConfig
    solution: SolutionConfig = Field(default_factory=SolutionConfig)

    @model_validator(mode="before")
    @classmethod
    def _collect_legacy_stage_keys(cls, data):
        if not isinstance(data, dict):
            return data

        stage_overrides = dict(data.get("ed_stacks") or {})
        max_stage = None
        legacy_keys = []
        for key, val in data.items():
            match = re.fullmatch(r"ed_stack_(\d+)", str(key))
            if match:
                idx = int(match.group(1))
                stage_overrides[idx] = val
                max_stage = idx if max_stage is None else max(max_stage, idx)
                legacy_keys.append(key)

        if stage_overrides:
            data["ed_stacks"] = stage_overrides
        for key in legacy_keys:
            data.pop(key, None)
        if max_stage is not None and "num_stages" not in data:
            data["num_stages"] = max_stage
        return data

    @field_validator("ed_stacks", mode="before")
    @classmethod
    def _normalize_stage_keys(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value

        normalized = {}
        for key, val in value.items():
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str) and key.isdigit():
                idx = int(key)
            else:
                raise ValueError(
                    f"Invalid stage key '{key}'. Expected integer stage indices."
                )
            normalized[idx] = val
        return normalized

    @model_validator(mode="after")
    def _validate_stage_bounds(self):
        for idx in self.ed_stacks:
            if idx < 1:
                raise ValueError(
                    f"Invalid stage index {idx}. Stage indices must be >= 1."
                )
            if idx > self.num_stages:
                raise ValueError(
                    f"Stage index {idx} exceeds num_stages={self.num_stages}."
                )
        return self
