from pydantic import BaseModel, ConfigDict, Field, field_validator, BeforeValidator
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
    # solver_linear: str = "ma27"
    # solver_max_iter: Optional[int] = None
    tee: bool = True
    output_dir: Optional[str] = None


class IPOPTconfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tol: float = 1e-8
    max_iter: int = 3000
    linear_solver: str = "ma27"
    bound_push: float = 1e-5
    mu_strategy: str = "monotone"
    nlp_scaling_method: str = "user-scaling"


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
