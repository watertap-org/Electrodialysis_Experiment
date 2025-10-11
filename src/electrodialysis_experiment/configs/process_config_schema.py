from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Dict, List, Optional, Tuple
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

## Helper function

def _flatten_phase_map(nested: Optional[Dict[str, Dict[str, float]]]) \
        -> Optional[Dict[Tuple[str, str], float]]:
    """
    Convert YAML-friendly nested map:
        {"Liq": {"A": 1e-9, "B": 1e-10}}
    into tuple-keyed map:
        {("Liq", "A"): 1e-9, ("Liq", "B"): 1e-10}
    """
    if not nested:
        return None
    out: Dict[Tuple[str, str], float] = {}
    for phase, comp_map in nested.items():
        if comp_map is None:
            continue
        for comp, val in comp_map.items():
            out[(phase, comp)] = float(val)
    return out

class ProcessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    build_costing: bool = False                    
    solver_linear: str = "ma27"
    solver_max_iter: Optional[int] = None
    tee: bool = True
    output_dir: Optional[str] = None

class EDStackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dynamic: bool = Field(default=False)                # ED_base requires False
    has_holdup: bool = Field(default=False)             # ED_base requires False
    has_pressure_change: bool = True
    pressure_drop_method: PressureDropMethod = PressureDropMethod.experimental
    friction_factor_method: FrictionFactorMethod = FrictionFactorMethod.fixed
    hydraulic_diameter_method: HydraulicDiameterMethod = HydraulicDiameterMethod.conventional
    operation_mode: ElectricalOperationMode = ElectricalOperationMode.Constant_Voltage
    limiting_current_density_method: LimitingCurrentDensityMethod = LimitingCurrentDensityMethod.InitialValue
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
    

    # YAML-friendly nested forms (phase -> {species: value})
    diffusivity_data_yaml: Optional[Dict[str, Dict[str, float]]] = Field(default=None)
    elec_mobility_data_yaml: Optional[Dict[str, Dict[str, float]]] = Field(default=None)
    trans_num_data_yaml: Optional[Dict[str, Dict[str, float]]] = Field(default=None)

    # Normalizers: if tuple-keyed dict not provided, derive it from *_yaml
    @field_validator("diffusivity_data", mode="before")
    @classmethod
    def _norm_diff(cls, v, info):
        if isinstance(v, dict) and all(isinstance(k, tuple) for k in v):
            return v
        return _flatten_phase_map(info.data.get("diffusivity_data_yaml"))

    @field_validator("elec_mobility_data", mode="before")
    @classmethod
    def _norm_mobility(cls, v, info):
        if isinstance(v, dict) and all(isinstance(k, tuple) for k in v):
            return v
        return _flatten_phase_map(info.data.get("elec_mobility_data_yaml"))

    @field_validator("trans_num_data", mode="before")
    @classmethod
    def _norm_tn(cls, v, info):
        if isinstance(v, dict) and all(isinstance(k, tuple) for k in v):
            return v
        return _flatten_phase_map(info.data.get("trans_num_data_yaml"))

class SolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    electrical_mobility_calculation: ElectricalMobilityCalculation = ElectricalMobilityCalculation.Nernst_Einstein
    equivalent_conductivity_calculation: EquivalentConductivityCalculation = EquivalentConductivityCalculation.Lange
    transport_number_calculation: TransportNumberCalculation = TransportNumberCalculation.FixedValue
    equiv_conductivity_phase_data: Optional[Dict[str,float]] = None  
    
class OneStageSinglePassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ed_stack: EDStackConfig = Field(default_factory=EDStackConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    ion: IonConfig 
    solution: SolutionConfig = Field(default_factory=SolutionConfig)

    

