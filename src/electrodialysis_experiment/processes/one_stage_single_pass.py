###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

###############################################################################
# Copyright …
###############################################################################
from __future__ import annotations
from pydantic import BaseModel
from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    value,
    Constraint,
    Objective,
    Expression,
    TransformationFactory,
    assert_optimal_termination,
    units as pyunits,
    NonNegativeReals,
    SolverFactory,
)
from pyomo.network import Arc
from typing import Union, Mapping, TypeVar

from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
    declare_process_block_class,
    ProcessBlockData,
)
from pyomo.common.config import ConfigBlock, ConfigValue, IsInstance, Bool
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
import idaes.core.util.model_statistics as mstat
from idaes.models.unit_models import Feed, Product, Separator
from watertap.unit_models.pressure_changer import Pump
import pandas as pd
import numpy as np
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslogger

from electrodialysis_experiment.processes.base import (
    ED_base,
)

from electrodialysis_experiment.utils.user_scaling import apply_scaling_from_yaml

from watertap.costing.watertap_costing_package import WaterTAPCosting

from electrodialysis_experiment.processes.solution import MCASParameterBlock
from pathlib import Path
import yaml

from electrodialysis_experiment.schema.config.process_config_schema import (
    OneStageSinglePassConfig,
)
from electrodialysis_experiment.utils.value_setting import apply_value_updates_from_yaml
from electrodialysis_experiment.schema.experiment.data import FluidCondition
from electrodialysis_experiment.utils.user_scaling import check_badly_scaled_vars

_log = idaeslogger.getIdaesLogger(__name__)


@declare_process_block_class("OneStageSinglePass")
class OneStageSinglePassData(ProcessBlockData):
    """
    Object-oriented wrapper of the single stage and single pass ED process.
    """

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "process_cfg",
        ConfigValue(
            default=None,
            domain=IsInstance(OneStageSinglePassConfig),
            doc="Process config regulated by Pydantic schema",
        ),
    )
    CONFIG.declare("dynamic", ConfigValue(default=False, domain=Bool))
    # CONFIG.declare("name", ConfigValue(default="Unnamed", domain=str))

    def build(self):
        super().build()
        cfg = self.config.process_cfg
        self.fs = FlowsheetBlock(dynamic=self.config.dynamic)
        self._build_properties()
        self._build_units()
        if cfg.process.build_costing:
            self._build_costing()
        self._add_expressions_and_constraints()
        self._wire_arcs()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OneStageSinglePass":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = OneStageSinglePassConfig(**data)
        proc = OneStageSinglePass(process_cfg=cfg)
        return proc  ## TODO: consdier passing only configs provided in YAML to bypass the default values in the schema; maybe not necessary

    def _build_properties(self):
        ion = self.config.process_cfg.ion
        sol = self.config.process_cfg.solution
        # Assemble kwargs for MCASParameterBlock
        mcas_kwargs = {
            "solute_list": ion.solute_list,
            "mw_data": ion.mw_data,
            "charge": ion.charge,
            "elec_mobility_calculation": sol.electrical_mobility_calculation,
            "equiv_conductivity_calculation": sol.equivalent_conductivity_calculation,
        }
        # optional maps (only if provided)
        if ion.diffusivity_data is not None:
            mcas_kwargs["diffusivity_data"] = ion.diffusivity_data
        if ion.elec_mobility_data is not None:
            mcas_kwargs["elec_mobility_data"] = ion.elec_mobility_data
        if ion.trans_num_data is not None:
            mcas_kwargs["trans_num_data"] = ion.trans_num_data
        if sol.equiv_conductivity_phase_data is not None:
            mcas_kwargs["equiv_conductivity_phase_data"] = (
                sol.equiv_conductivity_phase_data
            )

        self.fs.properties = MCASParameterBlock(**mcas_kwargs)

    def _build_units(self):
        # m = self.m
        ed_config = self.config.process_cfg.ed_stack

        self.fs.feed = Feed(property_package=self.fs.properties)
        self.fs.sepa = Separator(
            property_package=self.fs.properties, outlet_list=["to_dil_in", "to_conc_in"]
        )

        # Pumps
        self.fs.pump0 = Pump(property_package=self.fs.properties)
        self.fs.pump0.del_component("ratioP")
        self.fs.pump0.del_component("ratioP_calculation")
        self.fs.pump1 = Pump(property_package=self.fs.properties)
        self.fs.pump1.del_component("ratioP")
        self.fs.pump1.del_component("ratioP_calculation")

        # ED stack
        ed_kwargs = {
            "dynamic": ed_config.dynamic,
            "has_holdup": ed_config.has_holdup,
            "has_pressure_change": ed_config.has_pressure_change,
            "pressure_drop_method": ed_config.pressure_drop_method,
            "friction_factor_method": ed_config.friction_factor_method,
            "hydraulic_diameter_method": ed_config.hydraulic_diameter_method,
            "operation_mode": ed_config.operation_mode,
            "limiting_current_density_method": ed_config.limiting_current_density_method,
            "limiting_current_density_data": ed_config.limiting_current_density_data,
            "has_nonohmic_potential_membrane": ed_config.has_nonohmic_potential_membrane,
            "has_Nernst_diffusion_layer": ed_config.has_Nernst_diffusion_layer,
            "is_isothermal": ed_config.is_isothermal,
            "property_package_args": ed_config.property_package_args,
            "transformation_method": ed_config.transformation_method,
            "transformation_scheme": ed_config.transformation_scheme,
            "finite_elements": ed_config.finite_elements,
            "collocation_points": ed_config.collocation_points,
        }
        self.fs.EDstack = ED_base(
            property_package=self.fs.properties,
            **ed_kwargs,
        )

        self.fs.prod = Product(property_package=self.fs.properties)
        self.fs.disp = Product(property_package=self.fs.properties)

        # Touch variables to ensure component construction
        self.fs.feed.properties[0].conc_mol_phase_comp[...]
        self.fs.prod.properties[0].conc_mol_phase_comp[...]
        self.fs.disp.properties[0].conc_mol_phase_comp[...]
        self.fs.feed.properties[0].flow_vol_phase[...]
        self.fs.prod.properties[0].flow_vol_phase[...]
        self.fs.disp.properties[0].flow_vol_phase[...]
        self.fs.EDstack.diluate.properties[...].flow_vol_phase[...]
        self.fs.EDstack.concentrate.properties[...].flow_vol_phase[...]
        self.fs.EDstack.diluate.properties[...].conc_mol_phase_comp[...]
        self.fs.EDstack.concentrate.properties[...].conc_mol_phase_comp[...]

    def _build_costing(self):
        m = self.m
        self.fs.costing = WaterTAPCosting()
        self.fs.EDstack.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.fs.costing
        )
        self.fs.pump0.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.fs.costing
        )
        self.fs.pump1.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.fs.costing
        )

        self.fs.costing.cost_process()
        self.fs.costing.add_annual_water_production(
            self.fs.prod.properties[0].flow_vol_phase["Liq"]
        )
        self.fs.costing.add_LCOW(self.fs.prod.properties[0].flow_vol)
        self.fs.costing.add_specific_energy_consumption(
            self.fs.prod.properties[0].flow_vol_phase["Liq"]
        )

    def _add_expressions_and_constraints(self):
        # m = self.m

        # Equal total flow in diluate and concentrate
        self.fs.eq_electrodialysis_equal_flow = Constraint(
            expr=self.fs.EDstack.diluate.properties[0, 0].flow_vol_phase["Liq"]
            - self.fs.EDstack.concentrate.properties[0, 0].flow_vol_phase["Liq"]
            == 0
        )

        # Recovery by volume
        self.fs.recovery_vol_H2O = Expression(
            expr=self.fs.feed.properties[0].flow_vol_phase["Liq"] ** -1
            * self.fs.prod.properties[0].flow_vol_phase["Liq"]
        )

        # Stack voltages
        self.fs.experimental_voltage = Var(
            initialize=100,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Stack voltage measured in a batch experiment",
        )
        self.fs.ocv = Var(
            initialize=0,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Stack open circuit voltage",
        )
        self.fs.eq_experimental_voltage = Constraint(
            expr=self.fs.experimental_voltage
            == self.fs.ocv + self.fs.EDstack.voltage_applied[0]
        )

        # NaCl-equivalent salinity calculations (cation-based weighting)
        self.fs.feed_salinity = Expression(
            expr=sum(
                self.fs.feed.properties[0].conc_mol_phase_comp["Liq", j]
                * (58.5e-3 * self.fs.prod.properties[0].charge_comp[j])
                for j in self.fs.properties.cation_set
            )
        )
        self.fs.prod_salinity = Expression(
            expr=sum(
                self.fs.prod.properties[0].conc_mol_phase_comp["Liq", j]
                * (58.5e-3 * self.fs.prod.properties[0].charge_comp[j])
                for j in self.fs.properties.cation_set
            )
        )
        self.fs.disp_salinity = Expression(
            expr=sum(
                self.fs.disp.properties[0].conc_mol_phase_comp["Liq", j]
                * (58.5e-3 * self.fs.disp.properties[0].charge_comp[j])
                for j in self.fs.properties.cation_set
            )
        )

        # Areas, voltages, current density
        self.fs.mem_area = Expression(
            expr=self.fs.EDstack.cell_width
            * self.fs.EDstack.cell_length
            * self.fs.EDstack.cell_pair_num
        )
        self.fs.voltage_per_cp = Expression(
            expr=self.fs.EDstack.voltage_applied[0] / self.fs.EDstack.cell_pair_num
        )
        self.fs.current_density_avg = Expression(
            expr=self.fs.EDstack.diluate.power_electrical_x[0, 1]
            / (
                self.fs.EDstack.voltage_applied[0]
                * self.fs.EDstack.cell_width
                * self.fs.EDstack.cell_length
            )
        )

    def _wire_arcs(self):
        # m = self.m
        self.fs.arc0 = Arc(source=self.fs.feed.outlet, destination=self.fs.sepa.inlet)
        self.fs.arc1b = Arc(
            source=self.fs.sepa.to_dil_in, destination=self.fs.pump1.inlet
        )
        self.fs.arc1f = Arc(
            source=self.fs.pump1.outlet, destination=self.fs.EDstack.inlet_diluate
        )
        self.fs.arc2b = Arc(
            source=self.fs.sepa.to_conc_in, destination=self.fs.pump0.inlet
        )
        self.fs.arc2f = Arc(
            source=self.fs.pump0.outlet, destination=self.fs.EDstack.inlet_concentrate
        )
        self.fs.arc4 = Arc(
            source=self.fs.EDstack.outlet_diluate, destination=self.fs.prod.inlet
        )
        self.fs.arc5 = Arc(
            source=self.fs.EDstack.outlet_concentrate, destination=self.fs.disp.inlet
        )
        TransformationFactory("network.expand_arcs").apply_to(self.fs)

    def import_scaling_config(self, path: str | Path):
        apply_scaling_from_yaml(self, path)

    def import_init_value_config(self, path: str | Path):
        apply_value_updates_from_yaml(self, path)

    def initialize_process(
        self,
        solve_after_init: bool = True,
        fluid_condition: FluidCondition = None,
        solver=None,
        tee: bool = True,
    ):
        """
        Initialize at DOF==0, optionally solve immediately.
        """
        # m = self.m
        iscale.calculate_scaling_factors(self.fs.feed)
        initargs = fluid_condition.get_state_dict()
        self.fs.feed.properties.calculate_state(initargs, hold_state=True)
        dof = mstat.degrees_of_freedom(self.fs)
        _log.info(f"The process is being intialized at DOF = {dof}.")
        try:
            self._initialize_units()
        except Exception as experr:
            _log.warning(f"Initialization failed at the unit level: {experr}")

        if solve_after_init:
            iscale.constraint_scaling_transform(
                self.fs.eq_electrodialysis_equal_flow,
                10
                * iscale.get_scaling_factor(
                    self.fs.feed.properties[0].flow_vol_phase["Liq"]
                ),
            )
            iscale.calculate_scaling_factors(self.fs)
            #check_badly_scaled_vars(self.fs, small=1e-2, large=1e2)
            res = self.solve(self.fs, solver=solver, tee=tee)
            if str(res.solver.termination_condition) != "optimal":
                _log.warning(
                    f"Process {self.name} did not yield optimal solution when solved at the initial point. "
                    f"Solver termination condition: {res.solver.termination_condition}"
                )
            else:
                _log.info(
                    f"Process {self.name} yielded optimal solution at the initial point."
                )
        else:
            _log.info(
                f"Process {self.name} was set at an estimated initial point determined at the unit level and not solved as a whole."
            )

    @staticmethod
    def solve(
        model: Union[ConcreteModel, Block],
        solver=None,
        tee: bool = True,
    ):
        if solver is None:
            _log.info(
                "No solver specified; using IPOPT solver from Pyomo SolverFactory with all default options."
            )
            solver = SolverFactory("ipopt")
        results = solver.solve(model, tee=tee)
        _log.info(
            f"Solved model: {model.name}; solver termination condition: {results.solver.termination_condition}"
        )
        return results

    def _initialize_units(self):
        # m = self.m
        iscale.calculate_scaling_factors(self.fs)

        # Initialize units and propagate states
        self.fs.feed.initialize()
        propagate_state(self.fs.arc0)

        self.fs.sepa.initialize()
        propagate_state(self.fs.arc1b)

        self.fs.pump1.deltaP[0].fix(2e5)
        self.fs.pump1.initialize()
        self.fs.pump1.deltaP[0].unfix()

        propagate_state(self.fs.arc2b)

        self.fs.pump0.deltaP[0].fix(2e5)
        self.fs.pump0.initialize()
        self.fs.pump0.deltaP[0].unfix()

        propagate_state(self.fs.arc1f)
        propagate_state(self.fs.arc2f)

        self.fs.EDstack.initialize()

        propagate_state(self.fs.arc4)
        self.fs.prod.initialize()

        propagate_state(self.fs.arc5)
        self.fs.prod.initialize()
        self.fs.disp.initialize()

        if hasattr(self.fs, "costing"):
            self.fs.costing.initialize()

    def add_prod_tds_inequality_constraint(self, tds: float = 2.0):
        # TDS in product smaller than specified value (kg/m3, NaCl equivalent)
        self.fs.prod_tds_inequality_constraint = Constraint(
            expr=tds >= self.fs.prod_salinity
        )

    def add_prod_tds_equality_constraint(self, tds: float = 2.0):
        # TDS in product equal to specified value (kg/m3, NaCl equivalent)
        self.fs.prod_tds_equality_constraint = Constraint(
            expr=tds == self.fs.prod_salinity
        )

    def add_sodium_adsorption_ratio(self):
        self.fs.sar = Expression(
            expr=self.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
            * (
                self.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]
                + self.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]
            )
            ** -0.5
        )

    def add_prod_sar_inequality_constraint(self, sar: float = 9.0):
        # SAR in product smaller than specified value
        self.fs.prod_sar_inequality_constraint = Constraint(expr=sar >= self.fs.sar)

    def add_prod_sar_equality_constraint(self, sar: float = 9.0):
        # SAR in product equal to specified value
        self.fs.prod_sar_equality_constraint = Constraint(expr=sar == self.fs.sar)

    def add_LCOW_objective(self):
        if not hasattr(self.fs, "costing"):
            raise AttributeError("Model does not have a costing block.")
        if hasattr(self.fs, "objective"):
            _log.warning(
                "Replacing existing objective {} with LCOW.".format(self.fs.objective)
            )
            self.fs.del_component(self.fs.objective)
        self.fs.objective = Objective(expr=self.fs.costing.LCOW)

    def update_cation_cem_transport_number(
        self,
        t_cation_cem_dict: dict,
    ):
        for ion, t_num in t_cation_cem_dict.items():
            self.fs.EDstack.ion_trans_number_membrane["cem", ion, :].fix(t_num)
            _log.info(f"Fixed cation transport number in CEM for ion '{ion}' to {t_num}.")

    def update_var_values(self, updates: dict | BaseModel) -> None:
        """
        Update variable values in the model.

        Parameters
        ----------
        updates : dict | BaseModel
            Either a plain dict of {var_name: value}
            or a Pydantic model (e.g., UpdateParam) containing the same keys.
        """
        # --- Normalize to a plain dict ---
        if isinstance(updates, Mapping):
            update_dict = dict(updates)
        elif isinstance(updates, BaseModel):
            # works for both v1 (.dict()) and v2 (.model_dump())
            if hasattr(updates, "model_dump"):
                update_dict = updates.model_dump(exclude_unset=True)
            else:
                update_dict = updates.dict()
        else:
            raise TypeError(f"Expected dict or BaseModel, got {type(updates).__name__}")

        for var_name, val in update_dict.items():
            var = type(self).search_var_by_name(model=self, var_name=var_name)
            if var is None:
                raise KeyError(f"Variable '{var_name}' not found in model.")

            if var.is_indexed():
                if not isinstance(val, Mapping):
                    raise TypeError(
                        f"Variable '{var_name}' is indexed; "
                        "expected a dict of {index: value}."
                    )
                for idx, vval in val.items():
                    idx = idx if isinstance(idx, tuple) else (idx,)
                    var[idx].fix(vval)
                    print(f"Fixed {var_name}{idx} to {vval}.")
            else:
                var.fix(val)
                print(f"Fixed {var_name} to {val}.")

    @staticmethod
    def search_var_by_name(model: Union[ConcreteModel, Block], var_name: str):
        var_candidates = []
        for var in model.component_objects(Var, descend_into=True):
            if var_name in str(var.name):
                var_candidates.append(var)
        if len(var_candidates) == 0:
            raise KeyError(f"Variable '{var_name}' not found in the model.")
        elif len(var_candidates) > 1:
            raise KeyError(
                f"Multiple variables found with name '{var_name}': {var_candidates}. Please specify more precisely."
            )
        else:
            return var_candidates[0]

    def display_selected_model_metrics(self, ion_list=None):
        ion_list = ion_list or []
        # m = self.m

        print("---Flow properties in feed, product and disposal---")
        feed_conc_list = [
            value(self.fs.feed.properties[0].flow_vol_phase["Liq"]),
            value(self.fs.feed_salinity),
        ]
        for i in ion_list:
            feed_conc_list.append(
                value(self.fs.feed.properties[0].conc_mol_phase_comp["Liq", i])
            )

        prod_conc_list = [
            value(self.fs.prod.properties[0].flow_vol_phase["Liq"]),
            value(self.fs.prod_salinity),
        ]
        for i in ion_list:
            prod_conc_list.append(
                value(self.fs.prod.properties[0].conc_mol_phase_comp["Liq", i])
            )

        disp_conc_list = [
            value(self.fs.disp.properties[0].flow_vol_phase["Liq"]),
            value(self.fs.disp_salinity),
        ]
        for i in ion_list:
            disp_conc_list.append(
                value(self.fs.disp.properties[0].conc_mol_phase_comp["Liq", i])
            )

        fp_table = pd.DataFrame(
            data={
                "Feed": feed_conc_list,
                "Product": prod_conc_list,
                "Disposal": disp_conc_list,
            },
            index=[
                "Volume Flow Rate (m3/s)",
                "Total Dissolved Solids (kg/m3, NaCl eqv)",
            ]
            + [f"{i} Molar Conc. (mol/m^3)" for i in ion_list],
        )
        print(fp_table)

        print("---Performance Metrics---")
        pm_table = pd.DataFrame(
            data=[
                value(self.fs.recovery_vol_H2O),
                value(self.fs.mem_area),
                value(self.fs.EDstack.cell_pair_num),
                value(self.fs.EDstack.channel_height),
                value(self.fs.EDstack.cell_length),
                value(self.fs.EDstack.cell_width),
                value(self.fs.experimental_voltage),
                value(self.fs.EDstack.voltage_applied[0]),
                value(self.fs.voltage_per_cp),
            ],
            columns=["value"],
            index=[
                "Water recovery by volume",
                "Total membrane area (aem or cem), m2",
                "ED cell pair number",
                "ED channel height, m",
                "ED cell flow path length",
                "ED cell width",
                "Experimental voltage, V",
                "Cell voltage, V",
                "Cell-pair voltage, V",
            ],
        )
        print(pm_table)

        print("---Pressure and Temperature point checking---")
        pt_dict = {
            "Feed": (
                value(self.fs.feed.outlet.pressure[0]),
                value(self.fs.feed.outlet.temperature[0]),
            ),
            "Pump0_in": (
                value(self.fs.pump0.inlet.pressure[0]),
                value(self.fs.pump0.inlet.temperature[0]),
            ),
            "Pump1_in": (
                value(self.fs.pump1.inlet.pressure[0]),
                value(self.fs.pump1.inlet.temperature[0]),
            ),
            "Pump0_out": (
                value(self.fs.pump0.outlet.pressure[0]),
                value(self.fs.pump0.outlet.temperature[0]),
            ),
            "Pump1_out": (
                value(self.fs.pump1.outlet.pressure[0]),
                value(self.fs.pump1.outlet.temperature[0]),
            ),
            "ED_in_dil": (
                value(self.fs.EDstack.inlet_diluate.pressure[0]),
                value(self.fs.EDstack.inlet_diluate.temperature[0]),
            ),
            "ED_in_conc": (
                value(self.fs.EDstack.inlet_concentrate.pressure[0]),
                value(self.fs.EDstack.inlet_concentrate.temperature[0]),
            ),
            "ED_out_dil": (
                value(self.fs.EDstack.outlet_diluate.pressure[0]),
                value(self.fs.EDstack.outlet_diluate.temperature[0]),
            ),
            "ED_out_conc": (
                value(self.fs.EDstack.outlet_concentrate.pressure[0]),
                value(self.fs.EDstack.outlet_concentrate.temperature[0]),
            ),
            "Prod": (
                value(self.fs.prod.inlet.pressure[0]),
                value(self.fs.prod.inlet.temperature[0]),
            ),
            "Disp": (
                value(self.fs.disp.inlet.pressure[0]),
                value(self.fs.disp.inlet.temperature[0]),
            ),
        }
        pt_table = pd.DataFrame(
            data=pt_dict, index=["Pressure (Pa)", "Temperature (K)"]
        )
        pd.set_option("display.max_columns", None)
        print(pt_table)


OneStageSinglePass.from_yaml = OneStageSinglePassData.from_yaml
OneStageSinglePass.solve = OneStageSinglePassData.solve
