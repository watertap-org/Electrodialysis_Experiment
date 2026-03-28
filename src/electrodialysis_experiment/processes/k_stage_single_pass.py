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

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Union

import pandas as pd
from pydantic import BaseModel
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, IsInstance
from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    Var,
    value,
    units as pyunits,
)
from pyomo.network import Arc

from idaes.core import (
    FlowsheetBlock,
    ProcessBlockData,
    UnitModelCostingBlock,
    declare_process_block_class,
)
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.misc import add_object_reference
import idaes.core.util.model_statistics as mstat
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslogger
from idaes.models.unit_models import Feed, Product, Separator
import plotly.graph_objs as go
from watertap.costing.watertap_costing_package import WaterTAPCosting
from watertap.unit_models.pressure_changer import Pump
import yaml

from electrodialysis_experiment.processes.base import ED_base, ElectricalOperationMode
from electrodialysis_experiment.processes.solution import MCASParameterBlock
from electrodialysis_experiment.schema.config.process_config_schema import (
    KStageSinglePassConfig,
)
from electrodialysis_experiment.schema.experiment.data import FluidCondition
from electrodialysis_experiment.utils.user_scaling import apply_scaling_from_yaml
from electrodialysis_experiment.utils.value_setting import apply_value_updates_from_yaml

_log = idaeslogger.getIdaesLogger(__name__)

@declare_process_block_class("KStageSinglePass")
class KStageSinglePassData(ProcessBlockData):
    """Object-oriented wrapper for a k-stage single-pass ED process."""

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "process_cfg",
        ConfigValue(
            default=None,
            domain=IsInstance(KStageSinglePassConfig),
            doc="Process config regulated by Pydantic schema",
        ),
    )
    CONFIG.declare("dynamic", ConfigValue(default=False, domain=Bool))

    def build(self):
        super().build()
        cfg = self.config.process_cfg
        self.fs = FlowsheetBlock(dynamic=self.config.dynamic)

        self.fs.stage_set = RangeSet(cfg.num_stages)
        self._build_properties()
        self._build_units()
        if cfg.process.build_costing:
            self._build_costing()
        self._add_expressions_and_constraints()
        self._wire_arcs()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KStageSinglePass":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = KStageSinglePassConfig(**data)
        proc = KStageSinglePass(process_cfg=cfg)
        return proc

    def _stage_indices(self) -> list[int]:
        return [int(s) for s in self.fs.stage_set]

    def _normalize_stage_list(
        self, stage_list: Iterable[int] | int | None
    ) -> list[int]:
        if stage_list is None:
            return self._stage_indices()
        if isinstance(stage_list, int):
            stage_list = [stage_list]

        stage_set = set(self._stage_indices())
        normalized = []
        for stage in stage_list:
            stage_i = int(stage)
            if stage_i not in stage_set:
                raise KeyError(
                    f"Stage {stage_i} is out of bounds. Available stages: {sorted(stage_set)}."
                )
            normalized.append(stage_i)
        return normalized

    def _get_stage_config(self, stage_num: int):
        cfg = self.config.process_cfg
        return cfg.ed_stacks.get(stage_num, cfg.ed_stack)

    @staticmethod
    def _build_ed_kwargs(ed_config):
        return {
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

    def _ed(self, stage_num: int):
        return self.fs.EDstack[int(stage_num)].unit

    def _build_properties(self):
        ion = self.config.process_cfg.ion
        sol = self.config.process_cfg.solution

        mcas_kwargs = {
            "solute_list": ion.solute_list,
            "mw_data": ion.mw_data,
            "charge": ion.charge,
            "elec_mobility_calculation": sol.electrical_mobility_calculation,
            "equiv_conductivity_calculation": sol.equivalent_conductivity_calculation,
        }
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
        fs = self.fs
        fs.feed = Feed(property_package=fs.properties)
        fs.sepa = Separator(
            property_package=fs.properties,
            outlet_list=["to_dil_in", "to_conc_in"],
        )

        fs.pump0 = Pump(property_package=fs.properties)
        fs.pump0.del_component("ratioP")
        fs.pump0.del_component("ratioP_calculation")
        fs.pump1 = Pump(property_package=fs.properties)
        fs.pump1.del_component("ratioP")
        fs.pump1.del_component("ratioP_calculation")

        fs.EDstack = Block(fs.stage_set)
        for stage_num in self._stage_indices():
            ed_config = self._get_stage_config(stage_num)
            ed_kwargs = self._build_ed_kwargs(ed_config)
            fs.EDstack[stage_num].unit = ED_base(
                property_package=fs.properties,
                **ed_kwargs,
            )

        fs.prod = Product(property_package=fs.properties)
        fs.disp = Product(property_package=fs.properties)

        # Touch variables to ensure component construction.
        fs.feed.properties[0].conc_mol_phase_comp[...]
        fs.prod.properties[0].conc_mol_phase_comp[...]
        fs.disp.properties[0].conc_mol_phase_comp[...]
        fs.feed.properties[0].flow_vol_phase[...]
        fs.prod.properties[0].flow_vol_phase[...]
        fs.disp.properties[0].flow_vol_phase[...]
        for stage_num in self._stage_indices():
            ed = self._ed(stage_num)
            ed.diluate.properties[...].flow_vol_phase[...]
            ed.concentrate.properties[...].flow_vol_phase[...]
            ed.diluate.properties[...].conc_mol_phase_comp[...]
            ed.concentrate.properties[...].conc_mol_phase_comp[...]

    def _build_costing(self):
        self.fs.costing = WaterTAPCosting()

        for stage_num in self._stage_indices():
            self._ed(stage_num).costing = UnitModelCostingBlock(
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
        first_stage = self._stage_indices()[0]
        self.fs.eq_electrodialysis_equal_flow = Constraint(
            expr=self._ed(first_stage).diluate.properties[0, 0].flow_vol_phase["Liq"]
            - self._ed(first_stage).concentrate.properties[0, 0].flow_vol_phase["Liq"]
            == 0
        )

        self.fs.recovery_vol_H2O = Expression(
            expr=self.fs.feed.properties[0].flow_vol_phase["Liq"] ** -1
            * self.fs.prod.properties[0].flow_vol_phase["Liq"]
        )

        def _current_density_rule(fs, stage):
            ed = self._ed(int(stage))
            if ed.config.operation_mode == ElectricalOperationMode.Constant_Voltage:
                return ed.diluate.power_electrical_x[0, 1] / (
                    ed.voltage_applied[0] * ed.cell_width * ed.cell_length
                )
            return ed.current_applied[0] / (ed.cell_width * ed.cell_length)

        self.fs.current_density_avg_stage = Expression(
            self.fs.stage_set, rule=_current_density_rule
        )

        for stage_num in self._stage_indices():
            ed = self._ed(stage_num)
            if ed.config.operation_mode == ElectricalOperationMode.Constant_Voltage:
                _log.info(
                    f"Constant_Voltage mode selected for {self.name}, stage {stage_num}."
                )
                ed.current_applied = Expression(
                    [0],
                    expr=self.fs.current_density_avg_stage[stage_num]
                    * ed.cell_width
                    * ed.cell_length,
                )
            else:
                _log.info(
                    f"Constant_Current mode selected for {self.name}, stage {stage_num}."
                )

        def _voltage_avg_rule(fs, stage):
            ed = self._ed(int(stage))
            if ed.config.operation_mode == ElectricalOperationMode.Constant_Voltage:
                return ed.voltage_applied[0]
            return ed.diluate.power_electrical_x[0, 1] / ed.current_applied[0]

        self.fs.voltage_avg_stage = Expression(
            self.fs.stage_set, rule=_voltage_avg_rule
        )
        self.fs.voltage_per_cp_stage = Expression(
            self.fs.stage_set,
            rule=lambda fs, stage: self.fs.voltage_avg_stage[stage]
            / self._ed(int(stage)).cell_pair_num,
        )

        # if len(self._stage_indices()) == 1:
        #     add_object_reference(
        #         self.fs, "current_density_avg", self.fs.current_density_avg_stage[1]
        #     )
        #     add_object_reference(self.fs, "voltage_avg", self.fs.voltage_avg_stage[1])
        #     add_object_reference(
        #         self.fs, "voltage_per_cp", self.fs.voltage_per_cp_stage[1]
        #     )

        for stage_num in self._stage_indices():
            ed = self._ed(stage_num)
            ed.experimental_voltage = Var(
                initialize=100,
                bounds=(0, 1000),
                units=pyunits.volt,
                doc=f"Stage {stage_num} voltage measured in experiment",
            )
            ed.ocv = Var(
                initialize=0,
                bounds=(0, 1000),
                units=pyunits.volt,
                doc=f"Stage {stage_num} open-circuit voltage",
            )
            ed.eq_experimental_voltage = Constraint(
                expr=ed.experimental_voltage
                == ed.ocv + self.fs.voltage_avg_stage[stage_num]
            )

            add_object_reference(
                self.fs,
                f"experimental_voltage_{stage_num}",
                ed.experimental_voltage,
            )
            add_object_reference(self.fs, f"ocv_{stage_num}", ed.ocv)
            add_object_reference(
                self.fs,
                f"eq_experimental_voltage_{stage_num}",
                ed.eq_experimental_voltage,
            )

        if len(self._stage_indices()) == 1:
            add_object_reference(
                self.fs,
                "experimental_voltage",
                self._ed(1).experimental_voltage,
            )
            add_object_reference(self.fs, "ocv", self._ed(1).ocv)
            add_object_reference(
                self.fs,
                "eq_experimental_voltage",
                self._ed(1).eq_experimental_voltage,
            )

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

        self.fs.mem_area = Expression(
            expr=sum(
                self._ed(stage_num).cell_width
                * self._ed(stage_num).cell_length
                * self._ed(stage_num).cell_pair_num
                for stage_num in self._stage_indices()
            )
        )

    def _wire_arcs(self):
        stages = self._stage_indices()
        self.fs.arc0 = Arc(source=self.fs.feed.outlet, destination=self.fs.sepa.inlet)
        self.fs.arc1b = Arc(
            source=self.fs.sepa.to_dil_in, destination=self.fs.pump1.inlet
        )
        self.fs.arc1f = Arc(
            source=self.fs.pump1.outlet, destination=self._ed(stages[0]).inlet_diluate
        )
        self.fs.arc2b = Arc(
            source=self.fs.sepa.to_conc_in, destination=self.fs.pump0.inlet
        )
        self.fs.arc2f = Arc(
            source=self.fs.pump0.outlet,
            destination=self._ed(stages[0]).inlet_concentrate,
        )

        for stage_num in stages[:-1]:
            next_stage = stage_num + 1
            self.fs.add_component(
                f"arc_intered{stage_num}_{next_stage}_dil",
                Arc(
                    source=self._ed(stage_num).outlet_diluate,
                    destination=self._ed(next_stage).inlet_diluate,
                ),
            )
            self.fs.add_component(
                f"arc_intered{stage_num}_{next_stage}_conc",
                Arc(
                    source=self._ed(stage_num).outlet_concentrate,
                    destination=self._ed(next_stage).inlet_concentrate,
                ),
            )

        self.fs.arc4 = Arc(
            source=self._ed(stages[-1]).outlet_diluate,
            destination=self.fs.prod.inlet,
        )
        self.fs.arc5 = Arc(
            source=self._ed(stages[-1]).outlet_concentrate,
            destination=self.fs.disp.inlet,
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
        """Initialize at DOF==0, optionally solve immediately."""
        if fluid_condition is None:
            raise ValueError("fluid_condition must be provided for initialization.")

        iscale.calculate_scaling_factors(self.fs.feed)
        init_solver, init_optarg, solve_solver = self._resolve_solver_settings(solver)
        initargs = fluid_condition.get_state_dict()
        self.fs.feed.properties.calculate_state(
            initargs, hold_state=True, solver=init_solver, optarg=init_optarg
        )
        dof = mstat.degrees_of_freedom(self.fs)
        _log.info(f"The process is being intialized at DOF = {dof}.")

        try:
            self._initialize_units(solver=init_solver, optarg=init_optarg)
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
            res = self.solve(self.fs, solver=solve_solver, tee=tee)
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
                "No solver specified; using the default IPOPT solver from WaterTAP."
            )
            solver = get_solver("ipopt-watertap")
        elif isinstance(solver, str):
            solver = SolverFactory(solver)
        results = solver.solve(model, tee=tee)
        _log.info(
            f"Solved model: {model.name}; solver termination condition: {results.solver.termination_condition}"
        )
        return results

    def _default_solver_and_optarg(self):
        cfg = self.config.process_cfg.ipopt
        optarg = {
            "tol": cfg.tol,
            "max_iter": cfg.max_iter,
            "linear_solver": cfg.linear_solver,
            "bound_push": cfg.bound_push,
            "mu_strategy": cfg.mu_strategy,
            "nlp_scaling_method": cfg.nlp_scaling_method,
        }
        optarg = {k: v for k, v in optarg.items() if v is not None}
        return cfg.solver_name, (optarg if optarg else None)

    def _resolve_solver_settings(self, solver):
        if solver is None:
            solver_name, optarg = self._default_solver_and_optarg()
            return solver_name, optarg, get_solver(
                solver=solver_name, solver_options=optarg
            )
        if isinstance(solver, str):
            return solver, None, get_solver(solver=solver)

        solver_name = getattr(solver, "name", None)
        solver_options = dict(getattr(solver, "options", {}))
        return solver_name, (solver_options if solver_options else None), solver

    def _initialize_units(self, solver=None, optarg=None):
        stages = self._stage_indices()
        iscale.calculate_scaling_factors(self.fs)

        self.fs.feed.initialize(solver=solver, optarg=optarg)
        propagate_state(self.fs.arc0)

        self.fs.sepa.initialize(solver=solver, optarg=optarg)
        propagate_state(self.fs.arc1b)

        self.fs.pump1.deltaP[0].fix(2e5)
        self.fs.pump1.initialize(solver=solver, optarg=optarg)
        self.fs.pump1.deltaP[0].unfix()

        propagate_state(self.fs.arc2b)

        self.fs.pump0.deltaP[0].fix(2e5)
        self.fs.pump0.initialize(solver=solver, optarg=optarg)
        self.fs.pump0.deltaP[0].unfix()

        propagate_state(self.fs.arc1f)
        propagate_state(self.fs.arc2f)

        self._ed(stages[0]).initialize(solver=solver, optarg=optarg)
        for stage_num in stages[:-1]:
            next_stage = stage_num + 1
            propagate_state(
                getattr(self.fs, f"arc_intered{stage_num}_{next_stage}_dil")
            )
            propagate_state(
                getattr(self.fs, f"arc_intered{stage_num}_{next_stage}_conc")
            )
            try:
                self._ed(next_stage).initialize(solver=solver, optarg=optarg)
            except Exception as err:
                _log.warning(f"Failed to initialize EDstack stage {next_stage}: {err}")

        propagate_state(self.fs.arc4)
        self.fs.prod.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc5)
        self.fs.prod.initialize(solver=solver, optarg=optarg)
        self.fs.disp.initialize(solver=solver, optarg=optarg)

        if hasattr(self.fs, "costing"):
            self.fs.costing.initialize()

    def add_prod_tds_inequality_constraint(self, tds: float = 2.0):
        self.fs.prod_tds_inequality_constraint = Constraint(
            expr=tds >= self.fs.prod_salinity
        )

    def add_prod_tds_equality_constraint(self, tds: float = 2.0):
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
        self.fs.prod_sar_inequality_constraint = Constraint(expr=sar >= self.fs.sar)

    def add_prod_sar_equality_constraint(self, sar: float = 9.0):
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
        stage_list: Iterable[int] | int | None = None,
    ):
        stage_list = self._normalize_stage_list(stage_list)
        for ion, t_num in t_cation_cem_dict.items():
            for stage_num in stage_list:
                self._ed(stage_num).ion_trans_number_membrane["cem", ion, :].fix(t_num)
                _log.info(
                    f"Fixed stage {stage_num}'s cation transport number in CEM for ion '{ion}' to {t_num}."
                )

    def fix_steady_constant_current(
        self,
        const_curr_upd: dict | BaseModel,
        stage_list: Iterable[int] | int | None = None,
    ):
        stage_list = self._normalize_stage_list(stage_list)

        if isinstance(const_curr_upd, Mapping):
            const_curr_dict = dict(const_curr_upd)
        elif isinstance(const_curr_upd, BaseModel):
            const_curr_dict = const_curr_upd.model_dump(
                exclude_unset=True, exclude_none=True
            )["current_applied"]
        else:
            raise TypeError(
                f"Expected dict or BaseModel, got {type(const_curr_upd).__name__}"
            )

        normalized_curr = {}
        for key, val in const_curr_dict.items():
            if isinstance(key, int):
                stage_key = key
            elif isinstance(key, str) and key.isdigit():
                stage_key = int(key)
            else:
                raise KeyError(f"Invalid stage key '{key}' in const_curr_upd.")
            normalized_curr[stage_key] = val

        for stage_num in stage_list:
            if stage_num not in normalized_curr:
                raise KeyError(
                    f"Stage {stage_num} not found in const_curr_upd; available stages: {sorted(normalized_curr.keys())}"
                )

            stage_val = normalized_curr[stage_num]
            if isinstance(stage_val, Mapping):
                values = list(stage_val.values())
            else:
                values = [stage_val]

            if len(values) == 0:
                raise KeyError(f"No current value supplied for stage {stage_num}.")

            curr_applied = values[0]
            edstack = self._ed(stage_num)
            if (
                edstack.config.operation_mode
                == ElectricalOperationMode.Constant_Voltage
            ):
                _log.info(
                    f"Stage {stage_num} is in Constant Voltage mode. Current is constrained to {curr_applied} A."
                )
                if hasattr(edstack, "current_applied_constr"):
                    edstack.del_component(edstack.current_applied_constr)
                edstack.current_applied_constr = Constraint(
                    expr=edstack.current_applied[0] == curr_applied * pyunits.ampere
                )
            elif (
                edstack.config.operation_mode
                == ElectricalOperationMode.Constant_Current
            ):
                if not edstack.current_applied.is_fixed():
                    edstack.current_applied[0].fix(curr_applied * pyunits.ampere)
                    _log.info(
                        f"Fixed stage {stage_num}'s current applied to {curr_applied} A."
                    )
                else:
                    _log.warning(
                        f"Stage {stage_num}'s current applied is already fixed."
                    )

    def update_var_values(self, updates: dict | BaseModel) -> None:
        if isinstance(updates, Mapping):
            update_dict = dict(updates)
        elif isinstance(updates, BaseModel):
            update_dict = updates.model_dump(exclude_unset=True, exclude_none=True)
        else:
            raise TypeError(f"Expected dict or BaseModel, got {type(updates).__name__}")

        for var_name, val in update_dict.items():
            var_search = type(self).search_var_by_name(model=self, var_name=var_name)
            if len(var_search) == 0:
                raise KeyError(f"Variable '{var_name}' not found in model.")
            for var in var_search:
                if var.is_indexed():
                    if not isinstance(val, Mapping):
                        raise TypeError(
                            f"Variable '{var_name}' is indexed; expected a dict of {{index: value}}."
                        )
                    for idx, vval in val.items():
                        idx = idx if isinstance(idx, tuple) else (idx,)
                        var[idx].fix(vval)
                        print(f"Fixed {var}{idx} to {vval}.")
                else:
                    var.fix(val)
                    print(f"Fixed {var} to {val}.")

    def update_stg_dep_var_values(
        self,
        stage_num: list[int],
        updates: dict | BaseModel,
    ) -> None:
        stage_num = self._normalize_stage_list(stage_num)

        if isinstance(updates, Mapping):
            update_dict = dict(updates)
        elif isinstance(updates, BaseModel):
            update_dict = updates.model_dump(exclude_unset=True, exclude_none=True)
        else:
            raise TypeError(f"Expected dict or BaseModel, got {type(updates).__name__}")

        for var_name, val in update_dict.items():
            for num in stage_num:
                var_search_stg = type(self).search_var_by_name(
                    model=self._ed(num), var_name=var_name
                )
                if len(var_search_stg) == 0:
                    raise KeyError(f"Variable '{var_name}' not found for stage {num}.")
                if len(var_search_stg) > 1:
                    raise TypeError(
                        f"Multiple variables found with name '{var_name}' for stage {num}: "
                        f"{[var.name for var in var_search_stg]}."
                    )

                var_stg = var_search_stg[0]
                val_stg = (
                    val.get(num, val.get(str(num), val))
                    if isinstance(val, Mapping)
                    else val
                )
                if var_stg.is_indexed():
                    if not isinstance(val_stg, Mapping):
                        raise TypeError(
                            f"Variable '{var_name}' is indexed; expected a dict of {{index: value}}."
                        )
                    for idx, vval in val_stg.items():
                        idx = idx if isinstance(idx, tuple) else (idx,)
                        var_stg[idx].fix(vval)
                        print(f"Fixed {var_stg}{idx} to {vval}.")
                else:
                    var_stg.fix(val_stg)
                    print(f"Fixed {var_stg} to {val_stg}.")

    @staticmethod
    def search_var_by_name(model: Union[ConcreteModel, Block], var_name: str):
        var_candidates = []
        for var in model.component_objects(Var, descend_into=True):
            if var_name in str(var.name):
                var_candidates.append(var)
        if len(var_candidates) == 0:
            raise KeyError(f"Variable '{var_name}' not found in the model.")
        elif len(var_candidates) > 1:
            _log.warning(
                f"Multiple variables found with name '{var_name}': {[var.name for var in var_candidates]}."
            )
        return var_candidates

    def display_selected_model_metrics(
        self,
        ion_list=None,
        feed_product_disposal_properties=True,
        stage_ion_treatment=True,
        stage_performance=True,
        pressure_temperature=True,
        ocv=True,
    ):
        ion_list = ion_list or []
        stages = self._stage_indices()
        if feed_product_disposal_properties:
            print("---Flow properties in feed, product and disposal---")
            feed_conc_list = [
                value(self.fs.feed.properties[0].flow_vol_phase["Liq"]),
                value(self.fs.feed_salinity),
            ]
            for ion in ion_list:
                feed_conc_list.append(
                    value(self.fs.feed.properties[0].conc_mol_phase_comp["Liq", ion])
                )

            prod_conc_list = [
                value(self.fs.prod.properties[0].flow_vol_phase["Liq"]),
                value(self.fs.prod_salinity),
            ]
            for ion in ion_list:
                prod_conc_list.append(
                    value(self.fs.prod.properties[0].conc_mol_phase_comp["Liq", ion])
                )

            disp_conc_list = [
                value(self.fs.disp.properties[0].flow_vol_phase["Liq"]),
                value(self.fs.disp_salinity),
            ]
            for ion in ion_list:
                disp_conc_list.append(
                    value(self.fs.disp.properties[0].conc_mol_phase_comp["Liq", ion])
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
                + [f"{ion} Molar Conc. (mol/m^3)" for ion in ion_list],
            )
            print(fp_table)

        if stage_ion_treatment:
            print("---Ion treatment in each stage---")
            ion_rows = [f"{ion} Molar Conc. (mol/m^3)" for ion in ion_list]
            stage_labels = [f"S{stage_num}" for stage_num in stages]
            ion_treatment_dict = {}

            for stage_num, stage_label in zip(stages, stage_labels):
                edstack = self._ed(stage_num)
                for ion, row_label in zip(ion_list, ion_rows):
                    conc_in = value(
                        edstack.diluate.properties[0, 0].conc_mol_phase_comp["Liq", ion]
                    )
                    conc_out_dil = value(
                        edstack.diluate.properties[0, 1].conc_mol_phase_comp["Liq", ion]
                    )
                    conc_out_conc = value(
                        edstack.concentrate.properties[0, 1].conc_mol_phase_comp[
                            "Liq", ion
                        ]
                    )
                    removal = (conc_in - conc_out_dil) / conc_in * 100

                    ion_treatment_dict.setdefault(f"{stage_label} Dil_out", {})[
                        row_label
                    ] = conc_out_dil
                    ion_treatment_dict.setdefault(f"{stage_label} Conc_out", {})[
                        row_label
                    ] = conc_out_conc
                    ion_treatment_dict.setdefault(f"{stage_label} Removal", {})[
                        row_label
                    ] = removal

            ordered_columns = (
                [f"{stage_label} Dil_out" for stage_label in stage_labels]
                + [f"{stage_label} Conc_out" for stage_label in stage_labels]
                + [f"{stage_label} Removal" for stage_label in stage_labels]
            )
            ion_treatment_table = pd.DataFrame(ion_treatment_dict).reindex(
                index=ion_rows,
                columns=ordered_columns,
            )
            print(ion_treatment_table.to_string(max_cols=None, line_width=None))

        if stage_performance:
            print("---Stage-wise Performance Metrics---")
            for stage_num in stages:
                edstack = self._ed(stage_num)
                pm_table = pd.DataFrame(
                    data=[
                        value(self.fs.recovery_vol_H2O),
                        value(self.fs.mem_area),
                        value(edstack.cell_pair_num),
                        value(edstack.channel_height),
                        value(edstack.cell_length),
                        value(edstack.cell_width),
                        value(edstack.experimental_voltage),
                        value(self.fs.voltage_avg_stage[stage_num]),
                        value(self.fs.voltage_per_cp_stage[stage_num]),
                        value(edstack.current_applied[0]),
                        value(edstack.current_utilization),
                    ],
                    columns=["value"],
                    index=[
                        "Water recovery by volume",
                        "Total membrane area (aem or cem), m2",
                        "ED cell pair number",
                        "ED channel height, m",
                        "ED cell flow path length",
                        "ED cell width",
                        "Voltage (w/ ocv), V",
                        "Cell voltage, V",
                        "Cell-pair voltage, V",
                        "Stack current, A",
                        "Current Utilization",
                    ],
                )
                print(f"\n--- ED Stack {stage_num} ---")
                print(pm_table)

        if pressure_temperature:
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
            }

            for stage_num in stages:
                edstack = self._ed(stage_num)
                pt_dict[f"ED{stage_num}_in_dil"] = (
                    value(edstack.inlet_diluate.pressure[0]),
                    value(edstack.inlet_diluate.temperature[0]),
                )
                pt_dict[f"ED{stage_num}_in_conc"] = (
                    value(edstack.inlet_concentrate.pressure[0]),
                    value(edstack.inlet_concentrate.temperature[0]),
                )
                pt_dict[f"ED{stage_num}_out_dil"] = (
                    value(edstack.outlet_diluate.pressure[0]),
                    value(edstack.outlet_diluate.temperature[0]),
                )
                pt_dict[f"ED{stage_num}_out_conc"] = (
                    value(edstack.outlet_concentrate.pressure[0]),
                    value(edstack.outlet_concentrate.temperature[0]),
                )

            pt_dict["Prod"] = (
                value(self.fs.prod.inlet.pressure[0]),
                value(self.fs.prod.inlet.temperature[0]),
            )
            pt_dict["Disp"] = (
                value(self.fs.disp.inlet.pressure[0]),
                value(self.fs.disp.inlet.temperature[0]),
            )
            pt_table = pd.DataFrame(
                data=pt_dict,
                index=["Pressure (Pa)", "Temperature (K)"],
            )
            pd.set_option("display.max_columns", None)
            print(pt_table)

        if ocv:
            print("---OCV checking---")
            ocv_dict = {
                f"Stage {stage_num}": value(self._ed(stage_num).ocv)
                for stage_num in stages
            }
            ocv_table = pd.DataFrame(data=ocv_dict, index=["Open Circuit Voltage (V)"])
            print(ocv_table)

    def plot_lengthwise_profile(
        self,
        var_name: str,
        stage_num: int = 1,
        default_time: float = 0,
        *,
        precision: int | None = None,
        **fixed_index,
    ):
        stage_num = self._normalize_stage_list([stage_num])[0]
        edstack = self._ed(stage_num)

        vars_found = self.search_var_by_name(edstack, var_name)
        if len(vars_found) != 1:
            raise TypeError(
                f"Expected one variable for '{var_name}' in stage {stage_num}, found {len(vars_found)}."
            )
        var = vars_found[0]
        if not var.is_indexed():
            raise TypeError(f"Variable '{var_name}' is not indexed.")

        length_domain = edstack.diluate.length_domain
        subsets = list(var.index_set().subsets())
        if not any(s is length_domain for s in subsets):
            raise TypeError(f"Variable '{var_name}' is not indexed over length domain.")

        provided_by_set = {}
        remaining_str_keys = {}
        for key, val in fixed_index.items():
            if hasattr(key, "dimen"):
                provided_by_set[id(key)] = val
            else:
                remaining_str_keys[key] = val

        def _resolve_subset_for_key(key: str):
            candidates = []
            for subset in subsets:
                sname = getattr(subset, "name", "") or ""
                if sname == key or sname.endswith(key):
                    candidates.append(subset)
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) == 0:
                avail = [getattr(s, "name", str(s)) for s in subsets]
                raise KeyError(
                    f"Key '{key}' did not match any index set. Available: {avail}"
                )
            avail = [getattr(s, "name", str(s)) for s in candidates]
            raise KeyError(
                f"Key '{key}' matched multiple index sets: {avail}. "
                "Use the Set object as the key instead."
            )

        for key, val in remaining_str_keys.items():
            subset = _resolve_subset_for_key(key)
            provided_by_set[id(subset)] = val

        time_set = getattr(self.fs, "time", None)
        if (
            time_set is not None
            and any(s is time_set for s in subsets)
            and id(time_set) not in provided_by_set
        ):
            provided_by_set[id(time_set)] = default_time

        x_vals = [value(x) for x in length_domain]
        y_vals = []
        for x in length_domain:
            key = []
            for subset in subsets:
                if subset is length_domain:
                    key.append(x)
                else:
                    if id(subset) in provided_by_set:
                        key.append(provided_by_set[id(subset)])
                    else:
                        raise KeyError(
                            f"Missing index for set '{getattr(subset, 'name', str(subset))}'. "
                            "Provide it via a string suffix or using the Set object as the key."
                        )
            y_vals.append(value(var[tuple(key)]))

        if precision is not None:
            y_vals = [round(y, precision) for y in y_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=var_name))
        fig.update_layout(
            xaxis_title="Length Domain (x/L)",
            yaxis_title=var_name,
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                ticks="outside",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                range=[0, 1],
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                ticks="outside",
                title_font=dict(size=16),
                tickfont=dict(size=14),
            ),
            width=700,
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.show()
        return fig


KStageSinglePass.from_yaml = KStageSinglePassData.from_yaml
KStageSinglePass.solve = KStageSinglePassData.solve
