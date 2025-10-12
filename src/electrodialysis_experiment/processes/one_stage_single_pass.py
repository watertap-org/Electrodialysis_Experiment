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

from pyomo.environ import (
    ConcreteModel,
    Var,
    value,
    Constraint,
    Objective,
    Expression,
    TransformationFactory,
    assert_optimal_termination,
    units as pyunits,
    NonNegativeReals,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock, UnitModelCostingBlock
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
    ElectricalOperationMode,
    PressureDropMethod,
    ED_base,
)

from watertap.costing.watertap_costing_package import WaterTAPCosting

from electrodialysis_experiment.processes.solution import (
    MCASParameterBlock,
    ElectricalMobilityCalculation,
    EquivalentConductivityCalculation,
    TransportNumberCalculation, 
)

import plotly.graph_objects as go
import yaml


_log = idaeslogger.getIdaesLogger(__name__)


class OneStageSinglePass:
    """
    Object-oriented wrapper of the original function-based single-pass ED build.
    Encapsulates:
      - Model construction
      - Optional costing
      - Constraints/Expressions
      - Initialization & solve
      - Updating/fixing parameters
      - Reporting & plotting helpers
    """

    # -----------------------------
    # Construction / Build
    # -----------------------------
    def __init__(self, *, build_costing: bool = False, finite_element: int = 10, **ion_dict):
        """
        Create a one-stage single-pass ED flowsheet.

        Parameters
        ----------
        build_costing : bool
            Whether to add WaterTAP costing blocks.
        finite_element : int
            Number of finite elements along ED length domain.
        **ion_dict : dict
            Ion/solution property inputs; default populated if empty.
        """
        self.build_costing = build_costing
        self.finite_element = finite_element
        self.ion_dict = dict(ion_dict or {})

        self.m = ConcreteModel()
        self.m.fs = FlowsheetBlock(dynamic=False)

        if not self.ion_dict:
            self.ion_dict = {
                "solute_list": ["Na_+", "Ca_2+", "Mg_2+", "Cl_-", "SO4_2-"],
                "mw_data": {
                    "H2O": 18e-3,
                    "Na_+": 23e-3,
                    "Mg_2+": 24.305e-3,
                    "Ca_2+": 40.078e-3,
                    "Cl_-": 35.5e-3,
                    "SO4_2-": 96.06e-3,
                },
                "diffusivity_data": {
                    ("Liq", "Na_+"): 1.33e-9,
                    ("Liq", "Ca_2+"): 0.793e-9,
                    ("Liq", "Mg_2+"): 0.705e-9,
                    ("Liq", "Cl_-"): 2.03e-9,
                    ("Liq", "SO4_2-"): 1.07e-9,
                },
                "charge": {"Na_+": 1, "Mg_2+": 2, "Ca_2+": 2, "Cl_-": -1, "SO4_2-": -2},
            }

        self._build_properties()
        self._build_units()
        if self.build_costing:
            self._build_costing()
        self._add_expressions_and_constraints()
        self._wire_arcs()

    def _build_properties(self):
        m = self.m
        m.fs.properties = MCASParameterBlock(
            **self.ion_dict,
            elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
            equiv_conductivity_calculation=EquivalentConductivityCalculation.Onsager_Falkenhagen,
        )

    def _build_units(self):
        m = self.m
        m.fs.feed = Feed(property_package=m.fs.properties)
        m.fs.sepa = Separator(property_package=m.fs.properties, outlet_list=["to_dil_in", "to_conc_in"])

        # Pumps
        m.fs.pump0 = Pump(property_package=m.fs.properties)
        m.fs.pump0.del_component("ratioP")
        m.fs.pump0.del_component("ratioP_calculation")
        m.fs.pump1 = Pump(property_package=m.fs.properties)
        m.fs.pump1.del_component("ratioP")
        m.fs.pump1.del_component("ratioP_calculation")

        # ED stack
        m.fs.EDstack = ED_base(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=self.finite_element,
            has_pressure_change=True,
            has_nonohmic_potential_membrane=True,
            has_Nernst_diffusion_layer=True,
            pressure_drop_method=PressureDropMethod.experimental,
        )

        # Sinks
        m.fs.prod = Product(property_package=m.fs.properties)
        m.fs.disp = Product(property_package=m.fs.properties)

        # Touch variables to ensure component construction (as in original)
        m.fs.feed.properties[0].conc_mol_phase_comp[...]
        m.fs.prod.properties[0].conc_mol_phase_comp[...]
        m.fs.disp.properties[0].conc_mol_phase_comp[...]
        m.fs.feed.properties[0].flow_vol_phase[...]
        m.fs.prod.properties[0].flow_vol_phase[...]
        m.fs.disp.properties[0].flow_vol_phase[...]
        m.fs.EDstack.diluate.properties[...].flow_vol_phase[...]
        m.fs.EDstack.concentrate.properties[...].flow_vol_phase[...]
        m.fs.EDstack.diluate.properties[...].conc_mol_phase_comp[...]
        m.fs.EDstack.concentrate.properties[...].conc_mol_phase_comp[...]

    def _build_costing(self):
        m = self.m
        m.fs.costing = WaterTAPCosting()
        m.fs.EDstack.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.pump0.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.pump1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

        m.fs.costing.cost_process()
        m.fs.costing.add_annual_water_production(m.fs.prod.properties[0].flow_vol_phase["Liq"])
        m.fs.costing.add_LCOW(m.fs.prod.properties[0].flow_vol)
        m.fs.costing.add_specific_energy_consumption(m.fs.prod.properties[0].flow_vol_phase["Liq"])

    def _add_expressions_and_constraints(self):
        m = self.m

        # Equal total flow in diluate and concentrate
        m.fs.eq_electrodialysis_equal_flow = Constraint(
            expr=m.fs.EDstack.diluate.properties[0, 0].flow_vol_phase["Liq"]
            - m.fs.EDstack.concentrate.properties[0, 0].flow_vol_phase["Liq"]
            == 0
        )

        # Recovery by volume
        m.fs.recovery_vol_H2O = Expression(
            expr=m.fs.feed.properties[0].flow_vol_phase["Liq"] ** -1
            * m.fs.prod.properties[0].flow_vol_phase["Liq"]
        )

        # Stack voltages
        m.fs.experimental_voltage = Var(
            initialize=100, bounds=(0, 1000), units=pyunits.volt, doc="Stack voltage measured in a batch experiment"
        )
        m.fs.ocv = Var(initialize=0, bounds=(0, 1000), units=pyunits.volt, doc="Stack open circuit voltage")
        m.fs.eq_experimental_voltage = Constraint(expr=m.fs.experimental_voltage == m.fs.ocv + m.fs.EDstack.voltage_applied[0])

        # NaCl-equivalent salinity calculations (cation-based weighting)
        m.fs.feed_salinity = Expression(
            expr=sum(
                m.fs.feed.properties[0].conc_mol_phase_comp["Liq", j] * (58.5e-3 * m.fs.prod.properties[0].charge_comp[j])
                for j in m.fs.properties.cation_set
            )
        )
        m.fs.prod_salinity = Expression(
            expr=sum(
                m.fs.prod.properties[0].conc_mol_phase_comp["Liq", j] * (58.5e-3 * m.fs.prod.properties[0].charge_comp[j])
                for j in m.fs.properties.cation_set
            )
        )
        m.fs.disp_salinity = Expression(
            expr=sum(
                m.fs.disp.properties[0].conc_mol_phase_comp["Liq", j] * (58.5e-3 * m.fs.disp.properties[0].charge_comp[j])
                for j in m.fs.properties.cation_set
            )
        )

        # Areas, voltages, current density
        m.fs.mem_area = Expression(expr=m.fs.EDstack.cell_width * m.fs.EDstack.cell_length * m.fs.EDstack.cell_pair_num)
        m.fs.voltage_per_cp = Expression(expr=m.fs.EDstack.voltage_applied[0] / m.fs.EDstack.cell_pair_num)
        m.fs.current_density_avg = Expression(
            expr=m.fs.EDstack.diluate.power_electrical_x[0, 1]
            / (m.fs.EDstack.voltage_applied[0] * m.fs.EDstack.cell_width * m.fs.EDstack.cell_length)
        )

        # Transport number estimate (cation-only)
        def rule_trans_num_ce_calc(m, i):
            num = (
                m.feed.properties[0].conc_mol_phase_comp["Liq", i] * m.properties.charge_comp[i]
                - m.prod.properties[0].conc_mol_phase_comp["Liq", i] * m.properties.charge_comp[i]
            )
            den = sum(
                (
                    m.feed.properties[0].conc_mol_phase_comp["Liq", k] * m.properties.charge_comp[k]
                    - m.prod.properties[0].conc_mol_phase_comp["Liq", k] * m.properties.charge_comp[k]
                )
                for k in m.EDstack.cation_set
            )
            return num / den

        m.fs.trans_num_ce_calc = Expression(m.fs.EDstack.cation_set, rule=rule_trans_num_ce_calc)

    def _wire_arcs(self):
        m = self.m
        m.fs.arc0 = Arc(source=m.fs.feed.outlet, destination=m.fs.sepa.inlet)
        m.fs.arc1b = Arc(source=m.fs.sepa.to_dil_in, destination=m.fs.pump1.inlet)
        m.fs.arc1f = Arc(source=m.fs.pump1.outlet, destination=m.fs.EDstack.inlet_diluate)
        m.fs.arc2b = Arc(source=m.fs.sepa.to_conc_in, destination=m.fs.pump0.inlet)
        m.fs.arc2f = Arc(source=m.fs.pump0.outlet, destination=m.fs.EDstack.inlet_concentrate)
        m.fs.arc4 = Arc(source=m.fs.EDstack.outlet_diluate, destination=m.fs.prod.inlet)
        m.fs.arc5 = Arc(source=m.fs.EDstack.outlet_concentrate, destination=m.fs.disp.inlet)
        TransformationFactory("network.expand_arcs").apply_to(m)

    # -----------------------------
    # Optional constraints / objectives / properties
    # -----------------------------
    def add_prod_tds_constraint(self, tds: float = 2.0):
        self.m.fs.prod_tds_constraint = Constraint(expr=tds >= self.m.fs.prod_salinity)

    def add_sodium_adsorption_ratio(self):
        m = self.m
        m.fs.sar = Var(
            initialize=10,
            domain=NonNegativeReals,
            units=pyunits.mol ** (1 / 2) * pyunits.m ** -(3 / 2),
            doc="sodium adsorption ratio",
        )
        m.fs.sar_calculation = Constraint(
            expr=m.fs.sar
            * (m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"] + m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]) ** 0.5
            == m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
        )

    def add_prod_sar_constraint(self, sar: float = 9.0):
        self.m.fs.prod_sar_constraint = Constraint(expr=sar >= self.m.fs.sar)

    def add_LCOW_objective(self):
        m = self.m
        if not hasattr(m.fs, "costing"):
            raise AttributeError("Model does not have a costing block.")
        if hasattr(m.fs, "objective"):
            m.del_component(m.fs.objective)
        m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    # -----------------------------
    # Parameterization / updates
    # -----------------------------
    @staticmethod
    def make_initarg_list(conc_mass_list, mw=0.0585, flow_rate_vol=5.2e-4):
        conc_mass_in = pd.DataFrame(data=conc_mass_list, columns=["C0"])  # g/L
        conc_mol_in = conc_mass_in / mw
        initarg = []
        for k in conc_mol_in["C0"]:
            initarg.append(
                {
                    ("flow_vol_phase", ("Liq")): flow_rate_vol,
                    ("conc_mol_phase_comp", ("Liq", "Na_+")): k,
                    ("conc_mol_phase_comp", ("Liq", "Cl_-")): k,
                }
            )
        return initarg

    def apply_param_values(self, yaml_file: str = "", yaml_data=None, prefix: str = "m.fs"):
        """
        Mirror of original recursive fixer, now rooted at self.m.
        """
        m = self.m
        if yaml_data is None:
            with open(yaml_file, "r") as file:
                yaml_data = yaml.safe_load(file) or {}

        def _apply(target, data, prefix_str):
            for key, val in data.items():
                if isinstance(val, dict):
                    if key == "0":
                        _apply(target=target, data=val, prefix_str=f"{prefix_str}[{key}]")
                    else:
                        _apply(target=getattr(target, key), data=val, prefix_str=f"{prefix_str}.{key}")
                else:
                    if target.is_indexed():
                        if "fs._time" in target.index_set().name:
                            var = getattr(target[0], key)
                        else:
                            for i, v in target.items():
                                if i == key:
                                    v.fix(val)
                            continue
                    else:
                        var = getattr(target, key)
                    var.fix(val)

        _apply(m, yaml_data, prefix)

    def set_ion_memb_properties(self, diff=3.28e-11, **property_dict):
        """
        Fix membrane diffusivity and transport numbers in CEM/AEM for each ion.
        """
        m = self.m
        for ion in property_dict["solute_list"]:
            m.fs.EDstack.solute_diffusivity_membrane["cem", ion].fix(diff)
            m.fs.EDstack.solute_diffusivity_membrane["aem", ion].fix(diff)
            m.fs.EDstack.ion_trans_number_membrane["cem", ion, :].fix(
                property_dict["membrane_transport_number"][ion]["cem"]
            )
            m.fs.EDstack.ion_trans_number_membrane["aem", ion, :].fix(
                property_dict["membrane_transport_number"][ion]["aem"]
            )

    def update_cation_cem_transport_number(self, t_cation_cem_dict: dict):
        m = self.m
        for ion, t_num in t_cation_cem_dict.items():
            print(t_num)
            m.fs.EDstack.ion_trans_number_membrane["cem", ion, :].fix(t_num)

    # -----------------------------
    # Init / Solve
    # -----------------------------
    def initialize_dof0_system(
        self,
        solve_after_init: bool = False,
        terminate_nonoptimal_sol: bool = False,
        report_bad_scaling: bool = False,
        initargs=None,
        linear_solver: str = "ma27",
        max_iter: int | None = None,
        tee: bool = True,
    ):
        """
        Initialize at DOF==0, optionally solve immediately.
        """
        m = self.m
        initargs = initargs or {}

        # Essential scaling
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2O"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 2e3, index=("Liq", "Na_+"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 5e4, index=("Liq", "Mg_2+"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 2e3, index=("Liq", "Cl_-"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1e5, index=("Liq", "Ca_2+"))
        m.fs.properties.set_default_scaling("flow_vol_phase", 5e4, index=("Liq"))

        iscale.calculate_scaling_factors(m.fs.feed)
        m.fs.feed.properties.calculate_state(initargs, hold_state=True)

        if mstat.degrees_of_freedom(m) != 0:
            print(f"None-zero DOF={mstat.degrees_of_freedom(m)}")
        assert mstat.degrees_of_freedom(m) == 0

        try:
            self.initialize_system(report_bad_scaling=report_bad_scaling)
        except Exception as experr:
            _log.warning(f"Initialization failed in initialize_dof0_system: {experr}")

        if solve_after_init:
            iscale.calculate_scaling_factors(m)
            res = self.solve(tee=tee, linear_solver=linear_solver, max_iter=max_iter, check_termination=terminate_nonoptimal_sol)
            if str(res.solver.termination_condition) != "optimal":
                _log.warning(
                    f"{m.name} not solved to optimality after initialization. "
                    f"Solver termination condition: {res.solver.termination_condition}"
                )
            else:
                _log.info(f"{m.name} solved to optimality after initialization.")
        else:
            _log.info("Model initialized at zero dof without solving.")

    def solve(
        self,
        solver=None,
        tee: bool = True,
        linear_solver: str = "ma27",
        max_iter: int | None = None,
        check_termination: bool = False,
    ):
        if solver is None:
            solver = get_solver()
        solver.options["linear_solver"] = linear_solver
        if max_iter is not None:
            solver.options["max_iter"] = max_iter
        results = solver.solve(self.m, tee=tee)
        if check_termination:
            assert_optimal_termination(results)
        return results

    def initialize_system(self, solver=None, report_bad_scaling: bool = False):
        m = self.m
        if solver is None:
            solver = get_solver()
        optarg = solver.options

        # Scaling
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2O"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 2e3, index=("Liq", "Na_+"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 5e4, index=("Liq", "Mg_2+"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 2e3, index=("Liq", "Cl_-"))
        m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1e5, index=("Liq", "Ca_2+"))
        m.fs.properties.set_default_scaling("flow_vol_phase", 5e4, index=("Liq"))

        iscale.set_scaling_factor(m.fs.EDstack.cell_width, 5)
        iscale.set_scaling_factor(m.fs.EDstack.cell_length, 1)
        iscale.set_scaling_factor(m.fs.EDstack.cell_pair_num, 0.01)
        iscale.set_scaling_factor(m.fs.EDstack.voltage_applied, 0.1)
        iscale.set_scaling_factor(m.fs.EDstack.diluate.power_electrical_x, 1)
        iscale.set_scaling_factor(m.fs.pump0.control_volume.work, 1e1)
        iscale.set_scaling_factor(m.fs.pump1.control_volume.work, 1e1)
        iscale.calculate_scaling_factors(m)
        iscale.constraint_scaling_transform(
            m.fs.eq_electrodialysis_equal_flow,
            10 * iscale.get_scaling_factor(m.fs.feed.properties[0].flow_vol_phase["Liq"]),
        )

        # Initialize units and propagate states
        m.fs.feed.initialize(optarg=optarg)
        propagate_state(m.fs.arc0)

        m.fs.sepa.initialize(optarg=optarg)
        propagate_state(m.fs.arc1b)

        m.fs.pump1.deltaP[0].fix(2e5)
        m.fs.pump1.initialize()
        m.fs.pump1.deltaP[0].unfix()

        propagate_state(m.fs.arc2b)

        m.fs.pump0.deltaP[0].fix(2e5)
        m.fs.pump0.initialize()
        m.fs.pump0.deltaP[0].unfix()

        propagate_state(m.fs.arc1f)
        propagate_state(m.fs.arc2f)

        m.fs.EDstack.initialize(optarg=optarg)

        propagate_state(m.fs.arc4)
        m.fs.prod.initialize(optarg=optarg)

        propagate_state(m.fs.arc5)
        m.fs.prod.initialize(optarg=optarg)
        m.fs.disp.initialize()

        if hasattr(m.fs, "costing"):
            m.fs.costing.initialize()

        iscale.calculate_scaling_factors(m)
        if report_bad_scaling:
            print("BADLY SCALED VARS & CONSTRAINTS")
            badly_scaled_var_values = {var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)}
            for j, k in badly_scaled_var_values.items():
                print(j, ":", k)

    # -----------------------------
    # Variable utilities
    # -----------------------------
    @staticmethod
    def search_var_by_name(model, var_name: str):
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

    def update_var_values_pydantic(self, update_param_obj):
        update_dict = update_param_obj.dict()
        self.update_var_values(update_dict)

    def update_var_values(self, update_dict: dict):
        m = self.m
        for var_name, val in update_dict.items():
            var = self.search_var_by_name(m, var_name)
            if var.is_indexed():
                if not isinstance(val, dict):
                    raise TypeError(
                        f"Variable '{var_name}' is indexed; expected a dict of {{index: value}}."
                    )
                for idx, vval in val.items():
                    idx = idx if isinstance(idx, tuple) else (idx,)
                    var[idx].fix(vval)
            else:
                var.fix(val)

    # -----------------------------
    # Display / Plot
    # -----------------------------
    def display_model_metrics(self, ion_list=None):
        ion_list = ion_list or []
        m = self.m

        print("---Flow properties in feed, product and disposal---")
        feed_conc_list = [value(m.fs.feed.properties[0].flow_vol_phase["Liq"]), value(m.fs.feed_salinity)]
        for i in ion_list:
            feed_conc_list.append(value(m.fs.feed.properties[0].conc_mol_phase_comp["Liq", i]))

        prod_conc_list = [value(m.fs.prod.properties[0].flow_vol_phase["Liq"]), value(m.fs.prod_salinity)]
        for i in ion_list:
            prod_conc_list.append(value(m.fs.prod.properties[0].conc_mol_phase_comp["Liq", i]))

        disp_conc_list = [value(m.fs.disp.properties[0].flow_vol_phase["Liq"]), value(m.fs.disp_salinity)]
        for i in ion_list:
            disp_conc_list.append(value(m.fs.disp.properties[0].conc_mol_phase_comp["Liq", i]))

        fp_table = pd.DataFrame(
            data={"Feed": feed_conc_list, "Product": prod_conc_list, "Disposal": disp_conc_list},
            index=["Volume Flow Rate (m3/s)", "Total Dissolved Solids (kg/m3, NaCl eqv)"]
                  + [f"{i} Molar Conc. (mol/m^3)" for i in ion_list]
        )
        print(fp_table)

        print("---Performance Metrics---")
        pm_table = pd.DataFrame(
            data=[
                value(m.fs.recovery_vol_H2O),
                value(m.fs.mem_area),
                value(m.fs.EDstack.cell_pair_num),
                value(m.fs.EDstack.channel_height),
                value(m.fs.EDstack.cell_length),
                value(m.fs.EDstack.cell_width),
                value(m.fs.experimental_voltage),
                value(m.fs.EDstack.voltage_applied[0]),
                value(m.fs.voltage_per_cp),
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
            "Feed": (value(m.fs.feed.outlet.pressure[0]), value(m.fs.feed.outlet.temperature[0])),
            "Pump0_in": (value(m.fs.pump0.inlet.pressure[0]), value(m.fs.pump0.inlet.temperature[0])),
            "Pump1_in": (value(m.fs.pump1.inlet.pressure[0]), value(m.fs.pump1.inlet.temperature[0])),
            "Pump0_out": (value(m.fs.pump0.outlet.pressure[0]), value(m.fs.pump0.outlet.temperature[0])),
            "Pump1_out": (value(m.fs.pump1.outlet.pressure[0]), value(m.fs.pump1.outlet.temperature[0])),
            "ED_in_dil": (value(m.fs.EDstack.inlet_diluate.pressure[0]), value(m.fs.EDstack.inlet_diluate.temperature[0])),
            "ED_in_conc": (value(m.fs.EDstack.inlet_concentrate.pressure[0]), value(m.fs.EDstack.inlet_concentrate.temperature[0])),
            "ED_out_dil": (value(m.fs.EDstack.outlet_diluate.pressure[0]), value(m.fs.EDstack.outlet_diluate.temperature[0])),
            "ED_out_conc": (value(m.fs.EDstack.outlet_concentrate.pressure[0]), value(m.fs.EDstack.outlet_concentrate.temperature[0])),
            "Prod": (value(m.fs.prod.inlet.pressure[0]), value(m.fs.prod.inlet.temperature[0])),
            "Disp": (value(m.fs.disp.inlet.pressure[0]), value(m.fs.disp.inlet.temperature[0])),
        }
        pt_table = pd.DataFrame(data=pt_dict, index=["Pressure (Pa)", "Temperature (K)"])
        pd.set_option("display.max_columns", None)
        print(pt_table)

        for i, c in m.fs.trans_num_ce_calc.items():
            print(f"Calculated transport number of {i} is {value(c)}.")
        if hasattr(m.fs, "sar"):
            m.fs.sar.pprint()

    @staticmethod
    def plot_length_profile(length, ed_property, name: str):
        l_ind = length * (np.round(np.arange(0, 1.05, 0.05), 2))
        marker = dict(size=8, color="darkblue")
        layout = dict(
            xaxis=dict(title=dict(text="Position along the cell length (m)", font=dict(size=12), standoff=5),
                       range=[0, float(np.ceil(l_ind[-1] * 100) / 100)], tick0=0, mirror=True),
            yaxis=dict(title=dict(text=name, font=dict(size=12), standoff=5), side="left", mirror=True),
            height=300, width=400, showlegend=False, template="simple_white",
        )
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=l_ind, y=value(ed_property), mode="markers+lines", name=name, marker=marker))
        fig.show()



