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
from idaes.core.util.initialization import (
    propagate_state,
)
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

__author__ = "Xiangyu Bi"

_log = idaeslogger.getIdaesLogger(__name__)

def build(build_costing=False, finite_element=10, **ion_dict):
    # ---building model---
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    if ion_dict == {}:
        ion_dict = {
            "solute_list": ["Na_+", "Ca_2+", "Mg_2+", "Cl_-", "SO4_2-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Mg_2+": 24.305e-3,
                "Ca_2+": 40.078e-3,
                "Cl_-": 35.5e-3,
                "SO4_2-": 96.06e-3,
            },
            # "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8,("Liq", "SO4_2-"): 8.27e-8,},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Ca_2+"): 0.793e-9,
                ("Liq", "Mg_2+"): 0.705e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "SO4_2-"): 1.07e-9,
            },  # https://www.aqion.de/site/diffusion-coefficients
            "charge": {"Na_+": 1, "Mg_2+": 2, "Ca_2+": 2, "Cl_-": -1, "SO4_2-": -2},
        }
    m.fs.properties = MCASParameterBlock(
        **ion_dict,
        elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
        equiv_conductivity_calculation=EquivalentConductivityCalculation.Onsager_Falkenhagen,
        # equiv_conductivity_phase_data={
        #     "Liq": 0.01
        # },  # TODO: to investigate why this affect the use fo ONsager_Falkenhagen method.
    )

    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.sepa = Separator(
        property_package=m.fs.properties,
        outlet_list=["to_dil_in", "to_conc_in"],
    )

    m.fs.pump0 = Pump(property_package=m.fs.properties)
    m.fs.pump0.del_component("ratioP")
    m.fs.pump0.del_component("ratioP_calculation")
    m.fs.pump1 = Pump(property_package=m.fs.properties)
    m.fs.pump1.del_component("ratioP")
    m.fs.pump1.del_component("ratioP_calculation")

    # Add electrodialysis (ED) stacks
    m.fs.EDstack = ED_base(
        property_package=m.fs.properties,
        operation_mode=ElectricalOperationMode.Constant_Voltage,
        finite_elements=finite_element,
        has_pressure_change=True,
        has_nonohmic_potential_membrane=True,
        has_Nernst_diffusion_layer=True,
        pressure_drop_method=PressureDropMethod.experimental,
        # hydraulic_diameter_method=HydraulicDiameterMethod.spacer_specific_area_known,
        # friction_factor_method=FrictionFactorMethod.Gurreri,
    )

    m.fs.prod = Product(property_package=m.fs.properties)
    m.fs.disp = Product(property_package=m.fs.properties)

    # Touching needed variables for initialization and displaying results
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

    if build_costing:
        m.fs.costing = WaterTAPCosting()
        # costing
        m.fs.EDstack.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing
        )
        m.fs.pump0.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.pump1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.cost_process()
        m.fs.costing.add_annual_water_production(
            m.fs.prod.properties[0].flow_vol_phase["Liq"]
        )
        m.fs.costing.add_LCOW(m.fs.prod.properties[0].flow_vol)
        m.fs.costing.add_specific_energy_consumption(
            m.fs.prod.properties[0].flow_vol_phase["Liq"]
        )
    # add extra variables and constraints

    m.fs.eq_electrodialysis_equal_flow = Constraint(
        expr=m.fs.EDstack.diluate.properties[0, 0].flow_vol_phase["Liq"]
        - m.fs.EDstack.concentrate.properties[0, 0].flow_vol_phase["Liq"]
        == 0
    )

    m.fs.recovery_vol_H2O = Expression(
        expr=m.fs.feed.properties[0].flow_vol_phase["Liq"] ** -1
        * m.fs.prod.properties[0].flow_vol_phase["Liq"]
    )

    m.fs.experimental_voltage = Var(
        initialize=100,
        bounds=(0, 1000),
        units=pyunits.volt,
        doc="Stack voltage measured in an batch experiment",
    )
    m.fs.ocv = Var(
        initialize=0,
        bounds=(0, 1000),
        units=pyunits.volt,
        doc="Stack open circuit voltage",
    )
    m.fs.eq_experimental_voltage = Constraint(
        expr=m.fs.experimental_voltage == m.fs.ocv + m.fs.EDstack.voltage_applied[0]
    )

    # Declare feed and product salinity variables. Note that the salinity of the multicomponent system is taken as an equivalent to NaCl.

    m.fs.feed_salinity = Expression(
        expr=sum(
            m.fs.feed.properties[0].conc_mol_phase_comp["Liq", j]
            * (58.5 * 1e-3 * m.fs.prod.properties[0].charge_comp[j])
            for j in m.fs.properties.cation_set
        )
    )
    m.fs.prod_salinity = Expression(
        expr=sum(
            m.fs.prod.properties[0].conc_mol_phase_comp["Liq", j]
            * (58.5 * 1e-3 * m.fs.prod.properties[0].charge_comp[j])
            for j in m.fs.properties.cation_set
        )
        # expr=sum(
        #     m.fs.prod.properties[0].conc_mass_phase_comp["Liq", j]
        #     for j in m.fs.properties.solute_set
        # )
    )

    m.fs.disp_salinity = Expression(
        expr=sum(
            m.fs.disp.properties[0].conc_mol_phase_comp["Liq", j]
            * (58.5 * 1e-3 * m.fs.disp.properties[0].charge_comp[j])
            for j in m.fs.properties.cation_set
        )
        # expr=sum(
        #     m.fs.disp.properties[0].conc_mass_phase_comp["Liq", j]
        #     for j in m.fs.properties.solute_set
        # )
    )

    m.fs.mem_area = Expression(
        expr=m.fs.EDstack.cell_width
        * m.fs.EDstack.cell_length
        * m.fs.EDstack.cell_pair_num
    )

    m.fs.voltage_per_cp = Expression(
        expr=m.fs.EDstack.voltage_applied[0] / m.fs.EDstack.cell_pair_num
    )

    m.fs.current_density_avg = Expression(
        expr=m.fs.EDstack.diluate.power_electrical_x[0, 1]
        / (
            m.fs.EDstack.voltage_applied[0]
            * m.fs.EDstack.cell_width
            * m.fs.EDstack.cell_length
        )
    )

    def rule_trans_num_ce_calc(m, i):
        return (
            m.feed.properties[0].conc_mol_phase_comp["Liq", i]
            * m.properties.charge_comp[i]
            - m.prod.properties[0].conc_mol_phase_comp["Liq", i]
            * m.properties.charge_comp[i]
        ) / sum(
            (
                m.feed.properties[0].conc_mol_phase_comp["Liq", i]
                * m.properties.charge_comp[i]
                - m.prod.properties[0].conc_mol_phase_comp["Liq", i]
                * m.properties.charge_comp[i]
            )
            for i in m.EDstack.cation_set
        )

    m.fs.trans_num_ce_calc = Expression(
        m.fs.EDstack.cation_set,
        rule=rule_trans_num_ce_calc,
    )

    # Add Arcs
    m.fs.arc0 = Arc(source=m.fs.feed.outlet, destination=m.fs.sepa.inlet)
    m.fs.arc1b = Arc(source=m.fs.sepa.to_dil_in, destination=m.fs.pump1.inlet)
    m.fs.arc1f = Arc(source=m.fs.pump1.outlet, destination=m.fs.EDstack.inlet_diluate)
    m.fs.arc2b = Arc(
        source=m.fs.sepa.to_conc_in,
        destination=m.fs.pump0.inlet,
    )
    m.fs.arc2f = Arc(
        source=m.fs.pump0.outlet, destination=m.fs.EDstack.inlet_concentrate
    )
    m.fs.arc4 = Arc(source=m.fs.EDstack.outlet_diluate, destination=m.fs.prod.inlet)
    m.fs.arc5 = Arc(source=m.fs.EDstack.outlet_concentrate, destination=m.fs.disp.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)
    # m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    # m.fs.objective.deactivate()
    return m

def add_prod_tds_constraint(m, tds=2):
    m.fs.prod_tds_constraint = Constraint(expr=tds >= m.fs.prod_salinity)


def add_sodium_adsorption_ratio(m):
    m.fs.sar = Var(
        initialize=10,
        domain=NonNegativeReals,
        units=pyunits.mol ** (1 / 2) * pyunits.m ** -(3 / 2),
        doc="sodium adsorption rate",
    )
    m.fs.sar_calculation = Constraint(
        expr=m.fs.sar
        * (
            m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Mg_2+"]
            + m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Ca_2+"]
        )
        ** (1 / 2)
        == m.fs.prod.properties[0].conc_mol_phase_comp["Liq", "Na_+"]
    )


def add_prod_sar_constraint(m, sar=9):
    m.fs.prod_sar_constraint = Constraint(expr=sar >= m.fs.sar)


def make_initarg_list(conc_mass_list, mw=0.0585, flow_rate_vol=5.2e-4):
    conc_mass_in = pd.DataFrame(data=conc_mass_list, columns=["C0"])  # g/L
    conc_mol_in = conc_mass_in / mw  # mol m-3
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


def apply_param_values(m, yaml_file="", yaml_data=None, prefix="m.fs"):
    if yaml_data == None:
        yaml_data = {}
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        if isinstance(value, dict):
            if key == "0":
                apply_param_values(m=m, yaml_data=value, prefix=f"{prefix}[{key}]")
            else:
                apply_param_values(
                    m=getattr(m, key), yaml_data=value, prefix=f"{prefix}.{key}"
                )
        else:
            if m.is_indexed():
                if "fs._time" in m.index_set().name:

                    var = getattr(m[0], key)
                else:
                    for i, v in m.items():

                        if i == key:
                            v.fix(value)

                    continue
            else:
                var = getattr(m, key)
            var.fix(value)


def initialize_dof0_system(
    m=None,
    solve_after_init=False,
    terminate_nonoptimal_sol=False,
    report_bad_scaling=False,
    initargs=None,
    linear_solver="ma27",
    max_iter: int = None,
    tee=True,
):
    """
    To initialize a system at zero dof and give the option to set up the
    decision variables to be controled.

    Keyword Arguments:
        m : model to be initalized.
        solve_after_init: Bool arg to control whether the model is to be solved after setting
                            all intial values.
        terminate_nonoptimal_sol: Bool arg to control whether exception is triggered upon a
                                    non-optimal solution
        initargs : list of dicts of initial states of the feed solution; the dict
                    should have the same length to the state vars.
        report_bad_scaling: control over whether checking badly scaled vars is performed.
        **deciargs: decision vars to be fixed, as a {name: val} dict.

    Returns: a list of the fixed decision vars and the solver results if solve_after_init
                == True.
    """
    if initargs == None:
        initargs = {}
    # set up somme essential scaling factors
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 2e3, index=("Liq", "Na_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 5e4, index=("Liq", "Mg_2+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 2e3, index=("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e5, index=("Liq", "Ca_2+")
    )
    m.fs.properties.set_default_scaling("flow_vol_phase", 5e4, index=("Liq"))

    iscale.calculate_scaling_factors(
        m.fs.feed
    )  ##TODO: why does this strip the scaling factors of the flow_mol_phase_comp of feed?
    m.fs.feed.properties.calculate_state(
        initargs,
        hold_state=True,
    )

    if not mstat.degrees_of_freedom(m) == 0:
        print(f"None-zero DOF={mstat.degrees_of_freedom(m)}")
    assert mstat.degrees_of_freedom(m) == 0
    try:
        initialize_system(m, report_bad_scaling=report_bad_scaling)
    except Exception as experr:
        _log.warning("Initialization Fails in initialize_dof0_system:{}".format(experr))
        pass
    if solve_after_init:
        iscale.calculate_scaling_factors(m)
        res = solve(
            m,
            tee=tee,
            linear_solver=linear_solver,
            max_iter=max_iter,
            check_termination=terminate_nonoptimal_sol,
        )
        if not res.solver.termination_condition == "optimal":
            _log.warning(
                "{} is not solved to optimality after initialization. ".format(m.name)
                + "Solver termination condition: {}".format(
                    res.solver.termination_condition
                )
            )
        else:
            _log.info("{} solved to optimality after initialization.".format(m.name))

    else:
        _log.info("Model initialized at zero dof without solving.")


def optimize_LCOW_fixed_tds(m, tee=False, tds=2, **optargs):
    """
    Function to optimize LCOW of a defined ED system with a definied product water salinity
    target.  The system should first be initialized to have zero dof before running this method.
    Keyword Arguments:
        m : model to be optimized.
        prod_sal: salinity of product water fixed for optimization
        **optargs: dict argument containing vars to be optimized; The dict member should
                    be in form of {var: (val, lb, ub)}, where (val, lb, ub) passes the initial
                    value, lower bound and upper bound of the var to be optimized. The user is
                    allowed to skip lb and ub, in which case the var will be optimized with no
                    bounds. However, providing lb and ub is encouraged.

    Returns: solver results of the optimization.
    """
    m.fs.objective.activate()
    if not mstat.degrees_of_freedom(m) == 0:
        _log.warning(
            "A model is expected to be fully defined at zero dof before being optimized."
        )
    add_prod_tds_constraint(m, tds=tds)

    for opt_var in optargs:
        var = m.fs.find_component(str(opt_var))
        try:
            assert not var is None
        except:
            var = m.fs.EDstack.find_component(str(opt_var))
            try:
                assert not var is None
            except:
                raise TypeError(
                    "Variable '{}' not found in the model components 'm.fs' or 'm.fs.EDstack'.".format(
                        opt_var
                    )
                )
        var.unfix()
        print(f"{opt_var} is unfixed")
        if len(optargs[opt_var]) == 1:
            var.set_value(optargs[opt_var][0])
        elif len(optargs[opt_var]) == 3:
            var.set_value(optargs[opt_var][0])
            var.setlb(optargs[opt_var][1])
            var.setub(optargs[opt_var][2])

    print(f"dof={mstat.degrees_of_freedom(m)}")
    # assert mstat.degrees_of_freedom(m) == len(optargs) - 1
    result = solve(m, tee=tee)

    if not float(m.fs.EDstack.cell_pair_num.value).is_integer():
        _log.warning(
            "The ED cell pair number is not a integer after optimization "
            "and the model is to be re-solved at its closest integer value."
        )
        print("===Outcome of first optimization===")
        display_model_metrics(m)
        m.fs.EDstack.cell_pair_num.fix(round(m.fs.EDstack.cell_pair_num.value))
    result = solve(m, tee=tee)
    if result.solver.termination_condition == "maxIterations":
        while result.solver.termination_condition == "maxIterations":
            result = solve(m, tee=tee)
    return result


def optimize_LCOW_constr_tds_sra(m, tee=False, tds=2, sar=9, **optargs):
    """
    Function to optimize LCOW of a defined ED system with a definied product water salinity
    target.  The system should first be initialized to have zero dof before running this method.
    Keyword Arguments:
        m : model to be optimized.
        prod_sal: salinity of product water fixed for optimization
        **optargs: dict argument containing vars to be optimized; The dict member should
                    be in form of {var: (val, lb, ub)}, where (val, lb, ub) passes the initial
                    value, lower bound and upper bound of the var to be optimized. The user is
                    allowed to skip lb and ub, in which case the var will be optimized with no
                    bounds. However, providing lb and ub is encouraged.

    Returns: solver results of the optimization.
    """
    m.fs.objective.activate()
    if not mstat.degrees_of_freedom(m) == 0:
        _log.warning(
            "A model is expected to be fully defined at zero dof before being optimized."
        )
    add_prod_tds_constraint(m, tds=tds)
    add_prod_sar_constraint(m, sar=sar)

    for opt_var in optargs:
        var = m.fs.find_component(str(opt_var))
        try:
            assert not var is None
        except:
            var = m.fs.EDstack.find_component(str(opt_var))
            try:
                assert not var is None
            except:
                raise TypeError(
                    "Var {} in the provided opt_var dict are not found in the model's component".format(
                        opt_var
                    )
                )
        var.unfix()
        print(f"{opt_var} is unfixed")
        if len(optargs[opt_var]) == 1:
            var.set_value(optargs[opt_var][0])
        elif len(optargs[opt_var]) == 3:
            var.set_value(optargs[opt_var][0])
            var.setlb(optargs[opt_var][1])
            var.setub(optargs[opt_var][2])

    print(f"dof={mstat.degrees_of_freedom(m)}")
    # assert mstat.degrees_of_freedom(m) == len(optargs) - 1
    result = solve(m, tee=tee)

    if not float(m.fs.EDstack.cell_pair_num.value).is_integer():
        _log.warning(
            "The ED cell pair number is not a integer after optimization "
            "and the model is to be re-solved at its closest integer value."
        )
        print("===Outcome of first optimization===")
        display_model_metrics(m)
        m.fs.EDstack.cell_pair_num.fix(round(m.fs.EDstack.cell_pair_num.value))
    result = solve(m, tee=tee)
    if result.solver.termination_condition == "maxIterations":
        while result.solver.termination_condition == "maxIterations":
            result = solve(m, tee=tee)
    return result


def search_var_by_name(model, var_name: str):
    """
    Search for a variable by its name in the model hierarchy.

    Arguments:
        model : Pyomo model or block.
        var_name : str
            The name of the variable to search for.

    Returns:
        Var object if found, otherwise raises KeyError.
    """
    var_cadidate = []
    for var in model.component_objects(Var, descend_into=True):

        if var_name in str(var.name):
            var_cadidate.append(var)
    if len(var_cadidate) == 0:
        raise KeyError(f"Variable '{var_name}' not found in the model.")
    elif len(var_cadidate) > 1:
        raise KeyError(
            f"Multiple variables found with name '{var_name}': {var_cadidate}. Please specify more precisely."
        )
    else:
        return var_cadidate[0]


def update_var_values_pydantic(model, update_param_obj):
    """
    Update values of scalar or indexed Pyomo variables, searching recursively through model hierarchy.

    Arguments:
        model : Pyomo model or block.
        update_param_obj : UpdateParam instance (Pydantic model)
            Values may be:
                - float (for scalar Vars)
                - dict {index: value} (for indexed Vars)
    """
    update_dict = update_param_obj.dict()

    for var_name, val in update_dict.items():
        var = search_var_by_name(model, var_name)

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


def update_var_values(model, update_dict):
    """
    Update values of scalar or indexed Pyomo variables, searching recursively through model hierarchy.

    Arguments:
        model : Pyomo model or block.
        update_dict : dict
            Keys: variable names as strings
            Values:
                - float (for scalar Vars)
                - dict {index: value} (for indexed Vars)
    """

    for var_name, val in update_dict.items():
        var = search_var_by_name(model, var_name)

        if var.is_indexed():
            if not isinstance(val, dict):
                raise TypeError(
                    f"Variable '{var_name}' is indexed; expected a dict of {{index: value}}."
                )
            for idx, vval in val.items():
                idx = idx if isinstance(idx, tuple) else (idx,)
                var[idx].fix(vval)
                # print(f"Fixed {var_name}{idx} = {vval}")
        else:
            var.fix(val)
            # print(f"Fixed scalar {var_name} = {val}")


def add_LCOW_objective(m):
    """
    Add LCOW objective to the model.

    Parameters:
    - m: Pyomo model with `fs.EDstack`

    Returns:
    - None (modifies model in-place)
    """
    if not hasattr(m.fs, "costing"):
        raise AttributeError("Model does not have a costing block.")

    if hasattr(m.fs, "objective"):
        m.del_component(m.fs.objective)

    m.fs.objective = Objective(expr=m.fs.costing.LCOW)


def set_ion_memb_properties(m, diff=3.28e-11, **property_dict):
    """
    Fixes ion diffusivity and transport numbers in CEM and AEM membranes
    for each solute.

    Parameters:
    - m: Pyomo model with `fs.EDstack`
    - diff: Ion diffusivity in membrane (default 3.28e-11 m²/s)
    - **property_dict:
        - "solute_list": list of ions
        - "membrane_transport_number": dict of transport numbers per ion for "cem" and "aem"

    Returns:
    - None (modifies model in-place)
    """

    for ion in property_dict["solute_list"]:
        m.fs.EDstack.solute_diffusivity_membrane["cem", ion].fix(diff)
        m.fs.EDstack.solute_diffusivity_membrane["aem", ion].fix(diff)

        m.fs.EDstack.ion_trans_number_membrane["cem", ion, :].fix(
            property_dict["membrane_transport_number"][ion]["cem"]
        )
        m.fs.EDstack.ion_trans_number_membrane["aem", ion, :].fix(
            property_dict["membrane_transport_number"][ion]["aem"]
        )


def update_cation_cem_transport_number(m, t_cation_cem_dict: dict):
    """
    Update the transport number of cations in the CEM membrane.

    Parameters:
    - m: Pyomo model with `fs.EDstack`
    - t_cation_cem_dict: dict of transport numbers for cations in CEM membrane

    Returns:
    - None (modifies model in-place)
    """
    for ion, t_num in t_cation_cem_dict.items():
        print(t_num)
        m.fs.EDstack.ion_trans_number_membrane["cem", ion, :].fix(t_num)
        # m.fs.EDstack.ion_trans_number_membrane["cem", ion, 1].pprint()
    # m.fs.EDstack.ion_trans_number_membrane["cem", "Na_+", :].pprint()


def solve(
    blk,
    solver=None,
    tee=True,
    linear_solver="ma27",
    max_iter: int = None,
    check_termination=False,
):
    if solver is None:
        solver = get_solver()
    solver.options["linear_solver"] = linear_solver
    if max_iter is not None:
        solver.options["max_iter"] = max_iter
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)
    return results


def initialize_system(m, solver=None, report_bad_scaling=False):
    # set up solver
    if solver is None:
        solver = get_solver()
    optarg = solver.options
    # set scaling factors for state vars and call the 'calculate_scaling_factors' function
    # set up somme essential scaling factors
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 2e3, index=("Liq", "Na_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 5e4, index=("Liq", "Mg_2+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 2e3, index=("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e5, index=("Liq", "Ca_2+")
    )
    m.fs.properties.set_default_scaling("flow_vol_phase", 5e4, index=("Liq"))
    # m.fs.properties.set_default_scaling(
    # "flow_mol_phase_comp", 1e5, index=("Liq", "SO4_2-")
    # )

    iscale.set_scaling_factor(m.fs.EDstack.cell_width, 5)
    iscale.set_scaling_factor(m.fs.EDstack.cell_length, 1)
    iscale.set_scaling_factor(m.fs.EDstack.cell_pair_num, 0.01)
    iscale.set_scaling_factor(m.fs.EDstack.voltage_applied, 0.1)
    # iscale.set_scaling_factor(m.fs.EDstack.ion_trans_number_membrane["cem",:,:], 10)

    # iscale.set_scaling_factor(m.fs.EDstack.dl_thickness, 1e5)
    iscale.set_scaling_factor(m.fs.EDstack.diluate.power_electrical_x, 1)
    iscale.set_scaling_factor(m.fs.pump0.control_volume.work, 1e1)
    iscale.set_scaling_factor(m.fs.pump1.control_volume.work, 1e1)
    iscale.calculate_scaling_factors(m)
    iscale.constraint_scaling_transform(
        m.fs.eq_electrodialysis_equal_flow,
        10 * iscale.get_scaling_factor(m.fs.feed.properties[0].flow_vol_phase["Liq"]),
    )

    # populate intitial properties throughout the system
    m.fs.feed.initialize(optarg=optarg)
    propagate_state(m.fs.arc0)
    m.fs.sepa.initialize(optarg=optarg)
    propagate_state(m.fs.arc1b)
    m.fs.pump1.deltaP[0].fix(2e5)
    m.fs.pump1.initialize()
    m.fs.pump1.deltaP[0].unfix()
    propagate_state(m.fs.arc2b)
    # propagate_state(destination=m.fs.pump0.inlet, source=m.fs.pump1.inlet)
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

    # Update essential scaling factors

    iscale.calculate_scaling_factors(m)
    if report_bad_scaling:
        print("BADLY SCALED VARS & CONSTRAINS")
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        for j, k in badly_scaled_var_values.items():
            print(j, ":", k)


def display_model_metrics(m, ion_list=[]):

    print("---Flow properties in feed, product and disposal---")

    feed_conc_list = [
        value(m.fs.feed.properties[0].flow_vol_phase["Liq"]),
        value(m.fs.feed_salinity),
    ]
    for i in ion_list:
        feed_conc_list.append(
            value(m.fs.feed.properties[0].conc_mol_phase_comp["Liq", i])
        )

    prod_conc_list = [
        value(m.fs.prod.properties[0].flow_vol_phase["Liq"]),
        value(m.fs.prod_salinity),
    ]
    for i in ion_list:
        prod_conc_list.append(
            value(m.fs.prod.properties[0].conc_mol_phase_comp["Liq", i])
        )

    disp_conc_list = [
        value(m.fs.disp.properties[0].flow_vol_phase["Liq"]),
        value(m.fs.disp_salinity),
    ]
    for i in ion_list:
        disp_conc_list.append(
            value(m.fs.disp.properties[0].conc_mol_phase_comp["Liq", i])
        )

    fp = {
        "Feed": feed_conc_list,
        "Product": prod_conc_list,
        "Disposal": disp_conc_list,
    }

    ind_list = [
        "Volume Flow Rate (m3/s)",
        "Total Dissolved Solids (kg/m3, NaCl eqv)",
    ]
    for i in ion_list:
        ind_list.append(i + " Molar Conc. (mol/m^3)")
    fp_table = pd.DataFrame(data=fp, index=ind_list)
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
            # value(m.fs.costing.specific_energy_consumption),
            # value(m.fs.EDstack.specific_power_electrical[0]),
            # value(m.fs.costing.LCOW),
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
            # "Specific energy consumption, kWh/m3",
            # "Specific energy consumption on ED stack, kWh/m3",
            # "Levelized cost of water, $/m3",
        ],
    )
    print(pm_table)
    print("---Pressure and Temperature point checking---")

    pt_dict = {
        "Feed": (
            value(m.fs.feed.outlet.pressure[0]),
            value(m.fs.feed.outlet.temperature[0]),
        ),
        "Pump0_in": (
            value(m.fs.pump0.inlet.pressure[0]),
            value(m.fs.pump0.inlet.temperature[0]),
        ),
        "Pump1_in": (
            value(m.fs.pump1.inlet.pressure[0]),
            value(m.fs.pump1.inlet.temperature[0]),
        ),
        "Pump0_out": (
            value(m.fs.pump0.outlet.pressure[0]),
            value(m.fs.pump0.outlet.temperature[0]),
        ),
        "Pump1_out": (
            value(m.fs.pump1.outlet.pressure[0]),
            value(m.fs.pump1.outlet.temperature[0]),
        ),
        "ED_in_dil": (
            value(m.fs.EDstack.inlet_diluate.pressure[0]),
            value(m.fs.EDstack.inlet_diluate.temperature[0]),
        ),
        "ED_in_conc": (
            value(m.fs.EDstack.inlet_concentrate.pressure[0]),
            value(m.fs.EDstack.inlet_concentrate.temperature[0]),
        ),
        "ED_out_dil": (
            value(m.fs.EDstack.outlet_diluate.pressure[0]),
            value(m.fs.EDstack.outlet_diluate.temperature[0]),
        ),
        "ED_out_conc": (
            value(m.fs.EDstack.outlet_concentrate.pressure[0]),
            value(m.fs.EDstack.outlet_concentrate.temperature[0]),
        ),
        "Prod": (
            value(m.fs.prod.inlet.pressure[0]),
            value(m.fs.prod.inlet.temperature[0]),
        ),
        "Disp": (
            value(m.fs.disp.inlet.pressure[0]),
            value(m.fs.disp.inlet.temperature[0]),
        ),
    }
    pt_table = pd.DataFrame(data=pt_dict, index=["Pressure (Pa)", "Temperature (K)"])
    pd.set_option("display.max_columns", None)
    print(pt_table)
    for i, c in m.fs.trans_num_ce_calc.items():
        print(f"Calculated transport number of {i} is {value(c)}.")
    m.fs.sar.pprint()


def plot_length_profile(length, ed_property, name):
    l_ind = length * (np.round(np.arange(0, 1.05, 0.05), 2))
    marker = dict(
        size=8,
        color="darkblue",
    )
    layout = dict(
        xaxis=dict(
            title=dict(
                text="Position along the cell length (m)",
                font=dict(size=12),
                standoff=5,
            ),
            range=[0, np.ceil(l_ind[-1] * 100) / 100],
            tick0=0,
            mirror=True,
        ),
        yaxis=dict(
            title=dict(
                text=name,
                font=dict(size=12),
                standoff=5,
            ),
            side="left",
            mirror=True,
        ),
        height=300,
        width=400,
        showlegend=False,
        template="simple_white",
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter(
            x=l_ind,
            y=value(ed_property),
            mode="markers+lines",
            name=name,
            marker=marker,
        )
    )
    fig.show()


if __name__ == "__main__":
    main()
