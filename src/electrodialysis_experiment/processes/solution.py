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
"""
This property package computes a multi-component aqueous solution that can
contain ionic and/or neutral solute species. It supports basic calculation 
of component quantities and some physical, chemical and electrical properties. 

This property package was formerly named the "ion_DSPMDE_prop_pack" and was originally 
designed for use with the Donnan Steric Pore Model with Dielectric Exclusion (DSPMDE) for
nanofiltration.
"""

# Import Python libraries
import idaes.logger as idaeslog

from enum import Enum, auto

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Expression,
    Reals,
    NonNegativeReals,
    Var,
    Param,
    Set,
    Suffix,
    value,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.common.config import ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
)
from idaes.core.base.components import Solute, Solvent, Cation, Anion
from idaes.core.base.phases import AqueousPhase
from idaes.core.util.constants import Constants
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core.util.exceptions import (
    ConfigurationError,
    InitializationError,
    PropertyPackageError,
)
import idaes.core.util.scaling as iscale
from watertap.core.util.scaling import transform_property_constraints

_log = idaeslog.getLogger(__name__)


class DensityCalculation(Enum):
    constant = auto()  # constant @ 1000 kg/m3
    seawater = auto()  # seawater correlation for TDS from Sharqawy


class DiffusivityCalculation(Enum):
    none = auto()
    HaydukLaudie = auto()


class ElectricalMobilityCalculation(Enum):
    none = auto()
    EinsteinRelation = auto()


class EquivalentConductivityCalculation(Enum):
    none = auto()
    ElectricalMobility = auto()
    Onsager_Falkenhagen = auto()


class TransportNumberCalculation(Enum):
    none = auto()
    ElectricalMobility = auto()


@declare_process_block_class("MCASParameterBlock")
class MCASParameterData(PhysicalParameterBlock):
    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare(
        "solute_list",
        ConfigValue(
            domain=list,
            description="Required argument.List of strings that specify names of solute species.",
        ),
    )
    CONFIG.declare(
        "diffusivity_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict of solute species names (keys) and bulk ion diffusivity data (values)",
        ),
    )
    CONFIG.declare(
        "membrane_transport_number",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict of ions' transport number in the cem or aem.",
        ),
    )
    CONFIG.declare(
        "molar_volume_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict of solute species names and molar volume of aqueous species",
        ),
    )
    CONFIG.declare(
        "mw_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Required argument. Dict of component names (keys)and molecular weight data (values)",
        ),
    )
    CONFIG.declare(
        "elec_mobility_data",
        ConfigValue(default={}, domain=dict, description="Ion electrical mobility"),
    )
    CONFIG.declare(
        "trans_num_data",
        ConfigValue(
            default={},
            domain=dict,
            description="transport number of ions in the liquid phase",
        ),
    )
    CONFIG.declare(
        "equiv_conductivity_phase_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Equivalent conductivity of ions in the liquid phase",
        ),
    )

    CONFIG.declare(
        "charge", ConfigValue(default={}, domain=dict, description="Ion charge")
    )
    CONFIG.declare(
        "density_calculation",
        ConfigValue(
            default=DensityCalculation.constant,
            domain=In(DensityCalculation),
            description="Solution density calculation construction flag",
            doc="""
           Options to account for solution density.

           **default** - ``DensityCalculation.constant``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``DensityCalculation.constant``", "Solution density assumed constant at 1000 kg/m3 by default in dens_mass_const parameter"
           "``DensityCalculation.seawater``", "Solution density based on correlation for seawater (TDS)"
       """,
        ),
    )
    CONFIG.declare(
        "diffus_calculation",
        ConfigValue(
            default=DiffusivityCalculation.none,
            domain=In(DiffusivityCalculation),
            description="Diffusivity calculation flag",
            doc="""
           Options to account for ionic or molecular diffusivity.

           **default** - ``DiffusivityCalculation.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``DiffusivityCalculation.none``", "Users provide data via the diffusivity_data configuration"
           "``DiffusivityCalculation.HaydukLaudie``", "Allow the nonelectrolyte (neutral) species to get diffusivity from the Hayduk Laudie equation"
       """,
        ),
    )
    CONFIG.declare(
        "elec_mobility_calculation",
        ConfigValue(
            default=ElectricalMobilityCalculation.none,
            domain=In(ElectricalMobilityCalculation),
            description="Electrical mobility calculation flag",
            doc="""
           Options to account for ion electrical mobility.

           **default** - ``ElectricalMobilityCalculation.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``ElectricalMobilityCalculation.none``", "Users provide data via the elec_mobility_data configuration"
           "``ElectricalMobilityCalculation.EinsteinRelation``", "Calculate from the diffusivity_data by the Einstein Relation"
       """,
        ),
    )

    CONFIG.declare(
        "trans_num_calculation",
        ConfigValue(
            default=TransportNumberCalculation.ElectricalMobility,
            domain=In(TransportNumberCalculation),
            description="Ion transport number calculation flag",
            doc="""
           Options to account for ion transport number in the solution.

           **default** - ``TransportNumberCalculation.ElectricalMobility``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``TransportNumberCalculation.none``", "Users provide data via the trans_num_data configuration"
           "``TransportNumberCalculation.ElectricalMobility``", "Calculated from the elec_mobility_data"
       """,
        ),
    )

    CONFIG.declare(
        "equiv_conductivity_calculation",
        ConfigValue(
            default=EquivalentConductivityCalculation.ElectricalMobility,
            domain=In(EquivalentConductivityCalculation),
            description="Ion equivalent conductivity calculation flag",
            doc="""
           Options to account for the total equivalent conductivity of the liquid phase (solution).

           **default** - ``EquivalentConductivityCalculation.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``EquivalentConductivityCalculation.none``", "Users provide data via the equiv_conductivity_data configuration"
           "``EquivalentConductivityCalculation.ElectricalMobility``", "Calculated from the electrical_mobility_data"
       """,
        ),
    )

    CONFIG.declare(
        "material_flow_basis",
        ConfigValue(
            default=MaterialFlowBasis.molar,
            domain=In(MaterialFlowBasis),
            description="Material flow basis",
            doc="""
           Material flow basis options.

           **default** - ``MaterialFlowBasis.molar``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``MaterialFlowBasis.molar``", "molar flowrate as the state variable"
           "``MaterialFlowBasis.mass``", "mass flowrate as the state variable"
           "``MaterialFlowBasis.other``", "other material flowrate as the state variable"
       """,
        ),
    )

    def build(self):
        """
        Callable method for Block construction.
        """
        super().build()

        self._state_block_class = MCASStateBlock

        # phases
        self.Liq = AqueousPhase()

        # list to hold all species (including water)
        self.component_list = Set(dimen=1)

        # components
        self.H2O = Solvent()

        # Other component sets
        self.solute_set = Set(dimen=1)  # All components except Solvent() ("H2O")
        self.cation_set = Set(dimen=1)  # Components with charge >0
        self.anion_set = Set(dimen=1)  # Components with charge <0
        self.neutral_set = Set(dimen=1)  # Components with charge =0
        self.ion_set = Set(dimen=1)  # All Ion Components (cations + anions)

        # Check that solute_list was not left empty
        if self.config.solute_list is None or not len(self.config.solute_list):
            raise ConfigurationError(
                "The solute_list argument was not provided while instantiating the MCAS property model. Provide a list of solutes to solute_list (as a list of strings)."
            )
        # charge_comp = self.config.charge
        charge_comp_abs = {k: abs(v) for k, v in self.config.charge.items()}

        # Group components into different sets
        for j in self.config.solute_list:
            if j == "H2O":
                raise ConfigurationError(
                    "'H2O'is reserved as the default solvent and cannot be a solute."
                )
            # Add valid members of solute_list into IDAES's Solute() class.
            # This triggers the addition of j into component_list and solute_set.
            self.add_component(j, Solute())
            if j in self.config.charge:
                if self.config.charge[j] > 0:
                    # Run a "del_component" and "add_component" to move ion j from IDAES's Solute to Cation class.
                    # Ion j has to be added into Solute to be registered in the component_list and solute_set.
                    # Reference to idaes.core.base.components.
                    self.del_component(j)
                    self.add_component(
                        j,
                        Cation(charge=self.config.charge[j], _electrolyte=True),
                    )
                    self.ion_set.add(j)
                else:
                    # The same comments above apply to anions.
                    self.del_component(j)
                    self.add_component(
                        j,
                        Anion(charge=self.config.charge[j], _electrolyte=True),
                    )
                    self.ion_set.add(j)
            else:
                self.neutral_set.add(j)

        
        # molecular weight
        self.mw_comp = Param(
            self.component_list,
            mutable=True,
            default=18e-3,
            initialize=self.config.mw_data,
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )
        # Stokes radius
        self.molar_volume_phase_comp = Param(
            self.phase_list,
            self.solute_set,
            mutable=True,
            default=1e-5,
            initialize=self.config.molar_volume_data,
            units=pyunits.m**3 / pyunits.mol,
            doc="molar volume of solutes",
        )
        # Assume ~1e-3 Pa*s for pure water
        self.visc_d_phase = Param(
            self.phase_list,
            mutable=True,
            default=1e-3,
            initialize=1e-3,
            units=pyunits.Pa * pyunits.s,
            doc="Fluid viscosity",
        )

        # Ion charge
        self.charge_comp = Param(
            self.ion_set,
            mutable=True,
            initialize=self.config.charge,
            units=pyunits.dimensionless,
            doc="Ion charge",
        )

        self.charge_comp_abs = Param(
            self.ion_set,
            mutable=True,
            initialize=charge_comp_abs,
            units=pyunits.dimensionless,
            doc="Ion charge (absolute value)",
        )

        if self.config.diffus_calculation == DiffusivityCalculation.none:
            self.diffus_phase_comp = Var(
                self.phase_list,
                self.solute_set,
                initialize=self.config.diffusivity_data,
                units=pyunits.m**2 * pyunits.s**-1,
                doc="Mass diffusivity of solute components",
            )
            self.diffus_phase_comp.fix()

        # Mass density
        # For constant density
        self.dens_mass_const = Param(
            mutable=True,
            default=1000,
            initialize=1000,
            units=pyunits.kg / pyunits.m**3,
            doc="Mass density used in DensityCalculation.constant",
        )

        # Parameters for seawater density, eq. 8 in Sharqawy et al. (2010)
        dens_units = pyunits.kg / pyunits.m**3
        t_inv_units = pyunits.K**-1

        self.dens_mass_param_A1 = Var(
            within=Reals,
            initialize=9.999e2,
            units=dens_units,
            doc="Mass density parameter A1",
        )
        self.dens_mass_param_A2 = Var(
            within=Reals,
            initialize=2.034e-2,
            units=dens_units * t_inv_units,
            doc="Mass density parameter A2",
        )
        self.dens_mass_param_A3 = Var(
            within=Reals,
            initialize=-6.162e-3,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter A3",
        )
        self.dens_mass_param_A4 = Var(
            within=Reals,
            initialize=2.261e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter A4",
        )
        self.dens_mass_param_A5 = Var(
            within=Reals,
            initialize=-4.657e-8,
            units=dens_units * t_inv_units**4,
            doc="Mass density parameter A5",
        )
        self.dens_mass_param_B1 = Var(
            within=Reals,
            initialize=8.020e2,
            units=dens_units,
            doc="Mass density parameter B1",
        )
        self.dens_mass_param_B2 = Var(
            within=Reals,
            initialize=-2.001,
            units=dens_units * t_inv_units,
            doc="Mass density parameter B2",
        )
        self.dens_mass_param_B3 = Var(
            within=Reals,
            initialize=1.677e-2,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter B3",
        )
        self.dens_mass_param_B4 = Var(
            within=Reals,
            initialize=-3.060e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter B4",
        )
        self.dens_mass_param_B5 = Var(
            within=Reals,
            initialize=-1.613e-5,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter B5",
        )

        # traditional parameters are the only Vars currently on the block and should be fixed
        for v in self.component_objects(Var):
            v.fix()

        # ---default scaling---
        self.set_default_scaling("temperature", 1e-2)
        self.set_default_scaling("pressure", 1e-4)
        self.set_default_scaling("dens_mass_phase", 1e-3, index="Liq")
        self.set_default_scaling("visc_d_phase", 1e3, index="Liq")
        self.set_default_scaling("diffus_phase_comp", 1e10, index="Liq")

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "flow_mol_phase_comp": {"method": "_flow_mol_phase_comp"},
                "temperature": {"method": None},
                "pressure": {"method": None},
                "flow_mass_phase_comp": {"method": "_flow_mass_phase_comp"},
                "mass_frac_phase_comp": {"method": "_mass_frac_phase_comp"},
                "dens_mass_phase": {"method": "_dens_mass_phase"},
                "flow_vol": {"method": "_flow_vol"},
                "flow_vol_phase": {"method": "_flow_vol_phase"},
                "conc_mol_phase_comp": {"method": "_conc_mol_phase_comp"},
                "conc_mass_phase_comp": {"method": "_conc_mass_phase_comp"},
                "diffus_phase_comp": {"method": "_diffus_phase_comp"},
                "visc_d_phase": {"method": "_visc_d_phase"},
                "pressure_osm_phase": {"method": "_pressure_osm_phase"},
                "mw_comp": {"method": "_mw_comp"},
            }
        )

        obj.define_custom_properties(
            {
                "flow_equiv_phase_comp": {"method": "_flow_equiv_phase_comp"},
                "charge_comp": {"method": "_charge_comp"},
                "charge_comp_abs": {"method": "_charge_comp_abs"},
                "conc_equiv_phase_comp": {"method": "_conc_equiv_phase_comp"},
                "equiv_conductivity_phase": {"method": "_equiv_conductivity_phase"},
                "elec_cond_phase": {"method": "_elec_cond_phase"},
                "elec_resistance_phase": {"method": "_elec_resistance_phase"},
                "dens_mass_solvent": {"method": "_dens_mass_solvent"},
                "molar_volume_phase_comp": {"method": "_molar_volume_phase_comp"},
                "elec_mobility_phase_comp": {"method": "_elec_mobility_phase_comp"},
                "trans_num_phase_comp": {"method": "_trans_num_phase_comp"},
            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _MCASStateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a whole, rather
    than individual elements of indexed Property Blocks.
    """

    def initialize(
        self,
        state_args=None,
        state_vars_fixed=False,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Initialization routine for property package.

        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered through the control
                         volume, and if initial guesses were not provided at the unit model
                         level, the control volume passes the inlet values as initial guess.
                         The keys for the state_args dictionary are:
                         flow_mol_phase_comp : value to initialize phase component flows;
                         pressure : value at which to initialize pressure;
                         temperature : value at which to initialize temperature.
            outlvl : sets output level of initialization routine (default=idaeslog.NOTSET)
            optarg : solver options dictionary object (default=None)
            state_vars_fixed : Flag to denote if state vars have already
                               been fixed.
                               - True - states have already been fixed by the control volume
                               1D. Control volume 0D does not fix the state vars, so will be
                               False if this state block is used with 0D blocks.
                               - False - states have not been fixed. The state block will deal
                               with fixing/unfixing.
            solver : Solver object to use during initialization. If None
                     is provided, it will use the default solver for IDAES (default = None)
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - state variables are not unfixed, and a dict of returned
                         containing flags for which states were fixed during initialization.
                         - False - state variables are unfixed after initialization by calling
                         the release_state method.

        Returns:
            If hold_states is True, returns a dict containing flags for which states were fixed
            during initialization.
        """
        # Get loggers
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="properties")

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix state variables
        flags = fix_state_vars(self, state_args)

        # initialize vars calculated from state vars
        for k in self.keys():
            # Vars indexed by phase and component_list
            for j in self[k].params.component_list:
                if self[k].params.config.material_flow_basis == MaterialFlowBasis.molar:
                    if self[k].is_property_constructed("flow_mass_phase_comp"):
                        self[k].flow_mass_phase_comp["Liq", j].set_value(
                            self[k].flow_mol_phase_comp["Liq", j]
                            * self[k].params.mw_comp[j]
                        )
                elif (
                    self[k].params.config.material_flow_basis == MaterialFlowBasis.mass
                ):
                    if self[k].is_property_constructed("flow_mol_phase_comp"):
                        self[k].flow_mol_phase_comp["Liq", j].set_value(
                            self[k].flow_mass_phase_comp["Liq", j]
                            / self[k].params.mw_comp[j]
                        )
                else:
                    pass

                if self[k].is_property_constructed("mass_frac_phase_comp"):
                    self[k].mass_frac_phase_comp["Liq", j].set_value(
                        self[k].flow_mol_phase_comp["Liq", j]
                        * self[k].params.mw_comp[j]
                        / sum(
                            self[k].flow_mol_phase_comp["Liq", j]
                            * self[k].params.mw_comp[j]
                            for j in self[k].params.component_list
                        )
                    )
                if self[k].is_property_constructed("conc_mass_phase_comp"):
                    self[k].conc_mass_phase_comp["Liq", j].set_value(
                        self[k].dens_mass_phase["Liq"]
                        * self[k].mass_frac_phase_comp["Liq", j]
                    )

                if self[k].is_property_constructed("conc_mol_phase_comp"):
                    self[k].conc_mol_phase_comp["Liq", j].set_value(
                        self[k].conc_mass_phase_comp["Liq", j]
                        / self[k].params.mw_comp[j]
                    )

            # Vars indexed by ion_set
            for j in self[k].params.ion_set:
                if (
                    self[k].is_property_constructed("elec_mobility_phase_comp")
                    and self[k].params.config.elec_mobility_calculation
                    == ElectricalMobilityCalculation.EinsteinRelation
                ):
                    self[k].elec_mobility_phase_comp["Liq", j].set_value(
                        self[k].diffus_phase_comp["Liq", j]
                        * self[k].charge_comp_abs[j]
                        * Constants.faraday_constant
                        / (Constants.gas_constant * self[k].temperature)
                    )
                if self[k].is_property_constructed("conc_equiv_phase_comp"):
                    self[k].conc_equiv_phase_comp["Liq", j].set_value(
                        self[k].conc_mol_phase_comp["Liq", j]
                        / self[k].charge_comp_abs[j]
                    )
                if self[k].is_property_constructed("flow_equiv_phase_comp"):
                    self[k].flow_equiv_phase_comp["Liq", j].set_value(
                        self[k].flow_mol_phase_comp["Liq", j]
                        * self[k].charge_comp_abs[j]
                    )
            # Vars not indexed or indexed only by phase
            if self[k].is_property_constructed("flow_vol_phase"):
                self[k].flow_vol_phase["Liq"].set_value(
                    sum(
                        self[k].flow_mol_phase_comp["Liq", j]
                        * self[k].params.mw_comp[j]
                        for j in self[k].params.component_list
                    )
                    / self[k].dens_mass_phase["Liq"]
                )
            if self[k].is_property_constructed("pressure_osm_phase"):
                self[k].pressure_osm_phase["Liq"].set_value(
                    sum(
                        self[k].conc_mol_phase_comp["Liq", j]
                        for j in self[k].params.solute_set
                    )
                    * Constants.gas_constant
                    * self[k].temperature
                )

            if (
                self[k].is_property_constructed("equiv_conductivity_phase")
                and self[k].params.config.equiv_conductivity_calculation
                == EquivalentConductivityCalculation.ElectricalMobility
            ):
                self[k].equiv_conductivity_phase["Liq"].set_value(
                    sum(
                        Constants.faraday_constant
                        * self[k].charge_comp_abs[j]
                        * self[k].elec_mobility_phase_comp["Liq", j]
                        * self[k].conc_mol_phase_comp["Liq", j]
                        for j in self[k].params.ion_set
                    )
                    / sum(
                        self[k].charge_comp_abs[j]
                        * self[k].conc_mol_phase_comp["Liq", j]
                        for j in self[k].params.cation_set
                    )
                )
            if self[k].is_property_constructed("elec_cond_phase"):
                self[k].elec_cond_phase["Liq"].set_value(
                    self[k].equiv_conductivity_phase["Liq"]
                    * sum(
                        self[k].charge_comp_abs[j]
                        * self[k].conc_mol_phase_comp["Liq", j]
                        for j in self[k].params.cation_set
                    )
                )
            if self[k].is_property_constructed("elec_resistance_phase"):
                self[k].elec_resistance_phase["Liq"].set_value(
                    1 / self[k].elec_cond_phase["Liq"]
                )

        # Check when the state vars are fixed already result in dof 0
        for k in self.keys():
            dof = degrees_of_freedom(self[k])
            if dof != 0:
                raise InitializationError(
                    "\nWhile initializing {sb_name}, the degrees of freedom "
                    "are {dof}, when zero is required. \nInitialization assumes "
                    "that the state variables should be fixed and that no other "
                    "variables are fixed. \nIf other properties have a "
                    "predetermined value, use the calculate_state method "
                    "before using initialize to determine the values for "
                    "the state variables and avoid fixing the property variables."
                    "".format(sb_name=self.name, dof=dof)
                )

        # ---------------------------------------------------------------------
        skip_solve = True  # skip solve if only state variables are present
        for k in self.keys():
            if number_unfixed_variables(self[k]) != 0:
                skip_solve = False

        if not skip_solve:
            # Initialize properties
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            init_log.info_high(
                "Property initialization: {}.".format(idaeslog.condition(results))
            )

        # If input block, return flags, else release state
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                self.release_state(flags)

        if (not skip_solve) and (not check_optimal_termination(results)):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please "
                f"check the output logs for more information."
            )

    def release_state(self, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialisation.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of logging
        """
        # Unfix state variables
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        revert_state_vars(self, flags)
        init_log.info_high("{} State Released.".format(self.name))

    def calculate_state(
        self,
        var_args=None,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Solves state blocks given a set of variables and their values. These variables can be
        state variables or properties. This method is typically used before initialization to
        solve for state variables because non-state variables (i.e. properties) cannot be fixed
        in initialization routines.

        Keyword Arguments:
            var_args : dictionary with variables and their values, they
                       can be state variables or properties
                       {(VAR_NAME, INDEX): VALUE}
            hold_state : flag indicating whether all of the state
                         variables should be fixed after calculate state.
                         True - State variables will be fixed.
                         False - State variables will remain unfixed, unless already fixed.
            outlvl : idaes logger object that sets output level of solve
                     call (default=idaeslog.NOTSET)
            solver : solver name string if None is provided the default
                     solver for IDAES will be used (default = None)
            optarg : solver options dictionary object (default={})

        Returns:
            results object from state block solve
        """
        # Get logger
        solve_log = idaeslog.getSolveLogger(self.name, level=outlvl, tag="properties")

        # Initialize at current state values (not user provided)
        self.initialize(solver=solver, optarg=optarg, outlvl=outlvl)

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix variables and check degrees of freedom
        flags = (
            {}
        )  # dictionary noting which variables were fixed and their previous state
        for k in self.keys():
            sb = self[k]
            for (v_name, ind), val in var_args.items():
                var = getattr(sb, v_name)
                if iscale.get_scaling_factor(var[ind]) is None:
                    _log.warning(
                        "While using the calculate_state method on {sb_name}, variable {v_name} "
                        "was provided as an argument in var_args, but it does not have a scaling "
                        "factor. This suggests that the calculate_scaling_factor method has not been "
                        "used or the variable was created on demand after the scaling factors were "
                        "calculated. It is recommended to touch all relevant variables (i.e. call "
                        "them or set an initial value) before using the calculate_scaling_factor "
                        "method.".format(v_name=v_name, sb_name=sb.name)
                    )
                if var[ind].is_fixed():
                    flags[(k, v_name, ind)] = True
                    if value(var[ind]) != val:
                        raise ConfigurationError(
                            "While using the calculate_state method on {sb_name}, {v_name} was "
                            "fixed to a value {val}, but it was already fixed to value {val_2}. "
                            "Unfix the variable before calling the calculate_state "
                            "method or update var_args."
                            "".format(
                                sb_name=sb.name,
                                v_name=var.name,
                                val=val,
                                val_2=value(var[ind]),
                            )
                        )
                else:
                    flags[(k, v_name, ind)] = False
                    var[ind].fix(val)

            if degrees_of_freedom(sb) != 0:
                raise RuntimeError(
                    "While using the calculate_state method on {sb_name}, the degrees "
                    "of freedom were {dof}, but 0 is required. Check var_args and ensure "
                    "the correct fixed variables are provided."
                    "".format(sb_name=sb.name, dof=degrees_of_freedom(sb))
                )

        # Solve
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            solve_log.info_high(
                "Calculate state: {}.".format(idaeslog.condition(results))
            )

        if not check_optimal_termination(results):
            _log.error(
                "While using the calculate_state method on {sb_name}, the solver failed "
                "to converge to an optimal solution. This suggests that the user provided "
                "infeasible inputs, or that the model is poorly scaled, poorly initialized, "
                "or degenerate."
            )

        # unfix all variables fixed with var_args
        for (k, v_name, ind), previously_fixed in flags.items():
            if not previously_fixed:
                var = getattr(self[k], v_name)
                var[ind].unfix()

        # fix state variables if hold_state
        if hold_state:
            fix_state_vars(self)

        return results


@declare_process_block_class("MCASStateBlock", block_class=_MCASStateBlock)
class MCASStateBlockData(StateBlockData):
    def build(self):
        """Callable method for Block construction."""
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Add state variables
        self.temperature = Var(
            initialize=298.15,
            bounds=(273.15, 373.15),
            domain=NonNegativeReals,
            units=pyunits.K,
            doc="State temperature",
        )

        self.pressure = Var(
            initialize=101325,
            bounds=(1e5, None),
            domain=NonNegativeReals,
            units=pyunits.Pa,
            doc="State pressure",
        )

    # -----------------------------------------------------------------------------
    # Property Methods
    # Material flow state variables generated via on-demand props
    def _flow_mol_phase_comp(self):
        self.flow_mol_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.s,
            doc="Component molar flow rate",
        )
        if self.params.config.material_flow_basis == MaterialFlowBasis.mass:

            def rule_flow_mol_phase_comp(b, p, j):
                return (
                    b.flow_mass_phase_comp[p, j]
                    == b.flow_mol_phase_comp[p, j] * b.params.mw_comp[j]
                )

            self.eq_flow_mol_phase_comp = Constraint(
                self.params.phase_list,
                self.params.component_list,
                rule=rule_flow_mol_phase_comp,
            )

    def _flow_mass_phase_comp(self):
        self.flow_mass_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.5,
            bounds=(0, None),
            units=pyunits.kg / pyunits.s,
            doc="Component Mass flowrate",
        )
        if self.params.config.material_flow_basis == MaterialFlowBasis.molar:

            def rule_flow_mass_phase_comp(b, p, j):
                return (
                    b.flow_mass_phase_comp[p, j]
                    == b.flow_mol_phase_comp[p, j] * b.params.mw_comp[j]
                )

            self.eq_flow_mass_phase_comp = Constraint(
                self.params.phase_list,
                self.params.component_list,
                rule=rule_flow_mass_phase_comp,
            )

    def _mass_frac_phase_comp(self):
        self.mass_frac_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.5,
            bounds=(0, 1.001),
            units=pyunits.kg / pyunits.kg,
            doc="Mass fraction",
        )

        def rule_mass_frac_phase_comp(b, p, j):
            return b.mass_frac_phase_comp[p, j] == b.flow_mass_phase_comp[p, j] / sum(
                b.flow_mass_phase_comp[p, j] for j in self.params.component_list
            )

        self.eq_mass_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mass_frac_phase_comp,
        )

    def _dens_mass_phase(self):
        self.dens_mass_phase = Var(
            ["Liq"],
            initialize=1e3,
            bounds=(5e2, 2e3),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density",
        )

        def rule_dens_mass_phase(b, p):
            if b.params.config.density_calculation == DensityCalculation.constant:
                add_object_reference(
                    self, "dens_mass_const", self.params.dens_mass_const
                )
                return b.dens_mass_phase[p] == self.dens_mass_const
            elif b.params.config.density_calculation == DensityCalculation.seawater:
                # density, eq. 8 in Sharqawy
                t = b.temperature - 273.15 * pyunits.K
                s = sum(b.mass_frac_phase_comp[p, j] for j in b.params.solute_set)
                dens_mass = (
                    b.dens_mass_solvent
                    + b.params.dens_mass_param_B1 * s
                    + b.params.dens_mass_param_B2 * s * t
                    + b.params.dens_mass_param_B3 * s * t**2
                    + b.params.dens_mass_param_B4 * s * t**3
                    + b.params.dens_mass_param_B5 * s**2 * t**2
                )
                return b.dens_mass_phase[p] == dens_mass

        self.eq_dens_mass_phase = Constraint(["Liq"], rule=rule_dens_mass_phase)

    def _dens_mass_solvent(self):
        self.dens_mass_solvent = Var(
            initialize=1e3,
            bounds=(1, 1e6),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density of pure water",
        )

        def rule_dens_mass_solvent(b):  # density, eq. 8 in Sharqawy
            t = b.temperature - 273.15 * pyunits.K
            dens_mass_w = (
                b.params.dens_mass_param_A1
                + b.params.dens_mass_param_A2 * t
                + b.params.dens_mass_param_A3 * t**2
                + b.params.dens_mass_param_A4 * t**3
                + b.params.dens_mass_param_A5 * t**4
            )
            return b.dens_mass_solvent == dens_mass_w

        self.eq_dens_mass_solvent = Constraint(rule=rule_dens_mass_solvent)

    def _flow_vol_phase(self):
        self.flow_vol_phase = Var(
            self.params.phase_list,
            initialize=0.001,
            bounds=(0, None),
            units=pyunits.m**3 / pyunits.s,
            doc="Volumetric flow rate",
        )

        def rule_flow_vol_phase(b, p):
            return (
                b.flow_vol_phase[p]
                == sum(
                    b.flow_mol_phase_comp[p, j] * b.mw_comp[j]
                    for j in self.params.component_list
                )
                / b.dens_mass_phase[p]
            )

        self.eq_flow_vol_phase = Constraint(
            self.params.phase_list, rule=rule_flow_vol_phase
        )

    def _flow_vol(self):
        def rule_flow_vol(b):
            return sum(b.flow_vol_phase[p] for p in self.params.phase_list)

        self.flow_vol = Expression(rule=rule_flow_vol)

    def _conc_mol_phase_comp(self):
        self.conc_mol_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=500,
            bounds=(0, None),
            units=pyunits.mol * pyunits.m**-3,
            doc="Molar concentration",
        )

        def rule_conc_mol_phase_comp(b, p, j):
            return (
                b.conc_mol_phase_comp[p, j] * b.params.mw_comp[j]
                == b.conc_mass_phase_comp[p, j]
            )

        self.eq_conc_mol_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_conc_mol_phase_comp,
        )

    def _conc_mass_phase_comp(self):
        self.conc_mass_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=10,
            bounds=(0, 2e3),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass concentration",
        )

        def rule_conc_mass_phase_comp(b, p, j):
            return (
                b.conc_mass_phase_comp[p, j]
                == b.dens_mass_phase[p] * b.mass_frac_phase_comp[p, j]
            )

        self.eq_conc_mass_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_conc_mass_phase_comp,
        )

    def _flow_equiv_phase_comp(self):
        self.flow_equiv_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=0.1,
            bounds=(0, None),
            units=pyunits.mol / pyunits.s,
            doc="Component equivalent charge flowrate",
        )

        def rule_flow_equiv_phase_comp(b, p, j):
            return b.flow_equiv_phase_comp[p, j] == b.flow_mol_phase_comp[p, j] * abs(
                b.params.charge_comp[j]
            )

        self.eq_flow_equiv_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_flow_equiv_phase_comp,
        )

    def _conc_equiv_phase_comp(self):
        self.conc_equiv_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=500,
            bounds=(0, None),
            units=pyunits.mol / pyunits.m**3,
            doc="Equivalent charge concentration",
        )

        def rule_conc_equiv_phase_comp(b, p, j):
            return b.conc_equiv_phase_comp[p, j] == b.conc_mol_phase_comp[p, j] * abs(
                b.params.charge_comp[j]
            )

        self.eq_conc_equiv_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_conc_equiv_phase_comp,
        )

    def _molar_volume_phase_comp(self):
        add_object_reference(
            self, "molar_volume_phase_comp", self.params.molar_volume_phase_comp
        )

    def _diffus_phase_comp(self):
        # Retrieve component string names from diffusivity_data configuration
        diffus_data_indices = {i[1] for i in self.params.config.diffusivity_data.keys()}
        # Retrieve component string names from molar_volume_data configuration
        molar_volume_data_indices = {
            i[1] for i in self.params.config.molar_volume_data.keys()
        }
        missing_diffus_ind = [
            i
            for i in self.params.solute_set
            if i not in (molar_volume_data_indices | diffus_data_indices)
        ]

        if self.params.config.diffus_calculation == DiffusivityCalculation.HaydukLaudie:
            # warning for components with neither diffusivity_data nor molar_volume_data entry
            if not missing_diffus_ind == []:
                _log.warning(
                    f"Neither diffusivity_data nor molar_volume_data was provided for {missing_diffus_ind}; "
                    "there will be no diffus_phase_comp properties for these components."
                )
            common_ind = [
                i for i in molar_volume_data_indices if i in diffus_data_indices
            ]
            if not common_ind == []:
                # warning for components whose diffusivity_data will be overwritten by the HaydukLaudie method.
                _log.warning(
                    f"Both diffusivity_data and molar_volume_data were provided for {common_ind}; "
                    f"since the the HaydukLaudie method was selected, the diffus_phase_comp property of these components will "
                    f"be calculated based on their molar_volume_data and overwritten."
                )
            self.diffus_phase_comp = Var(
                self.params.phase_list,
                molar_volume_data_indices | diffus_data_indices,
                initialize=1e-9,
                units=pyunits.m**2 * pyunits.s**-1,
                doc="Mass diffusivity of solute components",
            )
            self.hl_diffus_cont = Param(
                mutable=True,
                default=13.26e-9,
                initialize=13.26e-9,
                units=pyunits.dimensionless,
                doc="Hayduk Laudie correlation constant",
            )
            self.hl_visc_coeff = Param(
                mutable=True,
                default=1.14,
                initialize=1.14,
                units=pyunits.dimensionless,
                doc="Hayduk Laudie viscosity coefficient",
            )
            self.hl_molar_volume_coeff = Param(
                mutable=True,
                default=0.589,
                initialize=0.589,
                units=pyunits.dimensionless,
                doc="Hayduk Laudie molar volume coefficient",
            )

            def rule_diffus_phase_comp(b, p, j):
                if (
                    self.params.config.diffus_calculation
                    == DiffusivityCalculation.HaydukLaudie
                ):
                    if j not in molar_volume_data_indices:
                        b.diffus_phase_comp[p, j].fix(
                            self.params.config.diffusivity_data[p, j]
                        )
                        return Constraint.Skip
                    else:
                        diffus_coeff_inv_units = pyunits.s * pyunits.m**-2
                        visc_solvent_inv_units = pyunits.cP**-1
                        molar_volume_inv_units = pyunits.mol * pyunits.cm**-3
                        return (b.diffus_phase_comp[p, j] * diffus_coeff_inv_units) * (
                            (
                                pyunits.convert(b.visc_d_phase[p], to_units=pyunits.cP)
                                * visc_solvent_inv_units
                            )
                            ** b.hl_visc_coeff
                        ) * (
                            (
                                pyunits.convert(
                                    b.molar_volume_phase_comp[p, j],
                                    to_units=pyunits.cm**3 * pyunits.mol**-1,
                                )
                                * molar_volume_inv_units
                            )
                            ** b.hl_molar_volume_coeff
                        ) == b.hl_diffus_cont

            self.eq_diffus_phase_comp = Constraint(
                self.params.phase_list,
                molar_volume_data_indices | diffus_data_indices,
                rule=rule_diffus_phase_comp,
            )

        elif self.params.config.diffus_calculation == DiffusivityCalculation.none:
            # warning for components with no diffusivity_data entry
            if not missing_diffus_ind == []:
                _log.warning(
                    f"Diffusivity data was not provided for {missing_diffus_ind}. "
                )

            add_object_reference(
                self, "diffus_phase_comp", self.params.diffus_phase_comp
            )

    def _visc_d_phase(self):
        add_object_reference(self, "visc_d_phase", self.params.visc_d_phase)

    def _mw_comp(self):
        add_object_reference(self, "mw_comp", self.params.mw_comp)

    def _elec_mobility_phase_comp(self):
        self.elec_mobility_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=5.19e-8,  # default as Na+
            units=pyunits.meter**2 * pyunits.volt**-1 * pyunits.second**-1,
            doc="Ion electrical mobility",
        )

        def rule_elec_mobility_phase_comp(b, p, j):
            if (
                self.params.config.elec_mobility_calculation
                == ElectricalMobilityCalculation.none
            ):
                if (p, j) not in self.params.config.elec_mobility_data.keys():
                    raise ConfigurationError(
                        """ 
                        Missing the "elec_mobility_data" configuration to build the elec_mobility_phase_comp 
                        and/or its derived variables for {} in {}. 
                        Provide this configuration or use ElectricalMobilityCalculation.EinsteinRelation.
                        """.format(
                            j, self.name
                        )
                    )
                else:
                    return (
                        b.elec_mobility_phase_comp[p, j]
                        == self.params.config.elec_mobility_data[p, j]
                        * pyunits.meter**2
                        * pyunits.volt**-1
                        * pyunits.second**-1
                    )
            else:
                if (p, j) not in self.params.config.diffusivity_data.keys():
                    raise ConfigurationError(
                        """
                        Missing a valid diffusivity_data configuration to use EinsteinRelation 
                        to compute the "elec_mobility_phase_comp" for {} in {} . 
                        Provide this configuration or 
                        use another "elec_mobility_calculation" configuration value. """.format(
                            j, self.name
                        )
                    )
                else:
                    if (p, j) in self.params.config.elec_mobility_data.keys():
                        _log.warning(
                            """
                            The provided elec_mobility_data of {} will be overwritten 
                            by the calculated data for {} because the EinsteinRelation 
                            method is selected.""".format(
                                j, self.name
                            )
                        )

                    return b.elec_mobility_phase_comp[p, j] == b.diffus_phase_comp[
                        p, j
                    ] * abs(b.charge_comp[j]) * Constants.faraday_constant / (
                        Constants.gas_constant * b.temperature
                    )

        self.eq_elec_mobility_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_elec_mobility_phase_comp,
        )

    def _charge_comp(self):
        add_object_reference(self, "charge_comp", self.params.charge_comp)

    def _charge_comp_abs(self):
        add_object_reference(self, "charge_comp_abs", self.params.charge_comp_abs)

    def _pressure_osm_phase(self):
        self.pressure_osm_phase = Var(
            self.params.phase_list,
            initialize=1e6,
            bounds=(0, None),
            units=pyunits.Pa,
            doc="van't Hoff Osmotic pressure",
        )

        def rule_pressure_osm_phase(b, p):
            return (
                b.pressure_osm_phase[p]
                == sum(b.conc_mol_phase_comp[p, j] for j in b.params.solute_set)
                * Constants.gas_constant
                * b.temperature
            )

        self.eq_pressure_osm_phase = Constraint(
            self.params.phase_list, rule=rule_pressure_osm_phase
        )

    def _trans_num_phase_comp(self):
        self.trans_num_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=0.5,
            units=pyunits.dimensionless,
            doc="Ion transport number in the liquid phase",
        )

        def rule_trans_num_phase_comp(b, p, j):
            if (
                self.params.config.trans_num_calculation
                == TransportNumberCalculation.none
            ):
                if (p, j) not in self.params.config.trans_num_data.keys():
                    raise ConfigurationError(
                        """ 
                        Missing a valid trans_num_data configuration to build "trans_num_phase_comp" for {} in {}.  
                        Provide this configuration or use another "trans_num_calculation"
                        configuration value to contruct the demanded variable(s))""".format(
                            j, self.name
                        )
                    )
                else:
                    return (
                        b.trans_num_phase_comp[p, j]
                        == self.params.config.trans_num_data[p, j]
                    )
            else:
                if (p, j) in self.params.config.trans_num_data.keys():
                    _log.warning(
                        """
                        The provided trans_num_data of {} will be overritten by the calculated data for {}
                        because "TransportNumberCalculation" is set as "ElectricalMobility".""".format(
                            j, self.name
                        )
                    )
                return b.trans_num_phase_comp[p, j] == abs(
                    b.charge_comp[j]
                ) * b.elec_mobility_phase_comp[p, j] * b.conc_mol_phase_comp[
                    p, j
                ] / sum(
                    abs(b.charge_comp[j])
                    * b.elec_mobility_phase_comp[p, j]
                    * b.conc_mol_phase_comp[p, j]
                    for j in self.params.ion_set
                )

        self.eq_trans_num_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_trans_num_phase_comp,
        )

    def _equiv_conductivity_phase(self):
        self.equiv_conductivity_phase = Var(
            self.params.phase_list,
            initialize=0.5,
            units=pyunits.meter**2 * pyunits.ohm**-1 * pyunits.mol**-1,
            doc="Total equivalent electrical conducitivty of the liquid phase",
        )

        def rule_equiv_conductivity_phase(b, p):
            if (
                self.params.config.equiv_conductivity_calculation
                == EquivalentConductivityCalculation.none
            ):
                if p not in self.params.config.equiv_conductivity_phase_data.keys():
                    raise ConfigurationError(
                        """ 
                        Missing a valid equiv_conductivity_phase_data configuration to build 
                        "equiv_conductivity_phase" and its derived variables for {}. 
                        Provide this configuration or use another "equiv_conductivity_calculation"
                        configuration value to contruct the demanded variable(s))""".format(
                            self.name
                        )
                    )
                else:
                    if len(self.params.ion_set) > 2:
                        _log.warning(
                            """ 
                            Caution should be taken to use a constant solution equivalent conductivity for a multi-electrolyte system.
                            Heterogeneous concentration variation among ions may lead to varying equivalent conductivity and computing
                            the phase equivalent conductivity using the "EquivalentConductivityCalculation.ElectricalMobility" method 
                            is recommended."""
                        )
                    return (
                        b.equiv_conductivity_phase[p]
                        == self.params.config.equiv_conductivity_phase_data[p]
                        * pyunits.meter**2
                        * pyunits.ohm**-1
                        * pyunits.mol**-1
                    )
            elif (
                self.params.config.equiv_conductivity_calculation
                == EquivalentConductivityCalculation.Onsager_Falkenhagen
            ):
                if len(self.params.config.equiv_conductivity_phase_data) != 0:
                    _log.warning(
                        """
                        The provided equiv_conductivity_phase_data will be overritten by the calculated data for {} because
                        "EquivalentConductivityCalculation" is set as "Onsager_Falkenhagen".""".format(
                            self.name
                        )
                    )
                b0 = 0.3286
                b1 = 0.2289
                b2 = 60.32
                EC0 = 126.45
                a = 4
                conc = value(
                    sum(
                        abs(b.charge_comp[j]) * b.conc_mol_phase_comp[p, j]
                        for j in self.params.cation_set
                    )
                )
                return (
                    b.equiv_conductivity_phase[p]
                    == (
                        EC0
                        - (b1 * EC0 + b2)
                        * (1 + b0 * a * (conc / 1000) ** 0.5) ** -1
                        * (conc / 1000) ** 0.5
                    )
                    * 1e-4
                    * pyunits.meter**2
                    * pyunits.ohm**-1
                    * pyunits.mol**-1
                )

            else:
                if len(self.params.config.equiv_conductivity_phase_data) != 0:
                    _log.warning(
                        """
                        The provided equiv_conductivity_phase_data will be overritten by the calculated data for {} because
                        "EquivalentConductivityCalculation" is set as "ElectricalMobility".""".format(
                            self.name
                        )
                    )
                return b.equiv_conductivity_phase[p] * sum(
                    b.charge_comp_abs[j] * b.conc_mol_phase_comp[p, j]
                    for j in self.params.cation_set
                ) == sum(
                    Constants.faraday_constant
                    * b.charge_comp_abs[j]
                    * b.elec_mobility_phase_comp[p, j]
                    * b.conc_mol_phase_comp[p, j]
                    for j in self.params.ion_set
                )

        self.eq_equiv_conductivity_phase = Constraint(
            self.params.phase_list, rule=rule_equiv_conductivity_phase
        )

    def _elec_cond_phase(self):
        self.elec_cond_phase = Var(
            self.params.phase_list,
            initialize=0.1,
            units=pyunits.ohm**-1 * pyunits.meter**-1,
            doc="Electrical conductivity",
        )

        def rule_elec_cond_phase(b, p):
            return b.elec_cond_phase[p] == b.equiv_conductivity_phase[p] * sum(
                abs(b.charge_comp[j]) * b.conc_mol_phase_comp[p, j]
                for j in self.params.cation_set
            )

        self.eq_elec_cond_phase = Constraint(
            self.params.phase_list, rule=rule_elec_cond_phase
        )

    def _elec_resistance_phase(self):
        self.elec_resistance_phase = Var(
            self.params.phase_list,
            initialize=10,
            bounds=(0, None),
            units=pyunits.ohm * pyunits.meter,
            doc="Electrical resistance",
        )

        def rule_elec_resistance_phase(b, p):
            return b.elec_resistance_phase[p] * b.elec_cond_phase[p] == 1

        self.eq_elec_resistance_phase = Constraint(
            self.params.phase_list, rule=rule_elec_resistance_phase
        )

    # -----------------------------------------------------------------------------
    # General Methods
    # NOTE: For scaling in the control volume to work properly, these methods must
    # return a pyomo Var or Expression

    def get_material_flow_terms(self, p, j):
        """Create material flow terms for control volume."""
        if self.params.config.material_flow_basis == MaterialFlowBasis.molar:
            return self.flow_mol_phase_comp[p, j]
        elif self.params.config.material_flow_basis == MaterialFlowBasis.mass:
            return self.flow_mass_phase_comp[p, j]

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.none

    def get_material_flow_basis(self):
        if self.params.config.material_flow_basis == MaterialFlowBasis.molar:
            return MaterialFlowBasis.molar
        elif self.params.config.material_flow_basis == MaterialFlowBasis.mass:
            return MaterialFlowBasis.mass
        else:
            raise PropertyPackageError(
                f"{self.name} MCAS Property Package set to use unsupported material flow basis: {self.get_material_flow_basis()}"
            )

    def define_state_vars(self):
        """Define state vars."""
        if self.params.config.material_flow_basis == MaterialFlowBasis.molar:
            return {
                "flow_mol_phase_comp": self.flow_mol_phase_comp,
                "temperature": self.temperature,
                "pressure": self.pressure,
            }
        elif self.params.config.material_flow_basis == MaterialFlowBasis.mass:
            return {
                "flow_mass_phase_comp": self.flow_mass_phase_comp,
                "temperature": self.temperature,
                "pressure": self.pressure,
            }
        else:
            raise PropertyPackageError(
                f"{self.name} MCAS Property Package set to use unsupported material flow basis: {self.get_material_flow_basis()}"
            )

    # -----------------------------------------------------------------------------
    # Scaling methods
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # setting scaling factors for variables

        # default scaling factors have already been set with
        # idaes.core.property_base.calculate_scaling_factors()
        # for the following variables: pressure,
        # temperature, dens_mass, visc_d_phase, diffus_phase_comp

        for j, v in self.mw_comp.items():
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(self.mw_comp[j], value(v) ** -1)
        # the following variables should have users' input of scaling factors;
        # missing input triggers a warning

        if self.params.config.material_flow_basis == MaterialFlowBasis.molar:
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.flow_mol_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(
                        self.flow_mol_phase_comp["Liq", j], default=1, warning=True
                    )
                    iscale.set_scaling_factor(self.flow_mol_phase_comp["Liq", j], sf)

            if self.is_property_constructed("flow_mass_phase_comp"):
                for j in self.params.component_list:
                    if (
                        iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", j])
                        is None
                    ):
                        sf = iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", j]
                        ) * iscale.get_scaling_factor(self.mw_comp[j])
                        iscale.set_scaling_factor(
                            self.flow_mass_phase_comp["Liq", j], sf
                        )

        elif self.params.config.material_flow_basis == MaterialFlowBasis.mass:
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(
                        self.flow_mass_phase_comp["Liq", j], default=1, warning=True
                    )
                    iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", j], sf)

            if self.is_property_constructed("flow_mol_phase_comp"):
                for j in self.params.component_list:
                    if (
                        iscale.get_scaling_factor(self.flow_mol_phase_comp["Liq", j])
                        is None
                    ):
                        sf = iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(self.mw_comp[j])
                        iscale.set_scaling_factor(
                            self.flow_mol_phase_comp["Liq", j], sf
                        )

        # The following variables and parameters have computed scaling factors;
        # Users do not have to input scaling factors but, if they do, their value
        # will override.
        if self.is_property_constructed("flow_equiv_phase_comp"):
            for j in self.flow_equiv_phase_comp.keys():
                if iscale.get_scaling_factor(self.flow_equiv_phase_comp[j]) is None:
                    sf = iscale.get_scaling_factor(self.flow_mol_phase_comp[j])
                    iscale.set_scaling_factor(self.flow_equiv_phase_comp[j], sf)

        if self.is_property_constructed("diffus_phase_comp"):
            for ind, v in self.diffus_phase_comp.items():
                if iscale.get_scaling_factor(v) is None:
                    if ind in self.params.config.diffusivity_data.keys():
                        sf = self.params.config.diffusivity_data[ind] ** -1
                    else:
                        sf = 1e10
                    iscale.set_scaling_factor(self.diffus_phase_comp[ind], sf)
        if self.is_property_constructed("dens_mass_solvent"):
            if iscale.get_scaling_factor(self.dens_mass_solvent) is None:
                iscale.set_scaling_factor(self.dens_mass_solvent, 1e-3)
        if self.is_property_constructed("dens_mass_phase"):
            for p, v in self.dens_mass_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.dens_mass_phase[p], 1e-3)
        if self.is_property_constructed("visc_d_phase"):
            for p, v in self.visc_d_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.visc_d_phase[p], 1e3)
        if self.is_property_constructed("flow_vol_phase"):
            sf = (
                iscale.get_scaling_factor(
                    self.flow_mol_phase_comp["Liq", "H2O"], default=1
                )
                * iscale.get_scaling_factor(self.mw_comp["H2O"])
                / iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
            )
            iscale.set_scaling_factor(self.flow_vol_phase, sf)

        if self.is_property_constructed("flow_vol"):
            sf = iscale.get_scaling_factor(self.flow_vol_phase)
            iscale.set_scaling_factor(self.flow_vol, sf)

        if self.is_property_constructed("mass_frac_phase_comp"):
            for j in self.params.component_list:
                comp = self.params.get_component(j)
                if (
                    iscale.get_scaling_factor(self.mass_frac_phase_comp["Liq", j])
                    is None
                ):
                    if comp.is_solute():
                        sf_mol_frac = iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", "H2O"]
                        )
                        sf = (
                            iscale.get_scaling_factor(self.mw_comp[j])
                            * 1.8e-2
                            * sf_mol_frac
                        )

                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], sf
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], 1
                        )

        if self.is_property_constructed("conc_mass_phase_comp"):
            for j in self.params.component_list:
                sf_dens = iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
                if (
                    iscale.get_scaling_factor(self.conc_mass_phase_comp["Liq", j])
                    is None
                ):
                    if j == "H2O":
                        # solvents typically have a mass fraction between 0.5-1
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j], sf_dens
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j],
                            sf_dens
                            * iscale.get_scaling_factor(
                                self.mass_frac_phase_comp["Liq", j]
                            ),
                        )

        if self.is_property_constructed("conc_mol_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(
                        self.conc_mass_phase_comp["Liq", j]
                    ) / iscale.get_scaling_factor(self.mw_comp[j])
                    iscale.set_scaling_factor(self.conc_mol_phase_comp["Liq", j], sf)
                    # sf = iscale.get_scaling_factor(
                    #     self.flow_mol_phase_comp["Liq", j]
                    # ) / iscale.get_scaling_factor(self.flow_vol_phase["Liq"])

        if self.is_property_constructed("conc_equiv_phase_comp"):
            for j in self.params.ion_set:
                if (
                    iscale.get_scaling_factor(self.conc_equiv_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                    iscale.set_scaling_factor(self.conc_equiv_phase_comp["Liq", j], sf)

        if self.is_property_constructed("pressure_osm_phase"):
            if iscale.get_scaling_factor(self.pressure_osm_phase) is None:
                sf_gas_constant = value(1 / Constants.gas_constant)
                sf_temp = iscale.get_scaling_factor(self.temperature)
                sf_conc_mol = (
                    sum(
                        iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                        ** -1
                        for j in self.params.solute_set
                    )
                ) ** -1
                sf = sf_gas_constant * sf_temp * sf_conc_mol
                iscale.set_scaling_factor(self.pressure_osm_phase, 1 * sf)

        if self.is_property_constructed("elec_mobility_phase_comp"):
            for ind, v in self.elec_mobility_phase_comp.items():
                if iscale.get_scaling_factor(v) is None:
                    if (
                        self.params.config.elec_mobility_calculation
                        == ElectricalMobilityCalculation.EinsteinRelation
                    ):
                        sf = iscale.get_scaling_factor(self.diffus_phase_comp[ind]) / 40
                    else:
                        sf = self.params.config.elec_mobility_data[ind] ** -1
                    iscale.set_scaling_factor(self.elec_mobility_phase_comp[ind], sf)

        if self.is_property_constructed("trans_num_phase_comp"):
            for ind, v in self.trans_num_phase_comp.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.trans_num_phase_comp[ind], 10)
        if self.is_property_constructed("equiv_conductivity_phase"):
            for ind, v in self.equiv_conductivity_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    if (
                        self.params.config.equiv_conductivity_calculation
                        == EquivalentConductivityCalculation.ElectricalMobility
                    ):
                        sf = (
                            1
                            / 96485
                            * sum(
                                iscale.get_scaling_factor(
                                    self.elec_mobility_phase_comp["Liq", j]
                                )
                                ** -1
                                * iscale.get_scaling_factor(
                                    self.conc_mol_phase_comp["Liq", j]
                                )
                                ** -1
                                for j in self.params.ion_set
                            )
                            ** -1
                            / sum(
                                iscale.get_scaling_factor(
                                    self.conc_mol_phase_comp["Liq", j]
                                )
                                ** -1
                                for j in self.params.cation_set
                            )
                            ** -1
                        )
                    elif (
                        self.params.config.equiv_conductivity_calculation
                        == EquivalentConductivityCalculation.Onsager_Falkenhagen
                    ):
                        sf = 100

                    else:
                        sf = self.params.config.equiv_conductivity_phase_data[ind] ** -1
                    iscale.set_scaling_factor(self.equiv_conductivity_phase[ind], sf)

        if self.is_property_constructed("elec_cond_phase"):
            if iscale.get_scaling_factor(self.elec_cond_phase) is None:
                for ind, v in self.elec_cond_phase.items():
                    sf_equiv_cond_phase = iscale.get_scaling_factor(
                        self.equiv_conductivity_phase[ind]
                    )
                    sf_conc_mol_z = (
                        sum(
                            iscale.get_scaling_factor(
                                self.conc_mol_phase_comp["Liq", j]
                            )
                            ** -1
                            * iscale.get_scaling_factor(self.charge_comp[j]) ** -1
                            for j in self.params.cation_set
                        )
                        ** -1
                    )
                    sf = sf_equiv_cond_phase * sf_conc_mol_z
                    iscale.set_scaling_factor(self.elec_cond_phase[ind], sf)

        if self.is_property_constructed("elec_resistance_phase"):
            for ind, v in self.elec_resistance_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    sf = iscale.get_scaling_factor(self.elec_cond_phase[ind]) ** -1
                    # print(f"scaling factor for elec_resistance_phase {ind} is {sf}")
                    iscale.set_scaling_factor(self.elec_resistance_phase[ind], sf)
        # transforming constraints
        transform_property_constraints(self)

        if hasattr(self, "eq_diffus_phase_comp"):
            for ind, v in self.eq_diffus_phase_comp.items():
                iscale.constraint_scaling_transform(v, 1e9)
