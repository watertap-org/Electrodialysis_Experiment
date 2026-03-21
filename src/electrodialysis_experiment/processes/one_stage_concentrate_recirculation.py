from __future__ import annotations

from pathlib import Path

from pyomo.environ import Constraint, Expression, NonNegativeReals, Var, units, value
from pyomo.network import Arc
import yaml

from idaes.core import declare_process_block_class
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Mixer, Product, Separator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslogger
from watertap.unit_models.pressure_changer import Pump

from electrodialysis_experiment.processes.base import ED_base
from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePassData,
)
from electrodialysis_experiment.schema.config.process_config_schema import (
    OneStageSinglePassConfig,
)

_log = idaeslogger.getIdaesLogger(__name__)


@declare_process_block_class("OneStageConcentrateRecirculation")
class OneStageConcentrateRecirculationData(OneStageSinglePassData):
    """
    One-stage ED process with concentrate recirculation (feed-and-bleed).

    Topology summary
    ---------------
    feed -> sepa0 -> (diluate branch to ED, concentrate branch to mixer)
    ED concentrate outlet -> sepa1 -> (bleed to disp, recycle to mixer)
    mixer outlet -> pump0 -> ED concentrate inlet
    """

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OneStageConcentrateRecirculation":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = OneStageSinglePassConfig(**data)
        proc = OneStageConcentrateRecirculation(process_cfg=cfg)
        return proc

    def _build_units(self):
        ed_config = self.config.process_cfg.ed_stack

        self.fs.feed = Feed(property_package=self.fs.properties)
        self.fs.sepa0 = Separator(
            property_package=self.fs.properties,
            outlet_list=["to_dil_in", "to_conc_in0"],
        )
        self.fs.mix0 = Mixer(
            property_package=self.fs.properties,
            energy_mixing_type=MixingType.none,
            momentum_mixing_type=MomentumMixingType.none,
            inlet_list=["from_feed", "from_conc_out"],
        )

        self.fs.pump0 = Pump(property_package=self.fs.properties)
        self.fs.pump0.del_component("ratioP")
        self.fs.pump0.del_component("ratioP_calculation")

        self.fs.pump1 = Pump(property_package=self.fs.properties)
        self.fs.pump1.del_component("ratioP")
        self.fs.pump1.del_component("ratioP_calculation")

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
        self.fs.EDstack = ED_base(property_package=self.fs.properties, **ed_kwargs)

        self.fs.sepa1 = Separator(
            property_package=self.fs.properties,
            outlet_list=["to_disp", "to_conc_in1"],
        )
        self.fs.prod = Product(property_package=self.fs.properties)
        self.fs.disp = Product(property_package=self.fs.properties)

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

    def _add_expressions_and_constraints(self):
        super()._add_expressions_and_constraints()
        self.fs.del_component(self.fs.recovery_vol_H2O)
        self.fs.recovery_vol_H2O = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=units.dimensionless,
            doc="Flowsheet water recovery by volume for feed-and-bleed operation",
        )
        self.fs.eq_recovery_vol_H2O = Constraint(
            expr=self.fs.recovery_vol_H2O
            * self.fs.feed.properties[0].flow_vol_phase["Liq"]
            - self.fs.prod.properties[0].flow_vol_phase["Liq"]
            == 0
        )
        self.fs.concentrate_recycle_fraction = Expression(
            expr=self.fs.sepa1.split_fraction[0, "to_conc_in1"]
        )
        self.fs.concentrate_bleed_fraction = Expression(
            expr=self.fs.sepa1.split_fraction[0, "to_disp"]
        )

    def _wire_arcs(self):
        self.fs.arc0 = Arc(source=self.fs.feed.outlet, destination=self.fs.sepa0.inlet)
        self.fs.arc1b = Arc(source=self.fs.sepa0.to_dil_in, destination=self.fs.pump1.inlet)
        self.fs.arc1f = Arc(source=self.fs.pump1.outlet, destination=self.fs.EDstack.inlet_diluate)
        self.fs.arc2 = Arc(source=self.fs.sepa0.to_conc_in0, destination=self.fs.mix0.from_feed)
        self.fs.arc3b = Arc(source=self.fs.mix0.outlet, destination=self.fs.pump0.inlet)
        self.fs.arc3f = Arc(source=self.fs.pump0.outlet, destination=self.fs.EDstack.inlet_concentrate)
        self.fs.arc4 = Arc(source=self.fs.EDstack.outlet_diluate, destination=self.fs.prod.inlet)
        self.fs.arc5 = Arc(source=self.fs.EDstack.outlet_concentrate, destination=self.fs.sepa1.inlet)
        self.fs.arc6 = Arc(source=self.fs.sepa1.to_disp, destination=self.fs.disp.inlet)
        self.fs.arc7 = Arc(source=self.fs.sepa1.to_conc_in1, destination=self.fs.mix0.from_conc_out)

        from pyomo.environ import TransformationFactory

        TransformationFactory("network.expand_arcs").apply_to(self.fs)

    def _initialize_units(self, solver=None, optarg=None):
        iscale.calculate_scaling_factors(self.fs)

        self.fs.feed.initialize(solver=solver, optarg=optarg)
        propagate_state(self.fs.arc0)
        self.fs.sepa0.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc1b)
        self.fs.pump1.deltaP[0].fix(2e5)
        self.fs.pump1.initialize(solver=solver, optarg=optarg)
        self.fs.pump1.deltaP[0].unfix()

        propagate_state(destination=self.fs.pump0.inlet, source=self.fs.pump1.inlet)
        self.fs.pump0.deltaP[0].fix(2e5)
        self.fs.pump0.initialize(solver=solver, optarg=optarg)
        self.fs.pump0.deltaP[0].unfix()

        propagate_state(self.fs.arc1f)
        propagate_state(self.fs.arc3f)
        self.fs.EDstack.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc4)
        self.fs.prod.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc5)
        recycle_frac_var = self.fs.sepa1.split_fraction[0, "to_conc_in1"]
        recycle_frac_was_fixed = recycle_frac_var.fixed
        if not recycle_frac_was_fixed:
            init_recycle_frac = max(2 - value(self.fs.recovery_vol_H2O) ** -1, 1e-8)
            recycle_frac_var.fix(init_recycle_frac)
        self.fs.sepa1.initialize(solver=solver, optarg=optarg)
        if not recycle_frac_was_fixed:
            recycle_frac_var.unfix()

        propagate_state(self.fs.arc6)
        self.fs.disp.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc7)
        propagate_state(self.fs.arc3b, direction="backward")
        propagate_state(self.fs.arc2)
        self.fs.mix0.initialize(solver=solver, optarg=optarg)

        if hasattr(self.fs, "costing"):
            self.fs.costing.initialize()

    def set_feed_split_to_diluate(self, split_fraction: float, *, fix: bool = False):
        if not (0.0 < split_fraction < 1.0):
            raise ValueError("split_fraction must be in (0, 1).")
        var = self.fs.sepa0.split_fraction[0, "to_dil_in"]
        var.set_value(split_fraction)
        if fix:
            var.fix()

    def set_concentrate_recycle_fraction(self, recycle_fraction: float, *, fix: bool = False):
        if not (0.0 < recycle_fraction < 1.0):
            raise ValueError("recycle_fraction must be in (0, 1).")
        var = self.fs.sepa1.split_fraction[0, "to_conc_in1"]
        var.set_value(recycle_fraction)
        if fix:
            var.fix()


OneStageConcentrateRecirculation.from_yaml = (
    OneStageConcentrateRecirculationData.from_yaml
)
