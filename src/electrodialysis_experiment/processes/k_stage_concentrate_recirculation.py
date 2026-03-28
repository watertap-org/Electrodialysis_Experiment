from __future__ import annotations

from pathlib import Path

from pyomo.environ import (
    Block,
    Constraint,
    Expression,
    NonNegativeReals,
    TransformationFactory,
    Var,
    units,
    value,
)
from pyomo.network import Arc
import yaml

from idaes.core import declare_process_block_class
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Mixer, Product, Separator
from idaes.models.unit_models.mixer import MixingType, MomentumMixingType
import idaes.core.util.scaling as iscale
from watertap.unit_models.pressure_changer import Pump

from electrodialysis_experiment.processes.base import ED_base
from electrodialysis_experiment.processes.k_stage_single_pass import KStageSinglePassData
from electrodialysis_experiment.schema.config.process_config_schema import (
    KStageSinglePassConfig,
)


@declare_process_block_class("KStageConcentrateRecirculation")
class KStageConcentrateRecirculationData(KStageSinglePassData):
    """
    K-stage ED process with concentrate recirculation around the full stage train.

    Recycle loop
    ------------
    final-stage concentrate outlet -> separator -> recycle stream -> mixer ->
    stage-1 concentrate inlet.
    """

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KStageConcentrateRecirculation":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = KStageSinglePassConfig(**data)
        proc = KStageConcentrateRecirculation(process_cfg=cfg)
        return proc

    def _build_units(self):
        fs = self.fs
        fs.feed = Feed(property_package=fs.properties)
        fs.sepa0 = Separator(
            property_package=fs.properties,
            outlet_list=["to_dil_in", "to_conc_in0"],
        )
        fs.mix0 = Mixer(
            property_package=fs.properties,
            energy_mixing_type=MixingType.none,
            momentum_mixing_type=MomentumMixingType.none,
            inlet_list=["from_feed", "from_conc_out"],
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
            fs.EDstack[stage_num].unit = ED_base(
                property_package=fs.properties,
                **self._build_ed_kwargs(ed_config),
            )

        fs.sepa1 = Separator(
            property_package=fs.properties,
            outlet_list=["to_disp", "to_conc_in1"],
        )
        fs.prod = Product(property_package=fs.properties)
        fs.disp = Product(property_package=fs.properties)

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
        stages = self._stage_indices()
        self.fs.arc0 = Arc(source=self.fs.feed.outlet, destination=self.fs.sepa0.inlet)

        self.fs.arc1b = Arc(source=self.fs.sepa0.to_dil_in, destination=self.fs.pump1.inlet)
        self.fs.arc1f = Arc(source=self.fs.pump1.outlet, destination=self._ed(stages[0]).inlet_diluate)

        self.fs.arc2 = Arc(source=self.fs.sepa0.to_conc_in0, destination=self.fs.mix0.from_feed)
        self.fs.arc3b = Arc(source=self.fs.mix0.outlet, destination=self.fs.pump0.inlet)
        self.fs.arc3f = Arc(source=self.fs.pump0.outlet, destination=self._ed(stages[0]).inlet_concentrate)

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

        self.fs.arc4 = Arc(source=self._ed(stages[-1]).outlet_diluate, destination=self.fs.prod.inlet)
        self.fs.arc5 = Arc(source=self._ed(stages[-1]).outlet_concentrate, destination=self.fs.sepa1.inlet)
        self.fs.arc6 = Arc(source=self.fs.sepa1.to_disp, destination=self.fs.disp.inlet)
        self.fs.arc7 = Arc(source=self.fs.sepa1.to_conc_in1, destination=self.fs.mix0.from_conc_out)

        TransformationFactory("network.expand_arcs").apply_to(self.fs)

    def _initialize_units(self, solver=None, optarg=None):
        stages = self._stage_indices()
        iscale.calculate_scaling_factors(self.fs)

        self.fs.feed.initialize(solver=solver, optarg=optarg)
        propagate_state(self.fs.arc0)
        self.fs.sepa0.initialize(solver=solver, optarg=optarg)

        propagate_state(self.fs.arc1b)
        self.fs.pump1.deltaP[0].fix(20e5)
        self.fs.pump1.initialize(solver=solver, optarg=optarg)
        self.fs.pump1.deltaP[0].unfix()

        propagate_state(destination=self.fs.pump0.inlet, source=self.fs.pump1.inlet)
        self.fs.pump0.deltaP[0].fix(20e5)
        self.fs.pump0.initialize(solver=solver, optarg=optarg)
        self.fs.pump0.deltaP[0].unfix()

        propagate_state(self.fs.arc1f)
        propagate_state(self.fs.arc3f)

        try:
            self._ed(stages[0]).initialize(solver=solver, optarg=optarg)
        except Exception as e:
            print(f"Error occurred while initializing ED Stage 1: {e}")

        for stage_num in stages[:-1]:
            next_stage = stage_num + 1
            propagate_state(getattr(self.fs, f"arc_intered{stage_num}_{next_stage}_dil"))
            propagate_state(getattr(self.fs, f"arc_intered{stage_num}_{next_stage}_conc"))
            try:
                self._ed(next_stage).initialize(solver=solver, optarg=optarg)
            except Exception as e:
                print(f"Error occurred while initializing ED Stage {next_stage}: {e}")

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


KStageConcentrateRecirculation.from_yaml = KStageConcentrateRecirculationData.from_yaml
