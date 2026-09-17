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
from pyomo.common.config import ConfigBlock, ConfigValue, ConfigDict, In
from pyomo.core.base.block import BlockData, declare_custom_block
from idaes.core import declare_process_block_class, ProcessBlockData
from electrodialysis_experiment.surrogates.transport_number_membrane.registry import (
    SURROGATE_REGISTRY,
    SURROGATE_INITIALIZE,
)
from idaes.core.util.misc import add_object_reference
from enum import Enum
from . import log_linear_conc_ratio
from . import softmax_covariates
from typing import List, Dict, Tuple


class SurrogateType(Enum):
    LOG_LINEAR_POLYNOMIAL = "log_linear_polynomial"
    LOG_LINEAR_LOG = "log_linear_log"
    SOFTMAX_COVARIATES = "softmax_covariates"
    # Add other surrogate types as needed


@declare_process_block_class(name="CationCemTransportNumberSimulator")
class CationCemTransportNumberSimulatorData(ProcessBlockData):
    """The transport number simulator will be attached to the experiment model directly.
    It will be indexed by the sample_set, with each member associated with the corresponding
    sample_blk. Each sample_blk has the structure of sample_blk.proc.fs.EDstack, etc., i.e., each
    sample_blk carries a ED flowsheet."""

    CONFIG = ConfigBlock(implicit=True)
    CONFIG.declare(
        "surrogate",
        ConfigValue(
            default=None,
            domain=In(SurrogateType),
            description="Key in SURROGATE_REGISTRY",
        ),
    )
    CONFIG.declare(
        "reference_ion",
        ConfigValue(default=None, domain=str, description="Reference Ion"),
    )
    CONFIG.declare(
        "poly_degree",
        ConfigValue(default=5, domain=int, description="Polynomial Degree"),
    )

    @staticmethod
    def _normalize_surrogate_key(surrogate):
        if isinstance(surrogate, SurrogateType):
            surrogate = surrogate.value
        # Backward compatibility for legacy shorthand found in older scripts.
        if surrogate == "s":
            return SurrogateType.LOG_LINEAR_POLYNOMIAL.value
        return surrogate

    def build(self, *args, **kwargs):
        super().build()
        # self is CationCemTransportNumberSimulatorData indexed by sample_set
        self._surrogate = self._normalize_surrogate_key(self.config.surrogate)
        if self.config.reference_ion is not None:
            self._reference_ion = self.config.reference_ion
        if self.config.poly_degree is not None:
            self._poly_degree = self.config.poly_degree
        if self.index() is None:
            raise ValueError(
                "The CationCemTransportNumberSimulator must be indexed by the experiment's sample_set."
            )

        self._setup_context()

        try:
            builder = SURROGATE_REGISTRY[self._surrogate]
        except KeyError as e:
            raise ValueError(
                f"Unknown surrogate '{self._surrogate}'. "
                f"Available: {list(SURROGATE_REGISTRY)}"
            ) from e

        builder(self)

    def _setup_context(self):
        # Check the experimental model has essential ED flowsheets on sample_blk.
        m = self.parent_block()
        for blk in m.sample_blk.values():
            if not hasattr(blk, "proc"):
                raise ValueError('Each sample_blk must have a "proc" attribute.')
            if not hasattr(blk.proc, "fs"):
                raise ValueError('Each sample_blk.proc must have a "fs" attribute.')
            if not hasattr(blk.proc.fs, "properties"):
                raise ValueError(
                    'Each sample_blk.proc.fs must have a "properties" attribute.'
                )
            if not hasattr(blk.proc.fs.properties, "cation_set"):
                raise ValueError(
                    'Each sample_blk.proc.fs.properties must have a "cation_set" attribute.'
                )

        add_object_reference(
            self, "cation_set", m.sample_blk[0].proc.fs.properties.cation_set
        )

    def initiate_surrogate(
        self,
        conc_data: List[Dict[str, float]] | None = None,
        trans_number_data: List[Dict[str, float]] | None = None,
        fitting_coef_guess: Dict[str, float] | None = None,
        reference_ion: str | None = None,
        coef_bounds: Dict[str, Tuple[float, float]] = None,
        log_objective: bool = False,
        polynomial_degree: int = 1,
        eps: float = 1e-12,
        plot_results: bool = True,
        feature_data: List[Dict[str, float]] | None = None,
        **extra_kwargs,
    ):
        # Initialize the surrogate model
        try:
            initiator = SURROGATE_INITIALIZE[self._surrogate]
            # print(initiator)
        except KeyError as e:
            raise ValueError(
                f"Unknown surrogate '{self._surrogate}'. "
                f"Available: {list(SURROGATE_INITIALIZE)}"
            ) from e

        init_kwargs = dict(
            conc_data=conc_data,
            trans_number_data=trans_number_data,
            fitting_coef_guess=fitting_coef_guess,
            reference_ion=reference_ion,
            log_objective=log_objective,
            polynomial_degree=polynomial_degree,
            coef_bounds=coef_bounds,
            eps=eps,
            plot_results=plot_results,
            **extra_kwargs,
        )
        if feature_data is not None:
            init_kwargs["feature_data"] = feature_data

        coef_init = initiator(**init_kwargs)
        if hasattr(self, "conc_ratio_coef"):
            for cation in self.cation_set:
                if cation in coef_init:
                    self.conc_ratio_coef[cation].set_value(coef_init[cation])
        if hasattr(self, "score_intercept"):
            intercept = coef_init.get("intercept", {})
            coef = coef_init.get("coef", {})
            feature_center = coef_init.get("feature_center", {})
            feature_scale = coef_init.get("feature_scale", {})
            for ion in self.score_intercept:
                if ion in intercept:
                    self.score_intercept[ion].set_value(intercept[ion])
            for ion in self.nonref_ion_set:
                if ion not in coef:
                    continue
                for feature in self.feature_set:
                    if feature in coef[ion]:
                        self.score_coef[ion, feature].set_value(coef[ion][feature])
            if hasattr(self, "feature_center"):
                for feature in self.feature_set:
                    if feature in feature_center:
                        self.feature_center[feature].set_value(feature_center[feature])
            if hasattr(self, "feature_scale"):
                for feature in self.feature_set:
                    if feature in feature_scale:
                        self.feature_scale[feature].set_value(feature_scale[feature])
        return coef_init
