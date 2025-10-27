from __future__ import annotations
import pandas as pd
import idaes.core.util.model_statistics as mstat
from idaes.core.solvers.get_solver import get_solver
import pyomo as pyo
import pyomo.environ as pyo
import pandas as pd
from typing import List, Dict, Optional, Union
import idaes.core.util.scaling as iscale
from idaes.core.util.misc import add_object_reference
import re
import math
from pyomo.dae import DerivativeVar
import os
import ast
import idaes.logger as log
from electrodialysis_experiment.schema.experiment.data import FluidCondition, UpdateParam
from electrodialysis_experiment.surrogates.transport_number_membrane.cation_cem_simulator import (
    CationCemTransportNumberSimulator,
    SurrogateType,
)
from typing import List, Dict, Tuple, Union, TypeVar
from electrodialysis_experiment.processes.one_stage_single_pass import (
    OneStageSinglePass,
)
from electrodialysis_experiment.schema.config.process_config_schema import (
    OneStageSinglePassConfig,
)
from pydantic import BaseModel
from pathlib import Path
import yaml

_author_ = "Xiangyu Bi"
_log = log.getLogger(__name__)

ProcConfig = TypeVar("ProcConfig", bound=BaseModel)


class MasterExperimentBuilder:

    @classmethod
    def proc_config_from_yaml(
        cls, path: str | Path, sample_size: int = 1, name: str = "UnnamedExperiment"
    ):
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        process_config = OneStageSinglePassConfig(**config_data)
        return cls(process_config=process_config, sample_size=sample_size, name=name)

    def __init__(
        self,
        process_config: ProcConfig,
        sample_size: int = 1,
        name: str = "UnnamedExperiment",
    ):
        # self.process_config = process_config
        self.sample_size = sample_size
        self.model = pyo.ConcreteModel(name=name)
        self._build_model(process_config=process_config)

    def _build_model(self, process_config: ProcConfig = None):
        def _prepare_sample_blk(b, i):
            b.proc = OneStageSinglePass(process_cfg=process_config)

        self.model.sample_set = pyo.Set(initialize=range(self.sample_size))
        self.model.sample_blk = pyo.Block(
            self.model.sample_set, rule=_prepare_sample_blk
        )

    def initialize_individual_sample_blks(
        self,
        scaling_cfg_path: str | Path = None,
        process_init_cfg_path: str | Path = None,
        fluid_condition: List[FluidCondition] = None,
        exp_setup_param: List[UpdateParam]= None,
        t_est: List[Dict] = None,
        solver=None,
        tee: bool = True,
    ):
        for i, b in self.model.sample_blk.items():
            if scaling_cfg_path:
                b.proc.import_scaling_config(scaling_cfg_path)
            if process_init_cfg_path:
                b.proc.import_init_value_config(process_init_cfg_path)
            if exp_setup_param:
                b.proc.update_var_values(exp_setup_param[i])
            if t_est:
                b.proc.update_cation_cem_transport_number(t_est[i])
            b.proc.initialize_process(
                fluid_condition=fluid_condition[i], solver=solver, tee=tee
            )
            _log.info(f"Block {i} initialized.")

    # def _initialize_model_edsp_blocks(
    #     self,
    #     fluid_cond_dt: List[Dict],
    #     param_dt: List[Dict],
    #     cation_cem_transport_number: List[Dict],
    #     base_param_yaml: str = "",
    #     max_iter: int = None,
    #     linear_solver: str = "ma27",
    # ):
    #     for i, b in self.model.sample_blk.items():
    #         edsp.set_ion_memb_properties(b, **self.ion)
    #         edsp.apply_param_values(
    #             m=b, yaml_file=base_param_yaml, yaml_data=None, prefix=b.name
    #         )
    #         edsp.update_var_values_pydantic(b, param_dt[i])
    #         edsp.update_cation_cem_transport_number(b, cation_cem_transport_number[i])
    #         edsp.initialize_dof0_system(
    #             m=b,
    #             initargs=fluid_cond_dt[i],
    #             solve_after_init=True,
    #             linear_solver=linear_solver,
    #             max_iter=max_iter,
    #             tee=True,
    #         )
    #         _log.info(f"Block {i} initialized.")

    # def condition_individual_experiments(
    #     self,
    #     fluid_cond: List[Dict],
    #     param_dt_upd: List[Dict],
    #     cation_cem_transport_number: List[Dict],
    #     max_iter: int = None,
    #     linear_solver: str = "ma27",
    #     param_yaml="edsp_param.yaml",
    # ):

    #     self._initialize_model_edsp_blocks(
    #         fluid_cond_dt=fluid_cond,
    #         param_dt=param_dt_upd,
    #         cation_cem_transport_number=cation_cem_transport_number,
    #         base_param_yaml=param_yaml,
    #         max_iter=max_iter,
    #         linear_solver=linear_solver,
    #     )

    def solve_individual_blocks(
        self, solver=None, tee=True
    ):  # tee=True, linear_solver="ma27", max_iter: int = None

        for i, b in self.model.sample_blk.items():
            print(f"DOF= {mstat.degrees_of_freedom(b)}.")
            # opt={ "linear_solver": linear_solver, "max_iter": max_iter}
            result = OneStageSinglePass.solve(b.proc, solver=solver, tee=tee)
            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                _log.info(f"Block {i} solved successfully.")
            else:
                _log.warning(
                    f"Block {i} failed to yield an optimal solution, with termination condition {result.solver.termination_condition}."
                )

    def initialize_specific_sample_block(
        self,
        index: int,
        scaling_cfg_path: str | Path = None,
        process_init_cfg_path: str | Path = None,
        fluid_condition: FluidCondition = None,
        solver=None,
        tee: bool = True,
    ):
        if scaling_cfg_path:
            self.model.sample_blk[index].proc.import_scaling_config(scaling_cfg_path)
        if process_init_cfg_path:
            self.model.sample_blk[index].proc.import_init_value_config(
                process_init_cfg_path
            )
        self.model.sample_blk[index].proc.initialize_process(
            fluid_condition=fluid_condition, solver=solver, tee=tee
        )

    def add_cation_cem_transport_number_simulator(
        self, surrogate_method: SurrogateType, **surrogate_params
    ):
        """
        Attach a CationCemTransportNumberSimulator to the experimental model.
        """
        idx = self.model.sample_set
        self.model.cation_cem_transport_number_simulator = (
            CationCemTransportNumberSimulator(
                idx, surrogate=surrogate_method, **surrogate_params
            )
        )
        _log.info(
            "The experiment model now has a block named cation_cem_transport_number_simulator."
        )

    def _safe_float(self, val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    def _resolve_path_from(self, path, model):
        obj = model
        parts = re.split(r"(?<!\()\.(?![^\[]*\])", path)
        for part in parts:
            match = re.match(r"(\w+)(\[(.*)\])?", part)
            if not match:
                raise ValueError(f"Unrecognized path element: {part}")
            name, _, idx_str = match.groups()
            obj = getattr(obj, name)
            if idx_str:
                idx = eval(idx_str)
                obj = obj[idx]
        return obj

    def _collect_vars(self):
        for var in self.model.component_data_objects(
            (pyo.Var, DerivativeVar), descend_into=True
        ):
            comp = var.parent_component()
            comp_path = comp.getname(fully_qualified=True)
            index = var.index()
            index_repr = repr(index) if index is not None else "None"
            scaling = iscale.get_scaling_factor(var)
            value = self._safe_float(var.value)
            sf = scaling if scaling is not None else 1
            scaled_value = value * sf if not math.isnan(value) else float("nan")
            yield {
                "component": comp_path,
                "index": index_repr,
                "value": value,
                "fixed": var.fixed,
                "scaling": self._safe_float(scaling) if scaling is not None else 1,
                "scaled_value": scaled_value,
            }

    def _collect_params(self):
        for param in self.model.component_data_objects(pyo.Param, descend_into=True):
            if not hasattr(param, "parent_component"):
                continue
            comp = param.parent_component()
            comp_path = comp.getname(fully_qualified=True)
            index = param.index()
            index_repr = repr(index) if index is not None else "None"
            yield {
                "component": comp_path,
                "index": index_repr,
                "value": self._safe_float(param.value),
            }

    def _collect_constraints(self):
        for con in self.model.component_data_objects(pyo.Constraint, descend_into=True):
            comp = con.parent_component()
            comp_path = comp.getname(fully_qualified=True)
            index = con.index()
            index_repr = repr(index) if index is not None else "None"
            scaling = iscale.get_constraint_transform_applied_scaling_factor(con) or 1

            # Safely get numeric value of the constraint body
            try:
                body_val = self._safe_float(pyo.value(con.body))
            except Exception:
                body_val = float("nan")

            # Compute unscaled constraint residual
            try:
                if con.equality:
                    rhs_val = self._safe_float(pyo.value(con.lower))
                    residue = body_val - rhs_val

                elif con.has_lb() and con.has_ub():
                    lb_val = self._safe_float(pyo.value(con.lower))
                    ub_val = self._safe_float(pyo.value(con.upper))
                    lower_violation = max(0, lb_val - body_val)
                    upper_violation = max(0, body_val - ub_val)
                    residue = max(lower_violation, upper_violation)

                elif con.has_lb():
                    lb_val = self._safe_float(pyo.value(con.lower))
                    residue = max(0, lb_val - body_val)

                elif con.has_ub():
                    ub_val = self._safe_float(pyo.value(con.upper))
                    residue = max(0, body_val - ub_val)

                else:
                    residue = float("nan")

            except Exception:
                residue = float("nan")

            # Compute scaled constraint residual (what IPOPT sees)
            try:
                scaled_residue = (
                    self._safe_float(scaling) * residue
                    if scaling is not None and residue is not None
                    else float("nan")
                )
            except Exception:
                scaled_residue = float("nan")

            yield {
                "component": comp_path,
                "index": index_repr,
                "scaling": self._safe_float(scaling) if scaling is not None else None,
                "body": body_val,
                "residue": residue,
                "scaled_residue": scaled_residue,
            }

    def _collect_expressions(self):
        for expr in self.model.component_data_objects(
            pyo.Expression, descend_into=True
        ):
            comp = expr.parent_component()
            comp_path = comp.getname(fully_qualified=True)
            index = expr.index()
            index_repr = repr(index) if index is not None else "None"
            scaling = iscale.get_scaling_factor(expr)
            yield {
                "component": comp_path,
                "index": index_repr,
                "scaling": self._safe_float(scaling) if scaling is not None else None,
            }

    def save_model_hdf(self, filename):
        var_df = pd.DataFrame(self._collect_vars())
        param_df = pd.DataFrame(self._collect_params())
        con_df = pd.DataFrame(self._collect_constraints())
        expr_df = pd.DataFrame(self._collect_expressions())
        with pd.HDFStore(filename, mode="w") as store:
            store.put("variables", var_df)
            store.put("parameters", param_df)
            store.put("constraints", con_df)
            store.put("expressions", expr_df)

    def _parse_index(self, index):
        """
        Parse an index from a string or primitive type.
        Handles cases like 'None', integers, floats, tuples, and lists.
        """
        if index == "None" or index is None:
            return None
        if isinstance(index, (int, float, tuple)):
            # print(index)
            return index
        try:
            return ast.literal_eval(index)
        except Exception:
            return index  # fallback (e.g., already a primitive)

    def load_model_data(self, filename, target_model=None):
        model = target_model or self.model
        with pd.HDFStore(filename, mode="r") as store:
            var_df = store["variables"] if "variables" in store else pd.DataFrame()
            param_df = store["parameters"] if "parameters" in store else pd.DataFrame()
            con_df = store["constraints"] if "constraints" in store else pd.DataFrame()
            expr_df = store["expressions"] if "expressions" in store else pd.DataFrame()

            # con_df = store.get("constraints")
            # expr_df = store.get("expressions")

        for _, row in param_df.iterrows():
            path, index, value = row["component"], row["index"], row["value"]
            try:
                comp = self._resolve_path_from(path, model)
                idx = eval(index) if index != "None" else None
                if not math.isnan(value):
                    (comp[idx] if idx else comp).set_value(value)
            except Exception as e:
                _log.warning(f"Couldn't restore parameter '{path}[{index}]': {e}")

        for _, row in var_df.iterrows():
            path, index, value, fixed, scaling = (
                row["component"],
                row["index"],
                row["value"],
                row["fixed"],
                row.get("scaling", None),
            )
            try:
                comp = self._resolve_path_from(path, model)
                idx = self._parse_index(index)
                target = comp[idx] if idx is not None else comp
                if not math.isnan(value):
                    target.set_value(value)
                target.fixed = bool(fixed)
                if scaling is not None and not math.isnan(scaling):
                    iscale.set_scaling_factor(target, scaling)
            except Exception as e:
                _log.warning(f"Couldn't restore variable '{path}[{index}]': {e}")

        for _, row in con_df.iterrows():
            path, index, scaling = (
                row["component"],
                row["index"],
                row.get("scaling", None),
            )
            try:
                if scaling is None or math.isnan(scaling):
                    continue
                comp = self._resolve_path_from(path, model)
                idx = eval(index) if index != "None" else None
                iscale.set_scaling_factor(comp[idx] if idx else comp, scaling)
            except Exception as e:
                _log.warning(
                    f"Couldn't restore constraint scaling '{path}[{index}]': {e}"
                )

        for _, row in expr_df.iterrows():
            path, index, scaling = (
                row["component"],
                row["index"],
                row.get("scaling", None),
            )
            try:
                if scaling is None or math.isnan(scaling):
                    continue
                comp = self._resolve_path_from(path, model)
                idx = eval(index) if index != "None" else None
                iscale.set_scaling_factor(comp[idx] if idx else comp, scaling)
            except Exception as e:
                _log.warning(
                    f"Couldn't restore expression scaling '{path}[{index}]': {e}"
                )

    def save_model_csv(self, directory: str):
        """Export variables, parameters, constraints, and expressions to CSV files."""
        os.makedirs(directory, exist_ok=True)

        var_df = pd.DataFrame(self._collect_vars())
        param_df = pd.DataFrame(self._collect_params())
        con_df = pd.DataFrame(self._collect_constraints())
        expr_df = pd.DataFrame(self._collect_expressions())

        var_df.to_csv(os.path.join(directory, "variables.csv"), index=False)
        param_df.to_csv(os.path.join(directory, "parameters.csv"), index=False)
        con_df.to_csv(os.path.join(directory, "constraints.csv"), index=False)
        expr_df.to_csv(os.path.join(directory, "expressions.csv"), index=False)

        _log.info(f"Saved model data as csv files to: {directory}")

    def add_sse_objective_of_selected_variables(
        self,
        variables_weights: Union[List[str], Dict[str, float]] = None,
        name: str = "sse_objective",
        data: pd.DataFrame = None,
    ):
        """
        Add a sum of squared errors (SSE) objective for selected variables.

        Args:
            variables_with_weights (List[str] or Dict[str, float]): Variable names or (variable → weight) mapping.
            name (str): Name of the objective.
            data (pd.DataFrame): Data containing target values.
        """
        if variables_weights is None:
            _log.warning("Skipping objective creation: no variables provided.")
            return
        if data is not None:
            # Ensure variables are in the DataFrame
            missing_vars = [var for var in variables_weights if var not in data.columns]
            if missing_vars:
                raise ValueError(
                    f"The following variables are missing in the data columns: {missing_vars}"
                )
            # variables = list(weights.keys())
        else:
            raise ValueError("Data must be provided to calculate SSE.")
        assert len(data) == len(
            self.model.sample_set
        ), f"Number of data points ({len(data)}) does not match number of samples ({len(self.model.sample_set)})."

        if isinstance(variables_weights, list):
            weights = {var: 1.0 for var in variables_weights}
        else:
            weights = variables_weights
        if hasattr(self.model, "sse_objective"):
            del self.model.sse_objective
        # Create the SSE objective
        expr = sum(
            weights[var]
            * (
                self._resolve_path_from(var, self.model.sample_blk[i].proc)
                - data[var].iloc[i]
            )
            ** 2
            for i in range(len(data))
            for var in weights
        )

        self.model.sse_objective = pyo.Objective(
            expr=expr, sense=pyo.minimize, doc=name
        )

    def free_cation_transport_numbers_in_cem(self):
        """
        Unfixes cation transport numbers in the CEM for all samples and adds a constraint so their sum equals 1 at each membrane position.
        """

        cation_set = self.model.sample_blk[0].proc.fs.EDstack.cation_set
        for i in self.model.sample_set:
            for ion in cation_set:
                if (
                    self.model.sample_blk[i]
                    .proc.fs.EDstack.ion_trans_number_membrane["cem", ion, :]
                    .is_fixed()
                ):
                    self.model.sample_blk[i].proc.fs.EDstack.ion_trans_number_membrane[
                        "cem", ion, :
                    ].unfix()

    def add_cation_transport_number_sum_constraint(self):
        # Add constraint: sum of ion transport numbers for cations at each x in length_domain equals 1
        cation_set = self.model.sample_blk[0].proc.fs.EDstack.cation_set
        self.model.cem_trans_number_sum_con = pyo.ConstraintList()
        for x in self.model.length_domain:
            expr = sum(
                self.model.sample_blk[0].proc.fs.EDstack.ion_trans_number_membrane[
                    "cem", ion, x
                ]
                for ion in cation_set
            )
            self.model.cem_trans_number_sum_con.add(expr == 1)

    def add_equal_ocv_constraint(self):
        m = self.model
        if not hasattr(m, "sample_blk"):
            raise ValueError("Model is missing sample_blk attribute.")

        # Add constraint to enforce OCV values are equal across all samples
        first_sample_idx = next(iter(m.sample_blk))
        first_sample = m.sample_blk[first_sample_idx]

        # Create constraint list
        m.ocv_equality_cons = pyo.ConstraintList()

        # Add equality constraints for each sample
        for sample_idx in m.sample_blk:
            if sample_idx != first_sample_idx:
                m.ocv_equality_cons.add(
                    first_sample.proc.fs.ocv == m.sample_blk[sample_idx].proc.fs.ocv
                )

    def add_log_linear_surr_coef_constraint(self):
        m = self.model
        if not hasattr(m, "cation_cem_transport_number_simulator"):
            raise ValueError(
                "Model is missing cation_cem_transport_number_simulator attribute."
            )

        # Add constraint to enforce conc_ratio_coef values are equal across all simulators
        if hasattr(m.cation_cem_transport_number_simulator, "_index_set"):
            # Get the first simulator to use as reference
            first_sim_idx = next(iter(m.cation_cem_transport_number_simulator))
            first_sim = m.cation_cem_transport_number_simulator[first_sim_idx]

            # Get all coefficient indices
            coef_indices = first_sim.conc_ratio_coef.index_set()

            # Create constraint list
            m.conc_ratio_coef_equality_cons = pyo.ConstraintList()

            # Add equality constraints for each coefficient
            for coef_idx in coef_indices:
                for sim_idx in m.cation_cem_transport_number_simulator:
                    if sim_idx != first_sim_idx:
                        m.conc_ratio_coef_equality_cons.add(
                            m.cation_cem_transport_number_simulator[
                                first_sim_idx
                            ].conc_ratio_coef[coef_idx]
                            == m.cation_cem_transport_number_simulator[
                                sim_idx
                            ].conc_ratio_coef[coef_idx]
                        )
        else:
            _log.warning(
                "Warning: cation_cem_transport_number_simulator is not indexed. Cannot add coefficient equality constraints."
            )

    def _add_t_cation_smoothness_penalty(self):
        """
        Adds a smoothness penalty expression for cation transport numbers (Na+, Ca2+, Mg2+) across the membrane length domain.

        The penalty is the sum of squared differences between adjacent transport numbers, promoting smoothness.
        It is added as a Pyomo Expression to each sample block and a total penalty is summed across all blocks.
        """

        def t_smoothness_penalty_rule(b):

            indices = sorted(list(b.diluate.length_domain))
            # Exclude the last index to avoid out-of-range for i+1
            return sum(
                (
                    b.ion_trans_number_membrane["cem", "Na_+", indices[i + 1]]
                    - b.ion_trans_number_membrane["cem", "Na_+", indices[i]]
                )
                ** 2
                + 1
                * (
                    b.ion_trans_number_membrane["cem", "Ca_2+", indices[i + 1]]
                    - b.ion_trans_number_membrane["cem", "Ca_2+", indices[i]]
                )
                ** 2
                + 1
                * (
                    b.ion_trans_number_membrane["cem", "Mg_2+", indices[i + 1]]
                    - b.ion_trans_number_membrane["cem", "Mg_2+", indices[i]]
                )
                ** 2
                for i in range(len(indices) - 1)
            )

        for blk in self.model.sample_blk.values():
            blk.proc.fs.EDstack.smoothness_penalty = pyo.Expression(
                rule=t_smoothness_penalty_rule
            )

        self.model.total_trans_num_smoothness_penalty = pyo.Expression(
            expr=sum(
                blk.proc.fs.EDstack.smoothness_penalty
                for blk in self.model.sample_blk.values()
            )
        )

    def add_penalty_to_sse_objective(self, alpha=0.01):
        if not isinstance(getattr(self.model, "sse_objective", None), pyo.Objective):
            raise TypeError(
                "'sse_objective' must exist and be a Pyomo Objective. Use the "
                "' add_sse_objective_of_selected_variables' method to build it. "
            )
        else:
            sse_expr = self.model.sse_objective.expr
            del self.model.sse_objective

        if not isinstance(
            getattr(self.model, "total_trans_num_smoothness_penalty", None),
            pyo.Expression,
        ):
            self._add_t_cation_smoothness_penalty()

        # Define the new penalized objective
        self.model.sse_t_smooth_objective = pyo.Objective(
            expr=sse_expr + alpha * self.model.total_trans_num_smoothness_penalty,
            sense=pyo.minimize,
        )

    def _find_conc_mol_ion_var(self, ion):

        conc_mol_var = []
        for var in self.model.component_data_objects(pyo.Var, descend_into=True):
            if "conc_mol_phase_comp" in var.name and var.index() == ("Liq", ion):
                conc_mol_var.append(var)
        if len(conc_mol_var) == 0:
            return None
        return conc_mol_var

    def set_ion_conc_mol_bounds(
        self, ion_list: List[str], bounds: Tuple[float, float] = None
    ):
        """
        Sets the molar concentration bounds for each ion in the system.
        """
        for ion in ion_list:
            var_list = self._find_conc_mol_ion_var(ion)
            if var_list is None:
                print(f"Could not find conc_mol_phase_comp variable for ion '{ion}'")
            else:
                if bounds is not None:
                    for var in var_list:
                        var.setlb(bounds[0])
                        var.setub(bounds[1])
                        # print(var.name)
                    print(
                        f"Set bounds of conc_mol_phase_comp for ion '{ion}' to {bounds} for {len(var_list)} variables."
                    )
                else:
                    print(f"Not setting bounds for ion '{ion}'")
                    continue

    def scale_variables_by_current_value(self, default_scale=1.0):
        for var in self.model.component_data_objects(pyo.Var, active=True):
            if var.fixed:
                continue  # skip fixed variables

            try:
                val = abs(pyo.value(var))
            except:
                val = None

            # If current value is 0 or None, fallback to bounds
            if not val or val < 1e-8:
                lower = var.lb if var.lb is not None else 0.0
                upper = var.ub if var.ub is not None else 0.0
                span = abs(upper - lower)
                val = abs(lower) if abs(lower) > 0 else abs(upper)
                val = max(val, span)

            # Determine scale: inverse of magnitude
            if val and val > 1e-8:
                scale = 1.0 / val
            else:
                scale = default_scale

            # Optional: clip scale to avoid extreme values
            scale = min(max(scale, 1e-6), 1e6)
            iscale.set_scaling_factor(var, scale)
