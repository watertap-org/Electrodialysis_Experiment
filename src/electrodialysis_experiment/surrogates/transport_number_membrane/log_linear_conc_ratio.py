# surrogates/log_linear.py
import pandas as pd
import pyomo.environ as pyo
from idaes.core import ProcessBlockData
from electrodialysis_experiment.surrogates.transport_number_membrane.registry import (
    register_transport_number_surrogate,
    register_transport_number_surrogate_init,
    register_transport_number_surrogate_fn_identity,
)
import numpy as np
from scipy.optimize import minimize
from IPython.display import display
import inspect
from typing import Dict, List, Tuple, Callable
import plotly.graph_objects as go


@register_transport_number_surrogate("log_linear_polynomial")
def build_log_linear_polynomial(b: pyo.Block, eps: float = 1e-12) -> None:
    """
    Adds variables and constraints to a Pyomo Block (or its members if indexed) for a log-linear surrogate of transport number ratios.

    Parameters
    ----------
    b : pyomo.Block
        Block to modify. Must provide cation_set, _reference_ion, and EDstack via model().sample_blk[ind].proc.fs.EDstack.
    eps : float, optional
        Small epsilon to keep logs well-defined. Default is 1e-12.
    """

    def _build_for_block(bd: ProcessBlockData, ind):
        m = bd.model()
        sed = m.sample_blk[ind].proc.fs.EDstack  # member-specific EDstack
        X = sed.diluate.length_domain

        if bd._reference_ion not in bd.cation_set:
            raise ValueError(
                f"Reference ion '{bd._reference_ion}' not in cation set {bd.cation_set}."
            )

        J = [cation for cation in bd.cation_set if cation != bd._reference_ion]

        # Small epsilon to keep logs well-defined
        bd.eps = pyo.Param(initialize=eps, mutable=True)
        bd.conc_ratio_coef = pyo.Var(J, bounds=(0, 100), initialize=1.0)
        # A_coef(x) := sum_j coef[j,x] * ((c_j + eps) / (c_ref + eps))
        bd.A = pyo.Expression(
            X,
            rule=lambda b, x: sum(
                b.conc_ratio_coef[j]
                * (
                    (sed.diluate.properties[0, x].conc_mol_phase_comp["Liq", j] + b.eps)
                    / (
                        sed.diluate.properties[0, x].conc_mol_phase_comp[
                            "Liq", b._reference_ion
                        ]
                        + b.eps
                    )
                )
                for j in J
            ),
        )
        bd.K = pyo.RangeSet(1, bd._poly_degree)
        bd.taylor_coef = pyo.Param(
            bd.K, initialize=lambda b, k: (-1) ** (k + 1) / k, mutable=False
        )
        bd.log1p_taylor_app = pyo.Expression(
            X, rule=lambda b, x: sum(b.taylor_coef[k] * (b.A[x] ** k) for k in b.K)
        )

        def ref_rule(_b, x):
            return sed.ion_trans_number_membrane[
                "cem", _b._reference_ion, x
            ] == 1.0 - sum(sed.ion_trans_number_membrane["cem", j, x] for j in J)

        bd.t_ref_ion_cem = pyo.Constraint(X, rule=ref_rule)

        def nonref_rule(_b, i, x):
            if i == _b._reference_ion:
                return pyo.Constraint.Skip
            c_i = sed.diluate.properties[0, x].conc_mol_phase_comp["Liq", i]
            c_ref = sed.diluate.properties[0, x].conc_mol_phase_comp[
                "Liq", _b._reference_ion
            ]
            return (
                pyo.log(sed.ion_trans_number_membrane["cem", i, x] + bd.eps)
                == pyo.log(bd.conc_ratio_coef[i] + bd.eps)
                + pyo.log((c_i + bd.eps) / (c_ref + bd.eps))
                - bd.log1p_taylor_app[x]
            )

        bd.t_nonref_ion_cem = pyo.Constraint(J, X, rule=nonref_rule)

    if b.is_indexed():
        for ind, bd in b.items():
            _build_for_block(bd, ind)
    else:
        _build_for_block(b, b.index())


@register_transport_number_surrogate("log_linear_log")
def build_log_linear_log(b: pyo.Block, eps: float = 1e-12) -> None:
    """
    Adds variables and constraints to a Pyomo Block (or its members if indexed) for a log-linear-log surrogate of transport number ratios.

    Parameters
    ----------
    b : pyomo.Block
        Block to modify. Must provide cation_set, _reference_ion, and EDstack via model().sample_blk[ind].proc.fs.EDstack.
    eps : float, optional
        Small epsilon to keep logs well-defined. Default is 1e-12.
    """

    def _build_for_block(bd: ProcessBlockData, ind):
        m = bd.model()
        sed = m.sample_blk[ind].proc.fs.EDstack  # member-specific EDstack
        X = sed.diluate.length_domain

        if bd._reference_ion not in bd.cation_set:
            raise ValueError(
                f"Reference ion '{bd._reference_ion}' not in cation set {bd.cation_set}."
            )

        J = [cation for cation in bd.cation_set if cation != bd._reference_ion]

        # Small epsilon to keep logs well-defined
        bd.eps = pyo.Param(initialize=eps, mutable=True)
        bd.conc_ratio_coef = pyo.Var(J, bounds=(0, None), initialize=1.0)
        # A_coef(x) := sum_j coef[j,x] * ((c_j + eps) / (c_ref + eps))
        bd.A = pyo.Expression(
            X,
            rule=lambda b, x: sum(
                b.conc_ratio_coef[j]
                * (
                    (sed.diluate.properties[0, x].conc_mol_phase_comp["Liq", j] + b.eps)
                    / (
                        sed.diluate.properties[0, x].conc_mol_phase_comp[
                            "Liq", b._reference_ion
                        ]
                        + b.eps
                    )
                )
                for j in J
            ),
        )

        def ref_rule(_b, x):
            return sed.ion_trans_number_membrane[
                "cem", _b._reference_ion, x
            ] == 1.0 - sum(sed.ion_trans_number_membrane["cem", j, x] for j in J)

        bd.t_ref_ion_cem = pyo.Constraint(X, rule=ref_rule)

        def nonref_rule(_b, i, x):
            if i == _b._reference_ion:
                return pyo.Constraint.Skip
            c_i = sed.diluate.properties[0, x].conc_mol_phase_comp["Liq", i]
            c_ref = sed.diluate.properties[0, x].conc_mol_phase_comp[
                "Liq", _b._reference_ion
            ]
            return pyo.log(
                sed.ion_trans_number_membrane["cem", i, x] + bd.eps
            ) == pyo.log(bd.conc_ratio_coef[i] + bd.eps) + pyo.log(
                (c_i + bd.eps) / (c_ref + bd.eps)
            ) - pyo.log(
                1 + bd.A[x]
            )

        bd.t_nonref_ion_cem = pyo.Constraint(J, X, rule=nonref_rule)

    if b.is_indexed():
        for ind, bd in b.items():
            _build_for_block(bd, ind)
    else:
        _build_for_block(b, b.index())


@register_transport_number_surrogate_init("log_linear_polynomial")
def init_log_linear_polynomial(
    conc_data: List[Dict[str, float]],
    trans_number_data: List[Dict[str, float]],
    fitting_coef_guess: Dict[str, float],
    reference_ion: str,
    coef_bounds: Dict[str, Tuple[float, float]] = None,
    log_objective: bool = False,
    polynomial_degree: int = 1,
    eps: float = 1e-12,
    plot_results: bool = True,
):
    ions = list(conc_data[0].keys())
    J = [j for j in ions if j != reference_ion]
    n_samples = len(conc_data)
    c_data = {j: np.array([d[j] for d in conc_data], dtype=float) for j in ions}
    c_ref = c_data[reference_ion]
    t_data = {j: np.array([d[j] for d in trans_number_data], dtype=float) for j in ions}

    # Prepare normalized data
    c_ratio = {j: (c_data[j] + eps) / (c_ref + eps) for j in J}
    log_t = {j: np.log(t_data[j] + eps) for j in ions}
    log_c_ratio = {
        j: np.log(c_ratio[j]) for j in J
    }  # {Ca_2+: array[...], Mg_2+: array[...], }

    # Design matrix A: each column is c_k/c_ref for ion k
    A = np.column_stack([c_ratio[k] for k in J])
    # show(A)  # TOBEDELETED
    # def _taylor_log1p(x, degree=1):
    #     """
    #     Taylor approximation to log(1+x) at x=0 up to given degree (degree >= 1).
    #     Works elementwise on numpy arrays x.
    #     """
    #     k = np.arange(1, degree + 1, dtype=float)  # [1,2,...,n]
    #     # shape tricks: raise x to powers 1..n, then sum coefficients along last axis
    #     terms = (x[..., None] ** k) * ((-1) ** (k + 1) / k)
    #     return np.sum(terms, axis=-1)

    def _predict_logt(
        coef_dict: Dict[str, float],
        ion: str,
        degree: int = 1,
        taylor_series: bool = True,
    ):
        """
        Predict log(transport number) for a given ion.
        Parameters
        ----------
        coef_dict : Dict[str, float]
            Dictionary mapping ion names to their corresponding model coefficients.
        ion : str
            The ion for which to predict the log(transport number).
        degree : int, optional
            Degree of the Taylor series expansion to use for approximation (default is 1).
        taylor_series : bool, optional
            If True, approximates -log(1 + A_coef) using a Taylor series expansion of the specified degree.
            If False, computes the exact value using numpy's log1p for numerical stability.

        Returns
        -------
        float
            The predicted value of log(transport number) for the specified ion.
        """
        coef_arr = np.array(
            [coef_dict[j] for j in J]
        )  # ensure consistent order with A's columns
        A_coef = A @ coef_arr
        if ion not in J:
            return np.log(
                1.0
                - sum(
                    np.exp(_predict_logt(coef_dict, j, degree, taylor_series))
                    for j in J
                )
            )

        base = np.log(coef_dict[ion] + eps) + log_c_ratio[ion]
        if taylor_series:
            approx = -_taylor_log1p(A_coef, degree)  # ~ -log(1 + A_coef)
        else:
            approx = -np.log1p(A_coef)  # exact, stable

        return base + approx

    def _get_ti_prediction(
        ions: List[str],
        coef_dict: Dict[str, float],
        degree: int = 1,
        taylor_series: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get predicted transport numbers for all ions.

        Parameters
        ----------
        ions : List[str]
            List of ion names to predict transport numbers for.
        coef_dict : Dict[str, float]
            Dictionary mapping ion names to their corresponding model coefficients.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of predicted transport numbers for each ion.
        """
        ti_pred = {}
        logti_pred = {}
        for ion in ions:
            ti_pred[ion] = np.exp(
                _predict_logt(
                    coef_dict, ion, degree=degree, taylor_series=taylor_series
                )
            )
            logti_pred[ion] = _predict_logt(
                coef_dict, ion, degree=degree, taylor_series=taylor_series
            )
        return ti_pred, logti_pred

    def _sse_objective(
        coef_array: np.ndarray, log_objective: bool = False, l2_coef: float = 0.0
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Objective function to minimize: sum of squared errors between predicted and actual transport numbers.

        Parameters
        ----------
        coef_dict : Dict[str, float]
            Dictionary mapping ion names to their corresponding model coefficients.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], float]
            A tuple containing:
            - A dictionary of predicted log(transport numbers) for each ion.
            - The total sum of squared errors (SSE).
        """
        total_sse = 0.0
        log_pred = {}
        coef_dict = {j: coef_array[idx] for idx, j in enumerate(J)}
        for ion in ions:
            log_pred[ion] = _predict_logt(
                coef_dict, ion, degree=polynomial_degree, taylor_series=True
            )
            if log_objective:
                residuals = log_t[ion] - log_pred[ion]
            else:
                residuals = t_data[ion] - np.exp(log_pred[ion])
            total_sse += np.sum(residuals**2)
            if l2_coef > 0.0:
                total_sse += l2_coef * sum((coef_dict[j] ** 2) for j in J)
        return total_sse

    initial_guess = np.array([fitting_coef_guess[j] for j in J])
    if coef_bounds:
        bounds = [coef_bounds[j] for j in J]
    else:
        bounds = [(0, None)] * len(J)

    def _sse_objective_wrapper(coef_array: np.ndarray) -> float:
        return _sse_objective(coef_array, log_objective=log_objective)

    fitting_outcome = regression_scipy_minimizer(
        _sse_objective_wrapper, initial_coef=initial_guess, bounds=bounds
    )
    fitted_coef = fitting_outcome["fitted_val"]
    fitted_coef_dict = {j: fitted_coef[idx] for idx, j in enumerate(J)}
    # Get predictions with the final coefficients
    ti_pred, logti_pred = _get_ti_prediction(
        ions, fitted_coef_dict, degree=polynomial_degree, taylor_series=True
    )
    # Calculate and print R-squared for all ions combined
    print("\n--- Fitting Quality ---")
    all_y_actual = np.concatenate([t_data[ion] for ion in ions])
    all_y_pred = np.concatenate([ti_pred[ion] for ion in ions])

    # Handle cases where variance is zero
    if np.var(all_y_actual) == 0:
        combined_r2 = 1.0 if np.allclose(all_y_actual, all_y_pred) else 0.0
    else:
        ss_res = np.sum((all_y_actual - all_y_pred) ** 2)
        ss_tot = np.sum((all_y_actual - np.mean(all_y_actual)) ** 2)
        combined_r2 = 1 - (ss_res / ss_tot)

    print(f"Combined R-squared for all ions: {combined_r2:.4f}")
    # Calculate R-squared for each ion
    for ion in ions:
        y_actual = t_data[ion]
        y_pred = ti_pred[ion]

        if np.var(y_actual) == 0:
            r2 = 1.0 if np.allclose(y_actual, y_pred) else 0.0
        else:
            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

        print(f"R-squared for {ion}: {r2:.4f}")

    if plot_results:
        # Plot predicted vs. actual transport numbers
        fig = go.Figure()

        for ion in ions:
            fig.add_trace(
                go.Scatter(
                    x=t_data[ion],
                    y=ti_pred[ion],
                    mode="markers",
                    name=f"{ion} (Predicted)",
                    hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
                )
            )

        # Add a y=x line for reference
        all_values = np.concatenate(list(t_data.values()))
        min_val, max_val = np.min(all_values), np.max(all_values)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Ideal (y=x)",
            )
        )

        fig.update_layout(
            title="Predicted vs. Actual Transport Numbers",
            xaxis_title="Actual Transport Number",
            yaxis_title="Predicted Transport Number",
            legend_title="Ions",
            template="plotly_white",
        )
        fig.show()

    return fitted_coef_dict


@register_transport_number_surrogate_fn_identity("log_linear_polynomial")
def log_linear_polynomial_fn(
    ions: str,
    reference_ion: str,
    fitted_coef_dict: dict,
    dependent_conc: dict,
    degree: int = 1,
    eps: float = 1e-12,
    **kwargs,
):
    J = [j for j in ions if j != reference_ion]
    fitted_coef_arr = np.array([fitted_coef_dict[j] for j in J])
    c_ratio = {
        j: (dependent_conc[j] + eps) / (dependent_conc[reference_ion] + eps) for j in J
    }
    log_c_ratio = {j: np.log(c_ratio[j]) for j in J}
    A = np.column_stack([c_ratio[k] for k in J])  # shape (n_samples, n_ions-1)
    A_coef = A @ fitted_coef_arr  # shape (n_samples,)
    logt_pred = {}
    t_pred = {}
    for ion in ions:
        logt_pred[ion] = _predict_logt_log_linear_polynomial(
            coef_dict=fitted_coef_dict,
            conc_ratio_dict=c_ratio,
            ion=ion,
            non_ref_ions=J,
            degree=degree,
            taylor_series=True,
        )
        t_pred[ion] = np.exp(logt_pred[ion]) - eps
    return t_pred, logt_pred


def predict_ti_by_surrogate(ions: list, surrogate_fn: Callable, *args, **kwargs):
    """
    Predict transport numbers using the fitted surrogate model.

    Parameters
    ----------
    ions : list
        List of ion names.
    surrogate_fn : Callable
        The surrogate function to use for predictions.
    **args : dict
        Additional arguments for the surrogate function.
    **kwargs : dict
        Additional keyword arguments for the surrogate function.

    """
    return surrogate_fn(ions=ions, *args, **kwargs)


# Shared functions for internal uses
def _taylor_log1p(x, degree=1):
    """
    Taylor approximation to log(1+x) at x=0 up to given degree (degree >= 1).
    Works elementwise on numpy arrays x.
    """
    k = np.arange(1, degree + 1, dtype=float)  # [1,2,...,n]
    # shape tricks: raise x to powers 1..n, then sum coefficients along last axis
    terms = (x[..., None] ** k) * ((-1) ** (k + 1) / k)
    return np.sum(terms, axis=-1)


def _predict_logt_log_linear_polynomial(
    coef_dict: Dict[str, float],
    conc_ratio_dict: Dict[str, np.array],
    ion: str,
    non_ref_ions: List[str],
    degree: int = 1,
    taylor_series: bool = True,
    eps: float = 1e-12,
):
    """
    Predict log(transport number) for a given ion.
    Parameters
    ----------
    coef_dict : Dict[str, float]
        Dictionary mapping ion names to their corresponding model coefficients.
    ion : str
        The ion for which to predict the log(transport number).
    degree : int, optional
        Degree of the Taylor series expansion to use for approximation (default is 1).
    taylor_series : bool, optional
        If True, approximates -log(1 + A_coef) using a Taylor series expansion of the specified degree.
        If False, computes the exact value using numpy's log1p for numerical stability.

    Returns
    -------
    float
        The predicted value of log(transport number) for the specified ion.
    """
    J = non_ref_ions
    assert J == list(coef_dict.keys())
    # print(conc_ratio_dict)
    assert J == list(conc_ratio_dict.keys())

    coef_arr = np.array(
        [coef_dict[j] for j in J]
    )  # ensure consistent order with A's columns
    conc_ratio_array = np.column_stack([conc_ratio_dict[k] for k in J])
    log_c_ratio_dict = {j: np.log(conc_ratio_dict[j]) for j in J}
    # print(conc_ratio_array.shape)
    # print(coef_arr.shape)
    A_coef = conc_ratio_array @ coef_arr
    if ion not in J:
        return np.log(
            1.0
            - sum(
                np.exp(
                    _predict_logt_log_linear_polynomial(
                        coef_dict, conc_ratio_dict, j, J, degree, taylor_series
                    )
                )
                for j in J
            )
        )

    base = np.log(coef_dict[ion] + eps) + log_c_ratio_dict[ion]
    if taylor_series:
        approx = -_taylor_log1p(A_coef, degree)  # ~ -log(1 + A_coef)
    else:
        approx = -np.log1p(A_coef)  # exact, stable

    return base + approx


def _get_ti_prediction(
    ions: List[str],
    coef_dict: Dict[str, float],
    degree: int = 1,
    taylor_series: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Get predicted transport numbers for all ions.

    Parameters
    ----------
    ions : List[str]
        List of ion names to predict transport numbers for.
    coef_dict : Dict[str, float]
        Dictionary mapping ion names to their corresponding model coefficients.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of predicted transport numbers for each ion.
    """
    ti_pred = {}
    logti_pred = {}
    for ion in ions:
        ti_pred[ion] = np.exp(
            _predict_logt(coef_dict, ion, degree=degree, taylor_series=taylor_series)
        )
        logti_pred[ion] = _predict_logt(
            coef_dict, ion, degree=degree, taylor_series=taylor_series
        )
    return ti_pred, logti_pred


def show(obj):
    # get caller’s local variables
    frame = inspect.currentframe().f_back
    names = [name for name, val in frame.f_locals.items() if val is obj]

    name_str = names[0] if names else "<?>"
    print(f"{name_str} = {obj!r} (type: {type(obj).__name__})")


def regression_scipy_minimizer(
    obj: Callable, initial_coef: np.ndarray, bounds: list = None
):
    """
    Minimize the given objective function using SciPy's optimization routines.

    Parameters
    ----------
    obj : Callable
        The objective function to minimize.
    initial_coef : np.ndarray
        Initial guess for the parameters.
    bounds : list
        Bounds for the parameters.

    Returns
    -------
    OptimizeResult
        The result of the optimization.
    """
    strategies = [
        ("L-BFGS-B", {"maxiter": 1e5}),
        ("SLSQP", {"maxiter": 1e5}),
        ("trust-constr", {"maxiter": 1e5}),
        ("COBYLA", {"maxiter": 1e5}),
    ]
    best_result = None
    best_objective = np.inf
    best_method = None

    # Try multiple initial values for robustness
    initial_guesses = [
        initial_coef,
        np.ones(len(initial_coef)),
        np.random.uniform(0.1, 10, len(initial_coef)) * initial_coef,
        np.random.uniform(0.1, 10, len(initial_coef)) * initial_coef,
    ]  # TODO: consider adding more initial value guesses

    for init_guess in initial_guesses:
        for method, options in strategies:
            try:
                result = minimize(
                    obj, init_guess, method=method, bounds=bounds, options=options
                )
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    best_method = method
                elif result.fun == best_objective:
                    best_method = f"{best_method}, {method}"
            except Exception as e:
                print(f"Warning: Optimization with {method} failed: {e}")
                continue

    print(f"Best optimization method: {best_method}")
    print(f"Final optimization result: {best_result}")
    if best_result is None:
        print(f"Warning: Final optimization failed: {best_result.message}")
        return {"fitted_val": initial_coef, "objective": None}
    else:
        print(f"Final optimization succeeded: {best_result.message}")
        # Extract and return relevant information from the best result
        return {"fitted_val": best_result.x, "objective": best_result.fun}
