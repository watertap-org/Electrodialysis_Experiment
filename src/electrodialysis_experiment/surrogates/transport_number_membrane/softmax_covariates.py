from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyomo.environ as pyo
from idaes.core import ProcessBlockData
from scipy.optimize import minimize

from electrodialysis_experiment.surrogates.transport_number_membrane.registry import (
    register_transport_number_surrogate,
    register_transport_number_surrogate_fn_identity,
    register_transport_number_surrogate_init,
)


SOFTMAX_FEATURE_ORDER = ("log_ca_na", "log_mg_na", "log_current_density")


def _safe_log_ratio(numer, denom, eps: float = 1e-12):
    return np.log((np.asarray(numer, dtype=float) + eps) / (np.asarray(denom, dtype=float) + eps))


def prepare_softmax_feature_data_from_dataframe(
    df: pd.DataFrame,
    *,
    na_col: str = "CpNa",
    ca_col: str = "CpCa",
    mg_col: str = "CpMg",
    current_col: str = "Curr",
    current_density_col: str = "CurrD",
    width_col: str = "Weff",
    length_col: str = "Leff",
    eps: float = 1e-12,
) -> List[Dict[str, float]]:
    rows = []
    for _, row in df.iterrows():
        na = float(row[na_col])
        ca = float(row[ca_col])
        mg = float(row[mg_col])
        if current_density_col in row.index and pd.notna(row[current_density_col]):
            j = float(row[current_density_col])
        else:
            curr = float(row[current_col])
            width = float(row[width_col])
            length = float(row[length_col])
            j = curr / max(width * length, eps)
        rows.append(
            {
                "log_ca_na": float(_safe_log_ratio(ca, na, eps=eps)),
                "log_mg_na": float(_safe_log_ratio(mg, na, eps=eps)),
                "log_current_density": float(np.log(j + eps)),
            }
        )
    return rows


def compute_feature_standardization(
    feature_data: List[Dict[str, float]],
    *,
    feature_order: Tuple[str, ...] = SOFTMAX_FEATURE_ORDER,
):
    arr = np.asarray(
        [[float(row[feature]) for feature in feature_order] for row in feature_data],
        dtype=float,
    )
    center = arr.mean(axis=0)
    scale = arr.std(axis=0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (
        {feature: float(center[idx]) for idx, feature in enumerate(feature_order)},
        {feature: float(scale[idx]) for idx, feature in enumerate(feature_order)},
    )


def apply_feature_standardization(
    feature_data: List[Dict[str, float]],
    feature_center: Dict[str, float],
    feature_scale: Dict[str, float],
    *,
    feature_order: Tuple[str, ...] = SOFTMAX_FEATURE_ORDER,
):
    out = []
    for row in feature_data:
        scaled = dict(row)
        for feature in feature_order:
            center = float(feature_center.get(feature, 0.0))
            scale = float(feature_scale.get(feature, 1.0))
            if abs(scale) <= 1e-12:
                scale = 1.0
            scaled[feature] = (float(row[feature]) - center) / scale
        out.append(scaled)
    return out


def _softmax_transport_predictions(
    *,
    feature_data: List[Dict[str, float]],
    reference_ion: str,
    nonref_ions: List[str],
    intercept: Dict[str, float],
    coef: Dict[str, Dict[str, float]],
):
    out = []
    for feat in feature_data:
        eta = {reference_ion: 0.0}
        for ion in nonref_ions:
            eta[ion] = float(intercept.get(ion, 0.0)) + sum(
                float(coef.get(ion, {}).get(f, 0.0)) * float(feat[f])
                for f in SOFTMAX_FEATURE_ORDER
            )
        denom = 1.0 + sum(np.exp(eta[ion]) for ion in nonref_ions)
        pred = {reference_ion: float(1.0 / denom)}
        for ion in nonref_ions:
            pred[ion] = float(np.exp(eta[ion]) / denom)
        out.append(pred)
    return out


@register_transport_number_surrogate("softmax_covariates")
def build_softmax_covariates(b: pyo.Block, eps: float = 1e-12) -> None:
    def _build_for_block(bd: ProcessBlockData, ind):
        m = bd.model()
        fs = m.sample_blk[ind].proc.fs
        ed = fs.EDstack
        X = ed.diluate.length_domain

        if bd._reference_ion not in bd.cation_set:
            raise ValueError(
                f"Reference ion '{bd._reference_ion}' not in cation set {bd.cation_set}."
            )

        J = [cation for cation in bd.cation_set if cation != bd._reference_ion]
        bd.eps = pyo.Param(initialize=eps, mutable=True)
        bd.nonref_ion_set = pyo.Set(initialize=J, ordered=True)
        bd.feature_set = pyo.Set(initialize=list(SOFTMAX_FEATURE_ORDER), ordered=True)
        bd.feature_center = pyo.Param(
            bd.feature_set,
            initialize=lambda _b, _f: 0.0,
            mutable=True,
        )
        bd.feature_scale = pyo.Param(
            bd.feature_set,
            initialize=lambda _b, _f: 1.0,
            mutable=True,
        )
        bd.score_intercept = pyo.Var(bd.nonref_ion_set, bounds=(-20, 20), initialize=0.0)
        bd.score_coef = pyo.Var(
            bd.nonref_ion_set,
            bd.feature_set,
            bounds=(-20, 20),
            initialize=0.0,
        )

        bd.feature_expr = pyo.Expression(
            bd.feature_set,
            X,
            rule=lambda _b, f, x: _feature_expr_rule(_b, fs, ed, f, x),
        )
        bd.eta = pyo.Expression(
            bd.nonref_ion_set,
            X,
            rule=lambda _b, ion, x: _b.score_intercept[ion]
            + sum(_b.score_coef[ion, f] * _b.feature_expr[f, x] for f in _b.feature_set),
        )
        bd.softmax_denom = pyo.Expression(
            X,
            rule=lambda _b, x: 1.0 + sum(pyo.exp(_b.eta[ion, x]) for ion in _b.nonref_ion_set),
        )

        def _ref_rule(_b, x):
            return ed.ion_trans_number_membrane["cem", _b._reference_ion, x] == 1.0 / _b.softmax_denom[x]

        bd.t_ref_ion_cem = pyo.Constraint(X, rule=_ref_rule)

        def _nonref_rule(_b, ion, x):
            return ed.ion_trans_number_membrane["cem", ion, x] == pyo.exp(_b.eta[ion, x]) / _b.softmax_denom[x]

        bd.t_nonref_ion_cem = pyo.Constraint(bd.nonref_ion_set, X, rule=_nonref_rule)

    def _feature_expr_rule(_b, fs, ed, feature: str, x):
        if feature == "log_ca_na":
            raw = pyo.log(
                (ed.diluate.properties[0, x].conc_mol_phase_comp["Liq", "Ca_2+"] + _b.eps)
                / (ed.diluate.properties[0, x].conc_mol_phase_comp["Liq", _b._reference_ion] + _b.eps)
            )
            return (raw - _b.feature_center[feature]) / _b.feature_scale[feature]
        if feature == "log_mg_na":
            raw = pyo.log(
                (ed.diluate.properties[0, x].conc_mol_phase_comp["Liq", "Mg_2+"] + _b.eps)
                / (ed.diluate.properties[0, x].conc_mol_phase_comp["Liq", _b._reference_ion] + _b.eps)
            )
            return (raw - _b.feature_center[feature]) / _b.feature_scale[feature]
        if feature == "log_current_density":
            raw = pyo.log(ed.current_density_x[0, x] + _b.eps)
            return (raw - _b.feature_center[feature]) / _b.feature_scale[feature]
        raise KeyError(f"Unsupported softmax feature: {feature}")

    if b.is_indexed():
        for ind, bd in b.items():
            _build_for_block(bd, ind)
    else:
        _build_for_block(b, b.index())


@register_transport_number_surrogate_init("softmax_covariates")
def init_softmax_covariates(
    conc_data: List[Dict[str, float]] | None,
    trans_number_data: List[Dict[str, float]],
    fitting_coef_guess: Dict[str, float] | None,
    reference_ion: str,
    coef_bounds: Dict[str, Tuple[float, float]] | None = None,
    log_objective: bool = False,
    polynomial_degree: int = 1,
    eps: float = 1e-12,
    plot_results: bool = True,
    feature_data: List[Dict[str, float]] | None = None,
    figure_output_html: str | Path | None = None,
    figure_output_pdf: str | Path | None = None,
    standardize_features: bool = True,
):
    del conc_data, log_objective, polynomial_degree
    if feature_data is None:
        raise ValueError("softmax_covariates initializer requires feature_data.")
    if len(feature_data) != len(trans_number_data):
        raise ValueError("feature_data and trans_number_data must have the same length.")

    ions = list(trans_number_data[0].keys())
    J = [ion for ion in ions if ion != reference_ion]
    n_feat = len(SOFTMAX_FEATURE_ORDER)

    if standardize_features:
        feature_center, feature_scale = compute_feature_standardization(feature_data)
        fit_feature_data = apply_feature_standardization(
            feature_data,
            feature_center,
            feature_scale,
        )
    else:
        feature_center = {feature: 0.0 for feature in SOFTMAX_FEATURE_ORDER}
        feature_scale = {feature: 1.0 for feature in SOFTMAX_FEATURE_ORDER}
        fit_feature_data = feature_data
    Y = {ion: np.asarray([float(row[ion]) for row in trans_number_data], dtype=float) for ion in ions}

    def _unpack(theta):
        idx = 0
        intercept = {}
        coef = {}
        for ion in J:
            intercept[ion] = float(theta[idx])
            idx += 1
            coef[ion] = {}
            for feat in SOFTMAX_FEATURE_ORDER:
                coef[ion][feat] = float(theta[idx])
                idx += 1
        return intercept, coef

    def _predict(theta):
        intercept, coef = _unpack(theta)
        return _softmax_transport_predictions(
            feature_data=fit_feature_data,
            reference_ion=reference_ion,
            nonref_ions=J,
            intercept=intercept,
            coef=coef,
        )

    def _objective(theta):
        pred = _predict(theta)
        sse = 0.0
        for ion in ions:
            yp = np.asarray([row[ion] for row in pred], dtype=float)
            sse += float(np.sum((Y[ion] - yp) ** 2))
        return sse

    if fitting_coef_guess is None:
        theta0 = np.zeros(len(J) * (1 + n_feat), dtype=float)
    else:
        theta0 = []
        for ion in J:
            theta0.append(float(fitting_coef_guess.get("intercept", {}).get(ion, 0.0)))
            for feat in SOFTMAX_FEATURE_ORDER:
                theta0.append(float(fitting_coef_guess.get("coef", {}).get(ion, {}).get(feat, 0.0)))
        theta0 = np.asarray(theta0, dtype=float)

    if coef_bounds is None:
        bounds = [(-10.0, 10.0)] * len(theta0)
    else:
        bounds = []
        for ion in J:
            bounds.append(tuple(coef_bounds.get("intercept", {}).get(ion, (-10.0, 10.0))))
            for feat in SOFTMAX_FEATURE_ORDER:
                bounds.append(tuple(coef_bounds.get("coef", {}).get(ion, {}).get(feat, (-10.0, 10.0))))

    outcome = minimize(_objective, theta0, method="L-BFGS-B", bounds=bounds)
    theta = outcome.x
    intercept, coef = _unpack(theta)
    pred = _predict(theta)

    fig = go.Figure()
    for ion in ions:
        y_true = Y[ion]
        y_pred = np.asarray([row[ion] for row in pred], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name=ion,
                hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
            )
        )
    all_vals = np.concatenate([Y[ion] for ion in ions])
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Ideal (y=x)",
        )
    )
    fig.update_layout(
        title="Predicted vs. Actual Transport Numbers (Softmax Covariates)",
        xaxis_title="Actual Transport Number",
        yaxis_title="Predicted Transport Number",
        template="plotly_white",
    )

    if figure_output_html is not None:
        fig.write_html(str(figure_output_html))
    if figure_output_pdf is not None:
        try:
            fig.write_image(str(figure_output_pdf))
        except Exception:
            pass
    if plot_results:
        fig.show()

    return {
        "intercept": intercept,
        "coef": coef,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "standardize_features": bool(standardize_features),
    }


@register_transport_number_surrogate_fn_identity("softmax_covariates")
def softmax_covariates_fn(
    ions: List[str],
    reference_ion: str,
    fitted_coef_dict: dict,
    feature_data: List[Dict[str, float]],
    **kwargs,
):
    del kwargs
    nonref_ions = [ion for ion in ions if ion != reference_ion]
    scaled_feature_data = apply_feature_standardization(
        feature_data,
        fitted_coef_dict.get("feature_center", {}),
        fitted_coef_dict.get("feature_scale", {}),
    )
    pred = _softmax_transport_predictions(
        feature_data=scaled_feature_data,
        reference_ion=reference_ion,
        nonref_ions=nonref_ions,
        intercept=fitted_coef_dict.get("intercept", {}),
        coef=fitted_coef_dict.get("coef", {}),
    )
    arrays = {ion: np.asarray([row[ion] for row in pred], dtype=float) for ion in ions}
    log_arrays = {ion: np.log(arrays[ion] + 1e-12) for ion in ions}
    return arrays, log_arrays


def predict_ti_by_softmax_covariates(
    *,
    ions: List[str],
    reference_ion: str,
    fitted_coef_dict: dict,
    feature_data: List[Dict[str, float]],
):
    t_pred, _ = softmax_covariates_fn(
        ions=ions,
        reference_ion=reference_ion,
        fitted_coef_dict=fitted_coef_dict,
        feature_data=feature_data,
    )
    return t_pred
