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
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Callable, Literal, Dict
import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass


Number = Union[int, float]


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0 or not math.isfinite(raw_step):
        return 1.0
    exp = math.floor(math.log10(raw_step))
    base = raw_step / (10**exp)
    candidates = [1, 2, 2.5, 5, 10]
    nice_base = min(candidates, key=lambda c: abs(c - base))
    return nice_base * (10**exp)


def _nice_range(
    vmin: float, vmax: float, n_ticks: int = 6
) -> Tuple[float, float, float]:
    """Compute (tick0, dtick, tick_end) covering [vmin, vmax] with 'nice' spacing."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return 0.0, 1.0, 1.0
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.1
        vmin, vmax = vmin - pad, vmax + pad

    span = vmax - vmin
    raw_step = span / max(n_ticks - 1, 1)
    dtick = _nice_step(raw_step)

    tick0 = math.floor(vmin / dtick) * dtick
    tick_end = math.ceil(vmax / dtick) * dtick

    return tick0, dtick, tick_end


def scatter_2d_series(
    x: pd.Series,
    y: pd.Series,
    *,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    marker_color: str = "#000000",
    marker_symbol: str = "circle",
    marker_size: int = 8,
    marker_opacity: float = 1,
    show_diagonal: bool = False,
    diagonal_kwargs: Optional[dict] = None,
    tick_count: int = 6,
    width: int = 650,
    height: int = 650,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    x_tick0: Optional[float] = None,
    x_dtick: Optional[float] = None,
    y_tick0: Optional[float] = None,
    y_dtick: Optional[float] = None,
    x_label_font_size: int = 14,
    y_label_font_size: int = 14,
    tick_font_size: int = 12,
) -> go.Figure:
    """
    Plot y vs x as a scattered (scatter) plot using plotly.graph_objects.

    Features:
    - white background
    - square figure size (width==height by default)
    - all boxed borders (mirrored axes lines)
    - ticks cover the value range with "nice" regular digits (custom dtick/tick0)
    - marker color and shape controlled by arguments

    Parameters
    ----------
    x, y : pd.Series
        Series to plot. Values are aligned on index intersection.
    marker_color : str
        Any Plotly color string (hex, rgb, named).
    marker_symbol : str
        Plotly marker symbol name (e.g., "circle", "square", "diamond", "x", "triangle-up").
    show_diagonal : bool
        If True, draw y=x reference line spanning the visible range.

    Returns
    -------
    go.Figure
    """
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise TypeError("x and y must both be pandas Series.")

    # Align & drop NaNs
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping non-NaN data to plot after aligning indices.")

    xv = df["x"].astype(float).to_numpy()
    yv = df["y"].astype(float).to_numpy()

    # X axis
    if x_range is None:
        x0, xdtick, xend = _nice_range(
            float(df["x"].min()), float(df["x"].max()), n_ticks=tick_count
        )
        x_range_use = (x0, xend)
        x_tick0_use = x0
        x_dtick_use = xdtick
    else:
        x_range_use = x_range
        x_tick0_use = x_tick0
        x_dtick_use = x_dtick

    # Y axis
    if y_range is None:
        y0, ydtick, yend = _nice_range(
            float(df["y"].min()), float(df["y"].max()), n_ticks=tick_count
        )
        y_range_use = (y0, yend)
        y_tick0_use = y0
        y_dtick_use = ydtick
    else:
        y_range_use = y_range
        y_tick0_use = y_tick0
        y_dtick_use = y_dtick

    # Build figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xv,
            y=yv,
            mode="markers",
            marker=dict(
                color=marker_color,
                symbol=marker_symbol,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(width=0),  # clean look
            ),
            name="data",
            hovertemplate=(
                f"{(x_label or x.name or 'x')}: %{{x}}<br>"
                f"{(y_label or y.name or 'y')}: %{{y}}<extra></extra>"
            ),
        )
    )

    if show_diagonal:
        diag = diagonal_kwargs or {}
        # Span across the combined visible range so the diagonal is meaningful
        lo = min(x_range_use[0], y_range_use[0])
        hi = max(x_range_use[1], y_range_use[1])

        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(width=1, dash="dash", color=diag.get("color", "black")),
                name=diag.get("name", "y = x"),
                hoverinfo="skip",
            )
        )

    # Labels
    x_axis_title = x_label or (x.name if x.name is not None else "x")
    y_axis_title = y_label or (y.name if y.name is not None else "y")

    # Layout: white background, boxed borders, square size
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=25, t=60 if title else 25, b=60),
        showlegend=False,
    )

    # Axes styling: boxed borders, outward ticks, nice dtick/tick0, range padded to ticks
    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,  # puts lines on all sides (boxed)
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )

    fig.update_xaxes(
        title_text=x_axis_title,
        title_font=dict(size=x_label_font_size),
        tickfont=dict(size=tick_font_size),
        tick0=x_tick0_use,
        dtick=x_dtick_use,
        range=list(x_range_use),
        **axis_common,
    )
    fig.update_yaxes(
        title_text=y_axis_title,
        title_font=dict(size=y_label_font_size),
        tickfont=dict(size=tick_font_size),
        tick0=y_tick0_use,
        dtick=y_dtick_use,
        range=list(y_range_use),
        **axis_common,
    )

    return fig


def scatter_3d_series(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    *,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    marker_color: str = "z",
    colorscale: str = "Greys",
    reverse_colorscale: bool = False,
    show_colorbar: bool = False,
    marker_symbol: str = "circle",
    marker_size: int = 5,
    marker_opacity: float = 0.95,
    tick_count: int = 6,
    width: int = 650,
    height: int = 650,
    camera_eye: Tuple[float, float, float] = (1.55, 1.35, 0.95),
) -> go.Figure:
    """
    3D scatter with styling similar to the 2D version, but with better depth cues:
    - white background
    - closed-looking 3D box (subtle planes + black axis edges)
    - default grey depth shading (colorscale) to make 3D structure more readable
    - optional color mapping: marker_color="z" (default) or "distance"
    """
    if not (
        isinstance(x, pd.Series)
        and isinstance(y, pd.Series)
        and isinstance(z, pd.Series)
    ):
        raise TypeError("x, y, z must all be pandas Series.")

    df = pd.concat(
        [x.rename("x"), y.rename("y"), z.rename("z")], axis=1, join="inner"
    ).dropna()
    if df.empty:
        raise ValueError("No overlapping non-NaN data to plot after aligning indices.")

    xv = df["x"].astype(float).to_numpy()
    yv = df["y"].astype(float).to_numpy()
    zv = df["z"].astype(float).to_numpy()

    # ticks/ranges
    x0, xdtick, xend = _nice_range(
        float(df["x"].min()), float(df["x"].max()), n_ticks=tick_count
    )
    y0, ydtick, yend = _nice_range(
        float(df["y"].min()), float(df["y"].max()), n_ticks=tick_count
    )
    z0, zdtick, zend = _nice_range(
        float(df["z"].min()), float(df["z"].max()), n_ticks=tick_count
    )

    x_axis_title = x_label or (x.name if x.name is not None else "x")
    y_axis_title = y_label or (y.name if y.name is not None else "y")
    z_axis_title = z_label or (z.name if z.name is not None else "z")

    # Depth cue: default color by z (Greys). Alternatives: "distance" or fixed color string.
    if marker_color == "z":
        color_values = zv
        cmin, cmax = float(z0), float(zend)
        use_scale = True
    elif marker_color == "distance":
        # distance from centroid for a mild depth/structure cue
        cx, cy, cz = float(df["x"].mean()), float(df["y"].mean()), float(df["z"].mean())
        color_values = ((xv - cx) ** 2 + (yv - cy) ** 2 + (zv - cz) ** 2) ** 0.5
        cmin, cmax = float(color_values.min()), float(color_values.max())
        use_scale = True
    else:
        color_values = marker_color
        cmin, cmax = None, None
        use_scale = False

    marker_dict = dict(
        size=marker_size,
        opacity=marker_opacity,
        symbol=marker_symbol,
        line=dict(width=0.5, color="rgba(0,0,0,0.45)"),
    )

    if use_scale:
        marker_dict.update(
            dict(
                color=color_values,
                colorscale=colorscale,
                reversescale=reverse_colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=show_colorbar,
            )
        )
    else:
        marker_dict.update(dict(color=color_values))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xv,
                y=yv,
                z=zv,
                mode="markers",
                marker=marker_dict,
                hovertemplate=(
                    f"{x_axis_title}: %{{x}}<br>"
                    f"{y_axis_title}: %{{y}}<br>"
                    f"{z_axis_title}: %{{z}}<extra></extra>"
                ),
                name="data",
            )
        ]
    )

    plane_color = "rgba(0,0,0,0.04)"

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60 if title else 25, b=20),
        showlegend=False,
        scene=dict(
            bgcolor="white",
            aspectmode="cube",
            camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])),
            xaxis=dict(
                title=x_axis_title,
                range=[x0, xend],
                tick0=x0,
                dtick=xdtick,
                showbackground=True,
                backgroundcolor=plane_color,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                ticklen=6,
                tickwidth=1,
                tickcolor="black",
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title=y_axis_title,
                range=[y0, yend],
                tick0=y0,
                dtick=ydtick,
                showbackground=True,
                backgroundcolor=plane_color,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                ticklen=6,
                tickwidth=1,
                tickcolor="black",
                showgrid=False,
                zeroline=False,
            ),
            zaxis=dict(
                title=z_axis_title,
                range=[z0, zend],
                tick0=z0,
                dtick=zdtick,
                showbackground=True,
                backgroundcolor=plane_color,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                ticklen=6,
                tickwidth=1,
                tickcolor="black",
                showgrid=False,
                zeroline=False,
            ),
        ),
    )

    return fig


def violin_box_params(
    params: Union[pd.DataFrame, Dict[str, pd.Series], Sequence[pd.Series]],
    *,
    param_names: Optional[Sequence[str]] = None,
    display_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    y_label: str = "Parameter value",
    show_points: str = "outliers",  # "outliers" | "all" | False
    show_violin: bool = True,
    show_box: bool = True,
    marker_color: str = "#000000",
    line_color: str = "#000000",
    width: int = 650,
    height: int = 450,
    tick_count: int = 6,
    y_range: Optional[Tuple[float, float]] = None,
    y_tick0: Optional[float] = None,
    y_dtick: Optional[float] = None,
    x_label_font_size: int = 14,
    y_label_font_size: int = 14,
    tick_font_size: int = 12,
) -> go.Figure:
    """

    Input options
    -------------
    params can be:
      1) pd.DataFrame: columns as different parameters
      2) dict[str, pd.Series]: each series is values of a parameter
      3) sequence[pd.Series]: provide param_names to label them (or uses series.name)

    param_names:
      - If params is a DataFrame: optional subset/order of columns to plot
      - If params is dict: optional subset/order of keys to plot
      - If params is sequence: names for each series (or uses series.name)

    """
    # --- normalize into long-form dataframe: (param, value) ---
    if isinstance(params, pd.DataFrame):
        cols = list(params.columns) if param_names is None else list(param_names)
        missing = [c for c in cols if c not in params.columns]
        if missing:
            raise ValueError(f"param_names not found in DataFrame columns: {missing}")

        plot_order = list(display_names) if display_names is not None else cols
        label_map = dict(zip(cols, plot_order))

        df_long = (
            params[cols]
            .rename(columns=label_map)
            .melt(var_name="param", value_name="value")
            .dropna()
        )

        plot_order = list(display_names) if display_names is not None else cols

    elif isinstance(params, dict):
        keys = list(params.keys()) if param_names is None else list(param_names)
        missing = [k for k in keys if k not in params]
        if missing:
            raise ValueError(f"param_names not found in params dict: {missing}")
        frames = []
        for k in keys:
            s = params[k]
            if not isinstance(s, pd.Series):
                s = pd.Series(s, name=k)
            frames.append(
                pd.DataFrame({"param": k, "value": s.astype(float).to_numpy()})
            )
        df_long = pd.concat(frames, axis=0, ignore_index=True).dropna()
        plot_order = keys

    else:
        series_list = list(params)
        names = (
            [s.name for s in series_list] if param_names is None else list(param_names)
        )
        if len(names) != len(series_list):
            raise ValueError("param_names length must match number of series.")
        frames = []
        for s, nm in zip(series_list, names):
            if not isinstance(s, pd.Series):
                s = pd.Series(s, name=nm)
            frames.append(
                pd.DataFrame({"param": nm, "value": s.astype(float).to_numpy()})
            )
        df_long = pd.concat(frames, axis=0, ignore_index=True).dropna()
        plot_order = names

    if df_long.empty:
        raise ValueError("No data to plot (all values are NaN/empty).")

    # --- y-axis nice defaults unless overridden ---
    vmin = float(df_long["value"].min())
    vmax = float(df_long["value"].max())

    if y_range is None:
        y0, ydtick, yend = _nice_range(vmin, vmax, n_ticks=tick_count)
        y_range_use = (y0, yend)
        y_tick0_use = y0 if y_tick0 is None else y_tick0
        y_dtick_use = ydtick if y_dtick is None else y_dtick
    else:
        y_range_use = y_range
        if (y_tick0 is None) or (y_dtick is None):
            y0, ydtick, _ = _nice_range(
                y_range_use[0], y_range_use[1], n_ticks=tick_count
            )
            y_tick0_use = y0 if y_tick0 is None else y_tick0
            y_dtick_use = ydtick if y_dtick is None else y_dtick
        else:
            y_tick0_use = y_tick0
            y_dtick_use = y_dtick

    # --- figure ---
    fig = go.Figure()

    for p in plot_order:
        vals = df_long.loc[df_long["param"] == p, "value"].astype(float).to_numpy()
        if vals.size == 0:
            continue
        fig.add_trace(
            go.Violin(
                x=[p] * vals.size,
                y=vals,
                name=str(p),
                box_visible=show_box,
                meanline_visible=False,
                points=show_points if show_points else False,
                pointpos=0,
                jitter=0.2 if show_points else 0,
                spanmode="hard",
                line=dict(color=line_color, width=1),
                fillcolor="rgba(0,0,0,0.10)" if show_violin else "rgba(0,0,0,0)",
                opacity=1.0,
                marker=dict(
                    color=marker_color,
                    size=5,
                    line=dict(width=0),
                ),
                showlegend=False,
                visible=True,
            )
        )
        if not show_violin:
            # If violin hidden, switch to a box-only look via box_visible already,
            pass

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=25, t=60 if title else 25, b=60),
        showlegend=False,
        violingap=0.25,
        violinmode="group",
    )

    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )

    fig.update_xaxes(
        title_text="",  # parameter names already on tick labels
        title_font=dict(size=x_label_font_size),
        tickfont=dict(size=tick_font_size),
        categoryorder="array",
        categoryarray=list(plot_order),
        **axis_common,
    )

    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(size=y_label_font_size),
        tickfont=dict(size=tick_font_size),
        tick0=y_tick0_use,
        dtick=y_dtick_use,
        range=list(y_range_use),
        **axis_common,
    )

    return fig


DataLike = Union[pd.DataFrame, dict, Sequence[pd.Series]]


def _to_df(
    data: DataLike, *, col_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if col_names is not None:
            missing = [c for c in col_names if c not in df.columns]
            if missing:
                raise ValueError(f"col_names not found in DataFrame: {missing}")
            df = df[list(col_names)]
        return df

    if isinstance(data, dict):
        keys = list(data.keys()) if col_names is None else list(col_names)
        missing = [k for k in keys if k not in data]
        if missing:
            raise ValueError(f"col_names not found in dict keys: {missing}")
        return pd.DataFrame(
            {
                k: (data[k] if isinstance(data[k], pd.Series) else pd.Series(data[k]))
                for k in keys
            }
        )

    seq = list(data)
    names = (
        [getattr(s, "name", None) for s in seq]
        if col_names is None
        else list(col_names)
    )
    if any(n is None for n in names):
        raise ValueError("Series in sequence must have .name or provide col_names.")
    if len(names) != len(seq):
        raise ValueError("col_names length must match number of series.")
    return pd.DataFrame(
        {
            n: (s if isinstance(s, pd.Series) else pd.Series(s))
            for n, s in zip(names, seq)
        }
    )


def zscore_heatmap(
    data,
    *,
    col_names=None,
    row_names=None,
    display_col_names=None,
    display_row_names=None,
    title: Optional[str] = None,
    z_clip: Optional[float] = 3.0,
    ddof: int = 1,
    width: int = 900,
    height: int = 520,
    tick_font_size: int = 11,
) -> go.Figure:
    """Column-wise z-score heatmap."""
    df = _to_df(data, col_names=col_names).apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    if row_names is not None:
        missing = [r for r in row_names if r not in df.index]
        if missing:
            raise ValueError(f"row_names not found in DataFrame index: {missing}")
        df = df.loc[row_names]

    col_labels = list(df.columns)
    row_labels = list(df.index)

    if display_col_names is not None:
        if len(display_col_names) != len(col_labels):
            raise ValueError("display_col_names length must match number of columns.")
        col_labels = list(display_col_names)

    if display_row_names is not None:
        if len(display_row_names) != len(row_labels):
            raise ValueError("display_row_names length must match number of rows.")
        row_labels = list(display_row_names)

    mu = df.mean(axis=0, skipna=True).to_numpy()
    sd = df.std(axis=0, ddof=ddof, skipna=True).to_numpy()
    sd = np.where(sd == 0, np.nan, sd)

    Z = (df.to_numpy(dtype=float) - mu) / sd
    if z_clip is not None:
        Z = np.clip(Z, -float(z_clip), float(z_clip))

    colorscale = [
        [0.0, "#2b6cb0"],
        [0.5, "#ffffff"],
        [1.0, "#c53030"],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=Z,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            zmid=0.0,
            colorbar=dict(title="z", ticks="outside"),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=40, t=60 if title else 25, b=90),
    )

    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=tick_font_size),
    )

    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)

    return fig


def mean_pm_nstd_plot(
    data,
    *,
    col_names=None,
    display_col_names=None,
    title: Optional[str] = None,
    y_label: str = "Value",
    n_std: float = 1.0,
    ddof: int = 1,
    width: int = 900,
    height: int = 420,
    marker_color: str = "#000000",
    marker_size: int = 7,
    tick_font_size: int = 12,
    x_label_font_size: int = 14,
    y_label_font_size: int = 14,
) -> go.Figure:
    """Mean with ± n_std*SD error bars per column."""
    df = _to_df(data, col_names=col_names).apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    cols = list(df.columns)
    disp_cols = list(display_col_names) if display_col_names is not None else cols
    if len(disp_cols) != len(cols):
        raise ValueError("display_col_names length must match number of columns.")

    mu = df[cols].mean(axis=0, skipna=True)
    sd = df[cols].std(axis=0, ddof=ddof, skipna=True)
    err = (float(n_std) * sd).to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=disp_cols,
            y=mu.to_numpy(dtype=float),
            mode="markers",
            marker=dict(color=marker_color, size=marker_size, line=dict(width=0)),
            error_y=dict(type="data", array=err, visible=True, thickness=1, width=6),
            hovertemplate="%{x}<br>mean=%{y:.6g}<br>±=%{error_y.array:.6g}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=25, t=60 if title else 25, b=90),
        showlegend=False,
    )

    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=tick_font_size),
    )

    fig.update_xaxes(
        title_text="", title_font=dict(size=x_label_font_size), **axis_common
    )
    fig.update_yaxes(
        title_text=y_label, title_font=dict(size=y_label_font_size), **axis_common
    )

    return fig


SummaryMethod = Union[
    Literal["median", "mean"],
    Callable[[np.ndarray, int], np.ndarray],
]


@dataclass
class BandSummary:
    x: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    center_label: str


def _as_1d(a, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _as_2d(a, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def summarize_predictions(
    x: np.ndarray,
    preds: np.ndarray,
    *,
    summary: SummaryMethod = "median",
    interval: Union[float, Tuple[float, float]] = 0.90,
) -> BandSummary:
    """
    Summarize predictive samples into a center line + uncertainty band.

    Parameters
    ----------
    x : (n_points,)
        X locations for the predictions.
    preds : (n_draws, n_points)
        Predictive draws/samples. (e.g., ensemble members, bootstrap, posterior samples).
    summary : "median" | "mean" | callable
        Center line statistic. If callable, signature must be f(preds, axis)->center.
    interval : float | (low, high)
        If float in (0,1): central interval level, e.g. 0.90 -> 5%..95%.
        If tuple: explicit quantiles (low, high) in [0,1], e.g. (0.05, 0.95).

    Returns
    -------
    BandSummary
    """
    x = _as_1d(x, "x")
    preds = _as_2d(preds, "preds")
    if preds.shape[1] != x.size:
        raise ValueError(f"preds has {preds.shape[1]} points but x has {x.size}")

    if isinstance(interval, tuple):
        qlo, qhi = interval
        if not (0 <= qlo < qhi <= 1):
            raise ValueError("interval tuple must satisfy 0 <= low < high <= 1")
        level = qhi - qlo
    else:
        level = float(interval)
        if not (0 < level < 1):
            raise ValueError("interval must be in (0,1)")
        qlo = (1 - level) / 2
        qhi = 1 - qlo

    lower = np.quantile(preds, qlo, axis=0)
    upper = np.quantile(preds, qhi, axis=0)

    if summary == "median":
        center = np.median(preds, axis=0)
        center_label = "Median"
    elif summary == "mean":
        center = np.mean(preds, axis=0)
        center_label = "Mean"
    elif callable(summary):
        center = np.asarray(summary(preds, 0))
        if center.shape != (x.size,):
            raise ValueError("custom summary must return shape (n_points,)")
        center_label = getattr(summary, "__name__", "Center")
    else:
        raise ValueError("summary must be 'median', 'mean', or a callable")

    return BandSummary(
        x=x,
        center=center,
        lower=lower,
        upper=upper,
        level=level,
        center_label=center_label,
    )


## Prediction with uncertainty

SummaryMethod = Literal["median", "mean"]


@dataclass(frozen=True)
class BandSummary:
    x: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    center_label: str


def _as_1d(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _summarize_predictions(
    x: np.ndarray,
    preds: np.ndarray,
    *,
    summary: SummaryMethod = "median",
    interval: Union[float, Tuple[float, float]] = 0.90,
) -> BandSummary:
    x = _as_1d(x, "x")
    preds = np.asarray(preds)

    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D (n_draws, n_points), got {preds.shape}")
    if preds.shape[1] != x.size:
        raise ValueError(
            f"preds second dim must match x; got {preds.shape[1]} vs {x.size}"
        )

    # interval -> (q_lo, q_hi) and level for labeling
    if isinstance(interval, (float, int)):
        level = float(interval)
        if not (0.0 < level < 1.0):
            raise ValueError("interval as float must be in (0, 1)")
        alpha = (1.0 - level) / 2.0
        q_lo, q_hi = alpha, 1.0 - alpha
    else:
        q_lo, q_hi = float(interval[0]), float(interval[1])
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("interval tuple must satisfy 0 <= lo < hi <= 1")
        level = q_hi - q_lo

    if summary == "median":
        center = np.nanmedian(preds, axis=0)
        center_label = "median"
    elif summary == "mean":
        center = np.nanmean(preds, axis=0)
        center_label = "mean"
    else:
        raise ValueError("summary must be 'median' or 'mean'")

    lower = np.nanquantile(preds, q_lo, axis=0)
    upper = np.nanquantile(preds, q_hi, axis=0)

    return BandSummary(
        x=np.asarray(x),
        center=np.asarray(center),
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        level=level,
        center_label=center_label,
    )


def plot_prediction_with_uncertainty_band(
    x: np.ndarray,
    preds: np.ndarray,
    *,
    y_obs: Optional[np.ndarray] = None,
    x_obs: Optional[np.ndarray] = None,
    summary: SummaryMethod = "median",
    interval: Union[float, Tuple[float, float]] = 0.90,
    sort_x: bool = True,
    show_band: bool = True,
    show_center: bool = True,
    show_obs: bool = True,
    obs_style: Literal["points", "points+line"] = "points",
    obs_marker: str = "circle",
    obs_marker_size: int = 6,
    obs_marker_opacity: float = 1.0,
    band_opacity: float = 0.25,
    band_color: str = "black",  # used as rgba fill base
    center_color: str = "black",
    center_linewidth: float = 2.0,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    width: int = 700,
    height: int = 700,
    # axis tick formatting in the same spirit as your scatter function
    x_tick0: Optional[float] = None,
    x_dtick: Optional[float] = None,
    y_tick0: Optional[float] = None,
    y_dtick: Optional[float] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    x_label_font_size: int = 26,
    y_label_font_size: int = 26,
    tick_font_size: int = 22,
    fig: Optional[go.Figure] = None,
) -> Tuple[go.Figure, BandSummary]:
    """
    Plot predictive center line + uncertainty band (Plotly), with formatting consistent
    with your scatter plot function (white background, boxed axes, outward ticks, no grid).

    Parameters
    ----------
    x : (n_points,)
    preds : (n_draws, n_points)
    y_obs : optional (n_obs,)
    x_obs : optional (n_obs,)
    interval : float (e.g., 0.90) or quantile tuple (lo, hi)
    fig : optional go.Figure (if you want to add to an existing figure)

    Returns
    -------
    (fig, band_summary)
    """
    band = _summarize_predictions(x, preds, summary=summary, interval=interval)

    if sort_x:
        order = np.argsort(band.x)
        band = BandSummary(
            x=band.x[order],
            center=band.center[order],
            lower=band.lower[order],
            upper=band.upper[order],
            level=band.level,
            center_label=band.center_label,
        )

    if fig is None:
        fig = go.Figure()

    # helpers for rgba
    def _rgba_black(alpha: float) -> str:
        # base is black for "clean look"; if you want other colors later, expand this
        return f"rgba(0,0,0,{alpha})"

    band_fill = _rgba_black(band_opacity) if band_color == "black" else band_color

    # Uncertainty band (filled polygon)
    if show_band:
        fig.add_trace(
            go.Scatter(
                x=band.x,
                y=band.lower,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band.x,
                y=band.upper,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=(
                    band_fill if "rgba" in band_fill else _rgba_black(band_opacity)
                ),
                hoverinfo="skip",
                showlegend=False,
                name=f"{int(round(band.level * 100))}% band",
            )
        )

    # Center line
    if show_center:
        fig.add_trace(
            go.Scatter(
                x=band.x,
                y=band.center,
                mode="lines",
                line=dict(color=center_color, width=center_linewidth),
                name=band.center_label,
                hovertemplate=(
                    f"{(x_label or 'x')}: %{{x}}<br>"
                    f"{(y_label or 'y')}: %{{y}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Observations
    if y_obs is not None and show_obs:
        y_obs_ = _as_1d(y_obs, "y_obs")
        if x_obs is None:
            if y_obs_.size != band.x.size:
                raise ValueError("x_obs is None, so y_obs must have same length as x")
            x_obs_use = band.x
        else:
            x_obs_use = _as_1d(x_obs, "x_obs")
            if x_obs_use.size != y_obs_.size:
                raise ValueError("x_obs and y_obs must have the same length")

        mode = "markers" if obs_style == "points" else "lines+markers"
        fig.add_trace(
            go.Scatter(
                x=x_obs_use,
                y=y_obs_,
                mode=mode,
                marker=dict(
                    color="black",
                    symbol=obs_marker,
                    size=obs_marker_size,
                    opacity=obs_marker_opacity,
                    line=dict(width=0),  # clean look
                ),
                line=(
                    dict(color="black", width=1) if obs_style == "points+line" else None
                ),
                name="Observed",
                hovertemplate=(
                    f"{(x_label or 'x')}: %{{x}}<br>"
                    f"{(y_label or 'y')}: %{{y}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Labels
    x_axis_title = x_label or "x"
    y_axis_title = y_label or "y"

    # Layout: white background, boxed borders, square size
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=25, t=60 if title else 25, b=60),
        showlegend=False,
    )

    # Axes styling: boxed borders, outward ticks, no grid
    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,  # boxed
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )

    x_update = dict(
        title_text=x_axis_title,
        title_font=dict(size=x_label_font_size),
        tickfont=dict(size=tick_font_size),
        **axis_common,
    )
    if x_tick0 is not None:
        x_update["tick0"] = x_tick0
    if x_dtick is not None:
        x_update["dtick"] = x_dtick
    if x_range is not None:
        x_update["range"] = list(x_range)

    y_update = dict(
        title_text=y_axis_title,
        title_font=dict(size=y_label_font_size),
        tickfont=dict(size=tick_font_size),
        **axis_common,
    )
    if y_tick0 is not None:
        y_update["tick0"] = y_tick0
    if y_dtick is not None:
        y_update["dtick"] = y_dtick
    if y_range is not None:
        y_update["range"] = list(y_range)

    fig.update_xaxes(**x_update)
    fig.update_yaxes(**y_update)

    return fig, band


def observed_vs_pred_with_uncertainty(
    y_obs: np.ndarray,
    preds: np.ndarray,
    *,
    interval: Union[float, Tuple[float, float]] = 0.90,
    center: str = "median",  # "median" or "mean"
    title: Optional[str] = None,
    x_label: str = "Observed",
    y_label: str = "Predicted",
    # style to match your scatter function
    width: int = 700,
    height: int = 700,
    tick_font_size: int = 22,
    x_label_font_size: int = 26,
    y_label_font_size: int = 26,
    # axis tick controls (optional)
    x_tick0: Optional[float] = None,
    x_dtick: Optional[float] = None,
    y_tick0: Optional[float] = None,
    y_dtick: Optional[float] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    # marker style
    marker_color: str = "black",
    marker_symbol: str = "circle",
    marker_size: int = 8,
    marker_opacity: float = 1.0,
    # uncertainty bar style
    err_width: float = 1.0,
    err_thickness: float = 1.0,
    # diagonal
    show_diagonal: bool = True,
    diagonal_dash: str = "dash",
    diagonal_width: float = 1.0,
    diagonal_color: str = "black",
) -> go.Figure:
    """
    Fig A: Observed vs Predicted (center across draws) with vertical uncertainty bars.
    Inputs:
      y_obs: (n_cases,)
      preds: (n_draws, n_cases)  e.g., (24, 10)
    """
    y_obs = np.asarray(y_obs).ravel()
    preds = np.asarray(preds)

    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D (n_draws, n_cases), got {preds.shape}")
    if preds.shape[1] != y_obs.size:
        raise ValueError(
            f"preds second dim must match y_obs length; got {preds.shape[1]} vs {y_obs.size}"
        )

    # interval -> quantiles
    if isinstance(interval, (float, int)):
        level = float(interval)
        if not (0.0 < level < 1.0):
            raise ValueError("interval as float must be in (0, 1)")
        alpha = (1.0 - level) / 2.0
        q_lo, q_hi = alpha, 1.0 - alpha
    else:
        q_lo, q_hi = float(interval[0]), float(interval[1])
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("interval tuple must satisfy 0 <= lo < hi <= 1")
        level = q_hi - q_lo

    # summary + bounds per case
    if center == "median":
        y_center = np.nanmedian(preds, axis=0)
        center_name = "Median"
    elif center == "mean":
        y_center = np.nanmean(preds, axis=0)
        center_name = "Mean"
    else:
        raise ValueError("center must be 'median' or 'mean'")

    y_lo = np.nanquantile(preds, q_lo, axis=0)
    y_hi = np.nanquantile(preds, q_hi, axis=0)

    # error bars need + and - distances from center
    err_plus = y_hi - y_center
    err_minus = y_center - y_lo

    # default ranges include both obs and interval ends, and keep square-ish
    if x_range is None or y_range is None:
        lo = np.nanmin([np.nanmin(y_obs), np.nanmin(y_lo)])
        hi = np.nanmax([np.nanmax(y_obs), np.nanmax(y_hi)])
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        lo2, hi2 = lo - pad, hi + pad
        if x_range is None:
            x_range = (lo2, hi2)
        if y_range is None:
            y_range = (lo2, hi2)

    fig = go.Figure()

    # points with vertical uncertainty bars
    fig.add_trace(
        go.Scatter(
            x=y_obs,
            y=y_center,
            mode="markers",
            marker=dict(
                color=marker_color,
                symbol=marker_symbol,
                size=marker_size,
                opacity=marker_opacity,
                line=dict(width=0),  # clean look
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                width=err_width,
                thickness=err_thickness,
                color="black",
            ),
            name=f"{center_name} prediction",
            hovertemplate=(
                f"{x_label}: %{{x}}<br>"
                f"{y_label} ({center_name.lower()}): %{{y}}<br>"
                f"{int(round(level*100))}% interval: [%{{customdata[0]}}, %{{customdata[1]}}]"
                "<extra></extra>"
            ),
            customdata=np.column_stack([y_lo, y_hi]),
        )
    )

    # diagonal y=x
    if show_diagonal:
        lo = min(x_range[0], y_range[0])
        hi = max(x_range[1], y_range[1])
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(
                    width=diagonal_width, dash=diagonal_dash, color=diagonal_color
                ),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # layout: same style as your scatter function
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=25, t=60 if title else 25, b=60),
        showlegend=False,
    )

    axis_common = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,  # boxed
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )

    x_update = dict(
        title_text=x_label,
        title_font=dict(size=x_label_font_size),
        tickfont=dict(size=tick_font_size),
        range=list(x_range),
        **axis_common,
    )
    if x_tick0 is not None:
        x_update["tick0"] = x_tick0
    if x_dtick is not None:
        x_update["dtick"] = x_dtick

    y_update = dict(
        title_text=y_label,
        title_font=dict(size=y_label_font_size),
        tickfont=dict(size=tick_font_size),
        range=list(y_range),
        **axis_common,
    )
    if y_tick0 is not None:
        y_update["tick0"] = y_tick0
    if y_dtick is not None:
        y_update["dtick"] = y_dtick

    fig.update_xaxes(**x_update)
    fig.update_yaxes(**y_update)

    return fig
