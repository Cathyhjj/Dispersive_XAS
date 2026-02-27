"""Web-first plotting utilities for Dispersive_XAS.

All helpers in this module use Plotly and render in-browser / notebook,
which keeps plotting separate from numerical core functions.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "show_line",
    "show_lines",
    "show_image",
    "show_image_stack",
    "show_mask_overlay",
]


def _plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotly is required for web-based plotting. Install with: pip install plotly"
        ) from exc
    return go, pio


def _line_style(style_kwargs: Optional[dict]) -> tuple[dict, Optional[float]]:
    style_kwargs = style_kwargs or {}
    line = {}
    opacity = style_kwargs.get("alpha")
    if "color" in style_kwargs:
        line["color"] = style_kwargs["color"]
    width = style_kwargs.get("lw", style_kwargs.get("linewidth"))
    if width is not None:
        line["width"] = float(width)
    return line, opacity


def show_lines(
    traces: Sequence[dict],
    title: str = "",
    x_label: str = "X",
    y_label: str = "Y",
    show: bool = True,
    height: int = 440,
    width: Optional[int] = None,
):
    """Render one or more XY traces with Plotly."""
    go, pio = _plotly()
    fig = go.Figure()

    for tr in traces:
        line, opacity = _line_style(tr.get("style"))
        fig.add_trace(
            go.Scatter(
                x=np.asarray(tr["x"]).ravel(),
                y=np.asarray(tr["y"]).ravel(),
                mode=tr.get("mode", "lines"),
                name=tr.get("name"),
                line=line if line else None,
                marker=tr.get("marker"),
                opacity=opacity,
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if show:
        pio.show(fig)
    return fig


def show_line(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    name: str = "signal",
    show: bool = True,
    x_label: str = "X",
    y_label: str = "Y",
    **style,
):
    """Render a single XY trace with Plotly."""
    return show_lines(
        traces=[{"x": x, "y": y, "name": name, "style": style}],
        title=title,
        x_label=x_label,
        y_label=y_label,
        show=show,
    )


def show_image(
    img: np.ndarray,
    title: str = "",
    show: bool = True,
    colorscale: str = "magma",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
):
    """Render a 2-D image as an interactive Plotly heatmap."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        return show_image_stack(arr, title=title, show=show, colorscale=colorscale)

    go, pio = _plotly()
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=arr,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="Intensity"),
            )
        ]
    )
    fig.update_layout(template="plotly_white", title=title, height=520)
    fig.update_xaxes(title="Column")
    fig.update_yaxes(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1)
    if show:
        pio.show(fig)
    return fig


def show_image_stack(
    stack: np.ndarray,
    title: str = "",
    show: bool = True,
    colorscale: str = "magma",
):
    """Render a 3-D stack with a frame slider."""
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError("stack must have shape (N, H, W)")

    go, pio = _plotly()
    fig = go.Figure(
        data=[go.Heatmap(z=arr[0], colorscale=colorscale, colorbar=dict(title="Intensity"))]
    )

    frames = [
        go.Frame(
            data=[go.Heatmap(z=arr[i], colorscale=colorscale)],
            name=str(i),
            traces=[0],
        )
        for i in range(arr.shape[0])
    ]
    fig.frames = frames

    fig.update_layout(
        template="plotly_white",
        title=f"{title} (frame 0/{arr.shape[0]-1})" if title else f"Frame 0/{arr.shape[0]-1}",
        height=560,
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(arr.shape[0])
                ],
            }
        ],
    )
    fig.update_xaxes(title="Column")
    fig.update_yaxes(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1)
    if show:
        pio.show(fig)
    return fig


def show_mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    show: bool = True,
    colorscale: str = "magma",
    overlay_color: str = "rgba(255, 69, 0, 0.45)",
):
    """Render an image with a boolean mask overlay."""
    go, pio = _plotly()
    arr = np.asarray(img)
    m = np.asarray(mask).astype(bool)
    if arr.shape != m.shape:
        raise ValueError("img and mask must have the same shape")

    overlay = np.where(m, 1.0, np.nan)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=arr,
            colorscale=colorscale,
            colorbar=dict(title="Intensity"),
            showscale=True,
            name="image",
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=overlay,
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, overlay_color]],
            showscale=False,
            hoverinfo="skip",
            name="mask",
        )
    )
    fig.update_layout(template="plotly_white", title=title, height=520)
    fig.update_xaxes(title="Column")
    fig.update_yaxes(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1)
    if show:
        pio.show(fig)
    return fig
