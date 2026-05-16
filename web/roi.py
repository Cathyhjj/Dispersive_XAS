"""Web-based rectangular ROI selection with rotation and live spectrum preview."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.roi import (
    make_tilted_band_roi,
    normalize_roi_spec,
    roi_boundary_rows,
    roi_weighted_column_mean,
    save_roi_json,
    tilted_band_controls_from_roi,
)
from .plotting import show_image, show_mask_overlay

__all__ = [
    "show_roi",
    "PgSpec",
    "TiltedBandROIEditor",
    "select_rect_roi",
    "select_tilted_band_roi",
]


@dataclass
class _RotRectROI:
    cx: float
    cy: float
    width: float
    height: float
    angle: float = 0.0


def _default_roi(shape: tuple[int, int], frac: float = 0.4) -> _RotRectROI:
    h, w = shape
    return _RotRectROI(
        cx=w / 2.0,
        cy=h / 2.0,
        width=max(8.0, w * frac),
        height=max(8.0, h * frac),
        angle=0.0,
    )


def _roi_vertices(roi: _RotRectROI) -> np.ndarray:
    hw = roi.width / 2.0
    hh = roi.height / 2.0
    corners = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
        dtype=float,
    )
    theta = np.deg2rad(float(roi.angle))
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    pts = corners @ rot.T
    pts[:, 0] += roi.cx
    pts[:, 1] += roi.cy
    return pts


def _polygon_mask(shape: tuple[int, int], vertices: np.ndarray) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    xv = vertices[:, 0]
    yv = vertices[:, 1]

    inside = np.zeros((h, w), dtype=bool)
    j = len(xv) - 1
    for i in range(len(xv)):
        xi, yi = xv[i], yv[i]
        xj, yj = xv[j], yv[j]
        cross = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cross
        j = i
    return inside


def _spectrum_from_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.ma.array(img, mask=~mask)
    spec = np.ma.mean(masked, axis=0)
    if isinstance(spec, np.ma.MaskedArray):
        spec = spec.filled(np.nan)
    return np.asarray(spec, dtype=float)


def _shape_to_bounds(shape_obj) -> Optional[tuple[float, float, float, float]]:
    if hasattr(shape_obj, "to_plotly_json"):
        st = shape_obj.to_plotly_json()
    elif isinstance(shape_obj, dict):
        st = shape_obj
    else:
        try:
            st = dict(shape_obj)
        except Exception:
            return None
    if st.get("type") != "rect":
        return None
    x0, x1 = sorted((float(st["x0"]), float(st["x1"])))
    y0, y1 = sorted((float(st["y0"]), float(st["y1"])))
    return x0, y0, x1, y1


def _shape_to_path_points(shape_obj) -> Optional[np.ndarray]:
    if hasattr(shape_obj, "to_plotly_json"):
        st = shape_obj.to_plotly_json()
    elif isinstance(shape_obj, dict):
        st = shape_obj
    else:
        try:
            st = dict(shape_obj)
        except Exception:
            return None
    if st.get("type") != "path":
        return None
    path = st.get("path")
    if not isinstance(path, str):
        return None
    nums = [
        float(v)
        for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path)
    ]
    if len(nums) < 8 or (len(nums) % 2) != 0:
        return None
    pts = np.asarray(nums, dtype=float).reshape(-1, 2)
    if pts.shape[0] > 4 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if pts.shape[0] < 4:
        return None
    return pts[:4]


def _display_image_and_limits(
    img: np.ndarray, q_low: float = 1.0, q_high: float = 99.0
) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(img, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        disp = np.zeros_like(arr, dtype=float)
        return disp, 0.0, 1.0

    lo, hi = np.nanpercentile(finite, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        center = float(finite[0])
        span = max(1.0, abs(center) * 1e-3)
        lo = center - span
        hi = center + span

    disp = np.array(arr, copy=True, dtype=float)
    disp[~np.isfinite(disp)] = lo
    return disp, float(lo), float(hi)


def _clip_bounds(
    bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    h, w = shape
    max_x = max(1.0, float(w - 1))
    max_y = max(1.0, float(h - 1))
    x0, y0, x1, y1 = bounds
    x0 = float(np.clip(x0, 0.0, max_x))
    x1 = float(np.clip(x1, 0.0, max_x))
    y0 = float(np.clip(y0, 0.0, max_y))
    y1 = float(np.clip(y1, 0.0, max_y))
    if x1 <= x0:
        x1 = min(max_x, x0 + 1.0)
    if y1 <= y0:
        y1 = min(max_y, y0 + 1.0)
    return x0, y0, x1, y1


def _roi_from_vertices(vertices: np.ndarray, fallback_angle: float = 0.0) -> _RotRectROI:
    pts = np.asarray(vertices, dtype=float)
    if pts.shape[0] < 4:
        x0 = float(np.min(pts[:, 0]))
        x1 = float(np.max(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        y1 = float(np.max(pts[:, 1]))
        return _RotRectROI(
            cx=(x0 + x1) / 2.0,
            cy=(y0 + y1) / 2.0,
            width=max(1.0, x1 - x0),
            height=max(1.0, y1 - y0),
            angle=float(fallback_angle),
        )

    p0, p1, p2 = pts[0], pts[1], pts[2]
    edge_w = p1 - p0
    edge_h = p2 - p1
    width = float(np.linalg.norm(edge_w))
    height = float(np.linalg.norm(edge_h))
    if width < 1e-6 or height < 1e-6:
        x0 = float(np.min(pts[:, 0]))
        x1 = float(np.max(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        y1 = float(np.max(pts[:, 1]))
        return _RotRectROI(
            cx=(x0 + x1) / 2.0,
            cy=(y0 + y1) / 2.0,
            width=max(1.0, x1 - x0),
            height=max(1.0, y1 - y0),
            angle=float(fallback_angle),
        )

    angle = float(np.rad2deg(np.arctan2(edge_w[1], edge_w[0])))
    center = np.mean(pts[:4], axis=0)
    return _RotRectROI(
        cx=float(center[0]),
        cy=float(center[1]),
        width=max(1.0, width),
        height=max(1.0, height),
        angle=angle,
    )


def show_roi(
    img: np.ndarray,
    m: np.ndarray,
    viewer: str = "web",
    alpha: float = 0.1,
    **kwargs,
) -> None:
    """Overlay a boolean mask on an image with web plotting."""
    if viewer != "web":
        raise ValueError("Only viewer='web' is supported in this package.")
    show_mask_overlay(img, m, title=kwargs.get("title", "ROI overlay"), show=True)


class PgSpec:
    """Rectangle ROI manager with rotation support and live spectrum preview."""

    def __init__(self, data: np.ndarray, title: str = "", **kwargs):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("PgSpec expects a 2-D image.")
        self.data = data
        self.title = title or "DXAS Rectangular ROI Selector"
        self.rects: dict[str, _RotRectROI] = {}

        self._selector_figure = None
        self._selector_widget = None
        self._selector_name: Optional[str] = None
        self._rotation_slider = None
        self._width_slider = None
        self._height_slider = None
        self._selector_updating = False
        self._edits_callback = None
        self._relayout_callback = None
        self._layout_shapes_callback = None
        self._rotation_callback = None
        self._width_callback = None
        self._height_callback = None

    def _sanitize_roi(self, roi: _RotRectROI) -> _RotRectROI:
        h, w = self.data.shape
        roi.cx = float(np.clip(roi.cx, 0.0, max(0.0, float(w - 1))))
        roi.cy = float(np.clip(roi.cy, 0.0, max(0.0, float(h - 1))))
        roi.width = float(np.clip(roi.width, 1.0, max(1.0, float(w))))
        roi.height = float(np.clip(roi.height, 1.0, max(1.0, float(h))))
        roi.angle = float(roi.angle)
        return roi

    def _shape_path(self, roi_name: str) -> str:
        pts = _roi_vertices(self.rects[roi_name])
        return "M " + " L ".join(f"{p[0]},{p[1]}" for p in pts) + " Z"

    def _rot_shape(self, roi_name: str) -> dict:
        return dict(
            type="path",
            path=self._shape_path(roi_name),
            line=dict(color="#d62728", width=2),
            fillcolor="rgba(214,39,40,0.15)",
            editable=True,
            layer="above",
        )

    def _sync_roi_from_selector(self, name: str) -> bool:
        if self._selector_figure is None:
            return False
        if name not in self.rects:
            self.set_rect(name=name)

        shapes = list(self._selector_figure.layout.shapes or [])
        current = self.rects[name]
        updated: Optional[_RotRectROI] = None
        for shp in reversed(shapes):
            pts = _shape_to_path_points(shp)
            if pts is not None:
                updated = _roi_from_vertices(pts, fallback_angle=current.angle)
                break
            bounds = _shape_to_bounds(shp)
            if bounds is not None:
                x0, y0, x1, y1 = _clip_bounds(bounds, self.data.shape)
                updated = _RotRectROI(
                    cx=(x0 + x1) / 2.0,
                    cy=(y0 + y1) / 2.0,
                    width=max(1.0, x1 - x0),
                    height=max(1.0, y1 - y0),
                    angle=current.angle,
                )
                break

        if updated is None:
            return False
        updated = self._sanitize_roi(updated)

        changed = any(
            abs(a - b) > 1e-6
            for a, b in (
                (current.cx, updated.cx),
                (current.cy, updated.cy),
                (current.width, updated.width),
                (current.height, updated.height),
                (current.angle, updated.angle),
            )
        )
        self.rects[name] = updated
        return changed

    def _sync_roi_from_relayout_payload(self, name: str, payload: dict) -> bool:
        if name not in self.rects:
            self.set_rect(name=name)
        current = self.rects[name]
        updated: Optional[_RotRectROI] = None

        # Path edits (single-shape ROI): shapes[i].path
        path_val = None
        for k, v in payload.items():
            if isinstance(k, str) and (
                (k.startswith("shapes[") and k.endswith("].path"))
                or (k.startswith("layout.shapes[") and k.endswith("].path"))
            ):
                path_val = v
                break
            if isinstance(k, str) and (k.startswith("shapes[") or k.startswith("layout.shapes[")):
                if isinstance(v, dict) and isinstance(v.get("path"), str):
                    path_val = v["path"]
                    break
        if isinstance(path_val, str):
            pts = _shape_to_path_points({"type": "path", "path": path_val})
            if pts is not None:
                updated = _roi_from_vertices(pts, fallback_angle=current.angle)

        # Full shapes-array relayout payload
        if updated is None and isinstance(payload.get("shapes"), (list, tuple)):
            for shp in reversed(payload["shapes"]):
                pts = _shape_to_path_points(shp)
                if pts is not None:
                    updated = _roi_from_vertices(pts, fallback_angle=current.angle)
                    break
                bounds = _shape_to_bounds(shp)
                if bounds is not None:
                    x0, y0, x1, y1 = _clip_bounds(bounds, self.data.shape)
                    updated = _RotRectROI(
                        cx=(x0 + x1) / 2.0,
                        cy=(y0 + y1) / 2.0,
                        width=max(1.0, x1 - x0),
                        height=max(1.0, y1 - y0),
                        angle=current.angle,
                    )
                    break

        # Fallback for rect-like payloads
        if updated is None:
            vals: dict[str, float] = {}
            idx = None
            for k, v in payload.items():
                if not isinstance(k, str):
                    continue
                m = re.match(r"(?:layout\.)?shapes\[(\d+)\]\.(x0|x1|y0|y1)$", k)
                if not m:
                    continue
                if idx is None:
                    idx = m.group(1)
                if m.group(1) != idx:
                    continue
                vals[m.group(2)] = float(v)
            if len(vals) == 4:
                x0, y0, x1, y1 = _clip_bounds(
                    (vals["x0"], vals["y0"], vals["x1"], vals["y1"]),
                    self.data.shape,
                )
                updated = _RotRectROI(
                    cx=(x0 + x1) / 2.0,
                    cy=(y0 + y1) / 2.0,
                    width=max(1.0, x1 - x0),
                    height=max(1.0, y1 - y0),
                    angle=current.angle,
                )

        if updated is None:
            return False
        updated = self._sanitize_roi(updated)
        changed = any(
            abs(a - b) > 1e-6
            for a, b in (
                (current.cx, updated.cx),
                (current.cy, updated.cy),
                (current.width, updated.width),
                (current.height, updated.height),
                (current.angle, updated.angle),
            )
        )
        self.rects[name] = updated
        return changed

    def _extract_relayout_payload(self, raw) -> dict:
        if not isinstance(raw, dict):
            return {}
        # FigureWidget trait packs relayout payload in this wrapper.
        if isinstance(raw.get("relayout_data"), dict):
            return raw["relayout_data"]
        # Layout delta wrapper from widget internals.
        if isinstance(raw.get("layout_delta"), dict):
            return raw["layout_delta"]
        return raw

    def _sync_control_values(self, name: str) -> None:
        if name not in self.rects:
            return
        r = self.rects[name]
        if self._rotation_slider is not None:
            v = float(np.clip(r.angle, self._rotation_slider.min, self._rotation_slider.max))
            if abs(float(self._rotation_slider.value) - v) > 1e-9:
                self._rotation_slider.value = v
        if self._width_slider is not None:
            v = float(np.clip(r.width, self._width_slider.min, self._width_slider.max))
            if abs(float(self._width_slider.value) - v) > 1e-9:
                self._width_slider.value = v
        if self._height_slider is not None:
            v = float(np.clip(r.height, self._height_slider.min, self._height_slider.max))
            if abs(float(self._height_slider.value) - v) > 1e-9:
                self._height_slider.value = v

    def _update_selector_plot(self, name: str, reset_shape: bool = True) -> None:
        if self._selector_figure is None:
            return
        if name not in self.rects:
            self.set_rect(name=name)

        mask = self._mask_from_roi(name)
        spec = _spectrum_from_mask(self.data, mask)
        x_axis = np.arange(self.data.shape[1], dtype=float)
        rotated = self._rot_shape(name)

        self._selector_updating = True
        with self._selector_figure.batch_update():
            if reset_shape:
                self._selector_figure.layout.shapes = [rotated]
            self._selector_figure.layout.title = (
                f"{self.title} | angle={self.rects[name].angle:.2f} deg"
            )
            self._selector_figure.data[1].x = x_axis
            self._selector_figure.data[1].y = spec
            self._sync_control_values(name=name)
        self._selector_updating = False

    def _on_selector_edited(self, name: str) -> None:
        if self._selector_updating:
            return
        if self._selector_name != name:
            return
        changed = self._sync_roi_from_selector(name=name)
        if changed:
            self._update_selector_plot(name=name, reset_shape=True)

    def _on_selector_relayout(self, name: str, change: dict) -> None:
        if self._selector_updating:
            return
        if self._selector_name != name:
            return
        payload = self._extract_relayout_payload(change.get("new"))
        if not payload:
            return
        changed = self._sync_roi_from_relayout_payload(name=name, payload=payload)
        if not changed:
            changed = self._sync_roi_from_selector(name=name)
        if not changed:
            return
        # Keep user drag motion smooth: do not overwrite shape while dragging.
        self._update_selector_plot(name=name, reset_shape=False)

    def _on_selector_shapes_change(self, name: str) -> None:
        if self._selector_updating:
            return
        if self._selector_name != name:
            return
        changed = self._sync_roi_from_selector(name=name)
        if not changed:
            return
        self._update_selector_plot(name=name, reset_shape=False)

    def _on_rotation_change(self, name: str, change: dict) -> None:
        if self._selector_updating:
            return
        if change.get("name") != "value":
            return
        if change.get("new") is None:
            return
        if name not in self.rects:
            self.set_rect(name=name)
        self.rects[name].angle = float(change["new"])
        self._update_selector_plot(name=name, reset_shape=True)

    def _on_size_change(self, name: str, which: str, change: dict) -> None:
        if self._selector_updating:
            return
        if change.get("name") != "value":
            return
        if change.get("new") is None:
            return
        if name not in self.rects:
            self.set_rect(name=name)
        val = max(1.0, float(change["new"]))
        if which == "width":
            self.rects[name].width = val
        elif which == "height":
            self.rects[name].height = val
        self.rects[name] = self._sanitize_roi(self.rects[name])
        self._update_selector_plot(name=name, reset_shape=True)

    def set_rect(
        self,
        name: str = "main",
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        angle: float = 0.0,
    ) -> None:
        """Define/update a rectangular ROI."""
        if None in (cx, cy, width, height):
            roi = _default_roi(self.data.shape)
            roi.angle = float(angle)
            self.rects[name] = self._sanitize_roi(roi)
        else:
            self.rects[name] = self._sanitize_roi(
                _RotRectROI(
                    cx=float(cx),
                    cy=float(cy),
                    width=float(width),
                    height=float(height),
                    angle=float(angle),
                )
            )
        if self._selector_name == name and self._selector_figure is not None:
            self._update_selector_plot(name=name, reset_shape=True)

    def set_rotation(self, name: str = "main", angle: float = 0.0) -> None:
        """Set rotation angle (degrees) for a ROI."""
        if name not in self.rects:
            self.set_rect(name=name)
        self.rects[name].angle = float(angle)
        slider_driven = False
        if self._rotation_slider is not None and self._selector_name == name:
            if abs(float(self._rotation_slider.value) - float(angle)) > 1e-9:
                slider_driven = True
                self._rotation_slider.value = float(angle)
        if self._selector_name == name and self._selector_figure is not None and not slider_driven:
            self._update_selector_plot(name=name, reset_shape=True)

    def rotate(self, name: str = "main", delta: float = 0.0) -> None:
        """Increment rotation angle by `delta` degrees."""
        if name not in self.rects:
            self.set_rect(name=name)
        self.set_rotation(name=name, angle=self.rects[name].angle + float(delta))

    def getAngle(self, roi_name: str = "main") -> float:
        """Return rotation angle in degrees."""
        if roi_name not in self.rects:
            self.set_rect(name=roi_name)
        angle = float(self.rects[roi_name].angle)
        print(f"ROI '{roi_name}' angle: {angle:.2f} deg")
        return angle

    def launch_selector(self, name: str = "main", show: bool = True):
        """Open interactive rectangular ROI selector with live spectrum update."""
        if name not in self.rects:
            self.set_rect(name=name)
        self._selector_name = name

        try:
            import ipywidgets as widgets
            from IPython.display import display
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception:
            raise ImportError(
                "PgSpec.launch_selector() requires ipywidgets and anywidget for "
                "live ROI syncing. Install with: pip install ipywidgets anywidget"
            )

        x_axis = np.arange(self.data.shape[1], dtype=float)
        mask = self._mask_from_roi(name)
        spec = _spectrum_from_mask(self.data, mask)
        disp, zmin, zmax = _display_image_and_limits(self.data)

        base_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            row_heights=[0.68, 0.32],
            subplot_titles=("ROI Selector (single interactive ROI)", "ROI Vertical Average Spectrum"),
        )
        base_fig.add_trace(
            go.Heatmap(
                z=disp,
                zmin=zmin,
                zmax=zmax,
                colorscale="magma",
                colorbar=dict(title="Intensity"),
            ),
            row=1,
            col=1,
        )
        base_fig.add_trace(
            go.Scatter(x=x_axis, y=spec, mode="lines", line=dict(width=2), name="ROI avg"),
            row=2,
            col=1,
        )
        base_fig.update_layout(
            template="plotly_white",
            title=f"{self.title} | angle={self.rects[name].angle:.2f} deg",
            height=760,
            dragmode="zoom",
            uirevision="roi-selector",
            shapes=[self._rot_shape(name)],
        )
        base_fig.update_xaxes(title="Column", row=1, col=1)
        base_fig.update_yaxes(
            title="Row",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1,
        )
        base_fig.update_xaxes(title="Column", row=2, col=1)
        base_fig.update_yaxes(title="Mean Intensity", row=2, col=1)

        fig = go.FigureWidget(base_fig)
        fig._config = {
            "editable": True,
            "edits": {"shapePosition": True},
            "displaylogo": False,
        }
        slider_angle = widgets.FloatSlider(
            value=float(self.rects[name].angle),
            min=-180.0,
            max=180.0,
            step=0.5,
            description="Angle (deg):",
            continuous_update=True,
            readout_format=".1f",
            layout=widgets.Layout(width="33%"),
        )
        slider_width = widgets.FloatSlider(
            value=float(self.rects[name].width),
            min=1.0,
            max=float(self.data.shape[1]),
            step=1.0,
            description="Width:",
            continuous_update=True,
            readout_format=".0f",
            layout=widgets.Layout(width="33%"),
        )
        slider_height = widgets.FloatSlider(
            value=float(self.rects[name].height),
            min=1.0,
            max=float(self.data.shape[0]),
            step=1.0,
            description="Height:",
            continuous_update=True,
            readout_format=".0f",
            layout=widgets.Layout(width="33%"),
        )
        slider_help = widgets.HTML(
            value=(
                "Single ROI workflow: drag the red ROI to move, use Angle/Width/Height "
                "controls to rotate and resize. The spectrum updates live."
            )
        )
        controls = widgets.HBox([slider_angle, slider_width, slider_height])
        ui = widgets.VBox([fig, controls, slider_help])

        self._selector_figure = fig
        self._selector_widget = ui
        self._rotation_slider = slider_angle
        self._width_slider = slider_width
        self._height_slider = slider_height

        def _edits_cb() -> None:
            self._on_selector_edited(name=name)

        def _angle_cb(change: dict) -> None:
            self._on_rotation_change(name=name, change=change)

        self._edits_callback = _edits_cb
        self._rotation_callback = _angle_cb
        self._width_callback = lambda change: self._on_size_change(
            name=name, which="width", change=change
        )
        self._height_callback = lambda change: self._on_size_change(
            name=name, which="height", change=change
        )
        self._relayout_callback = lambda change: self._on_selector_relayout(
            name=name, change=change
        )
        self._layout_shapes_callback = (
            lambda layout, shapes: self._on_selector_shapes_change(name=name)
        )
        fig.on_edits_completed(self._edits_callback)
        fig.observe(self._relayout_callback, names="_js2py_relayout")
        fig.layout.on_change(self._layout_shapes_callback, "shapes")
        slider_angle.observe(self._rotation_callback, names="value")
        slider_width.observe(self._width_callback, names="value")
        slider_height.observe(self._height_callback, names="value")

        self._update_selector_plot(name=name)

        if show:
            display(ui)
            return self
        return ui

    def sync_from_selector(self, name: str = "main") -> _RotRectROI:
        """Sync center/size from the current selector while preserving angle."""
        if name not in self.rects:
            self.set_rect(name=name)
        if self._selector_figure is None:
            return self.rects[name]
        self._sync_roi_from_selector(name=name)
        return self.rects[name]

    def _mask_from_roi(self, roi_name: str = "main") -> np.ndarray:
        if roi_name not in self.rects:
            self.set_rect(name=roi_name)
        vertices = _roi_vertices(self.rects[roi_name])
        return _polygon_mask(self.data.shape, vertices).astype(bool)

    def show(self, name: str = "main"):
        """Show image with rotated ROI and ROI vertical-average spectrum below."""
        go, pio = __import__("plotly.graph_objects", fromlist=["go"]), __import__(
            "plotly.io", fromlist=["pio"]
        )
        from plotly.subplots import make_subplots

        if name not in self.rects:
            self.set_rect(name=name)
        self.sync_from_selector(name=name)
        mask = self._mask_from_roi(name)
        spec = _spectrum_from_mask(self.data, mask)
        x_axis = np.arange(self.data.shape[1], dtype=float)
        disp, zmin, zmax = _display_image_and_limits(self.data)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            row_heights=[0.68, 0.32],
            subplot_titles=("Rotated ROI", "ROI Vertical Average Spectrum"),
        )
        fig.add_trace(
            go.Heatmap(
                z=disp,
                zmin=zmin,
                zmax=zmax,
                colorscale="magma",
                colorbar=dict(title="Intensity"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_axis, y=spec, mode="lines", line=dict(width=2), name="ROI avg"),
            row=2,
            col=1,
        )
        fig.update_layout(
            template="plotly_white",
            title=f"{self.title} | angle={self.rects[name].angle:.2f} deg",
            height=760,
            shapes=[self._rot_shape(name)],
        )
        fig.update_xaxes(title="Column", row=1, col=1)
        fig.update_yaxes(
            title="Row",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1,
        )
        fig.update_xaxes(title="Column", row=2, col=1)
        fig.update_yaxes(title="Mean Intensity", row=2, col=1)
        pio.show(fig)
        return fig

    def getMask(self, roi_name: str = "main", show: bool = True) -> np.ndarray:
        """Return boolean mask for named rotated rectangle ROI."""
        self.sync_from_selector(name=roi_name)
        mask = self._mask_from_roi(roi_name)
        if show:
            self.show(roi_name)
        return mask

    def getMaskAndAngle(self, roi_name: str = "main", show: bool = True):
        """Return tuple: (mask, angle_deg)."""
        mask = self.getMask(roi_name, show=show)
        angle = self.getAngle(roi_name)
        return mask, angle

    def getAllMasks(self, show: bool = False, save: bool = False) -> dict:
        """Return masks for all current ROIs."""
        if not self.rects:
            self.set_rect(name="main")
        return {name: self.getMask(name, show=show) for name in self.rects}

    def getArrayRegion(self, roi_name: str = "main", show: bool = True) -> np.ndarray:
        """Return bounding sub-array containing ROI mask."""
        mask = self._mask_from_roi(roi_name)
        coords = np.argwhere(mask)
        if coords.size == 0:
            region = np.zeros((1, 1), dtype=float)
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            region = self.data[y0:y1, x0:x1]
            region = np.where(mask[y0:y1, x0:x1], region, 0.0)
        if show:
            show_image(region, title=f"ROI region: {roi_name}", show=True)
        return region


def select_rect_roi(
    img: np.ndarray,
    name: str = "main",
    show_selector: bool = True,
) -> PgSpec:
    """Convenience helper: create selector and open rectangular ROI editor."""
    roi = PgSpec(img)
    roi.launch_selector(name=name, show=show_selector)
    return roi


class TiltedBandROIEditor:
    """Interactive tilted-band ROI editor for a representative 2-D mux image."""

    def __init__(
        self,
        data: np.ndarray,
        initial_roi: Optional[dict] = None,
        title: str = "",
        save_path: str | Path | None = None,
    ):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("TiltedBandROIEditor expects a 2-D image.")

        self.data = data
        self.title = title or "DXAS Tilted-Band ROI Editor"
        self.save_path = None if save_path is None else Path(save_path).expanduser().resolve()
        self.roi = self._normalize_initial_roi(initial_roi=initial_roi)

        self._figure = None
        self._widget = None
        self._updating = False
        self._plot_backend = None
        self._plot_output = None
        self._center_slider = None
        self._left_slider = None
        self._right_slider = None
        self._half_width_slider = None
        self._summary_html = None
        self._path_text = None
        self._save_button = None
        self._save_status = None

    def _normalize_initial_roi(self, initial_roi: Optional[dict]) -> dict[str, object]:
        h, _w = self.data.shape
        if initial_roi is None:
            center = 0.5 * max(0.0, float(h - 1))
            return make_tilted_band_roi(
                self.data.shape,
                left_center_row=center,
                right_center_row=center,
                half_width=max(4.0, float(h) * 0.06),
            )
        return normalize_roi_spec(self.data.shape, roi=initial_roi)

    def get_spec(self) -> dict[str, object]:
        """Return the current tilted-band ROI spec."""
        if self._left_slider is not None:
            self._sync_roi_from_controls()
        return normalize_roi_spec(self.data.shape, roi=self.roi)

    def save(
        self,
        path: str | Path | None = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        """Save the current ROI spec to JSON."""
        if path is None and self.save_path is None:
            raise ValueError("No ROI JSON path was provided.")
        out_path = Path(path or self.save_path).expanduser()
        payload_meta = {"shape": [int(self.data.shape[0]), int(self.data.shape[1])]}
        if metadata:
            payload_meta.update(metadata)
        return save_roi_json(out_path, self.get_spec(), metadata=payload_meta)

    def _sync_roi_from_controls(self) -> None:
        self.roi = make_tilted_band_roi(
            self.data.shape,
            left_center_row=float(self._left_slider.value),
            right_center_row=float(self._right_slider.value),
            half_width=float(self._half_width_slider.value),
        )

    def _summary_text(self) -> str:
        left, right, half_width = tilted_band_controls_from_roi(self.data.shape, roi=self.roi)
        center = 0.5 * (left + right)
        slope = float(self.roi.get("slope_per_col", 0.0))
        bounds = self.roi.get("row_bounds", [])
        return (
            f"<b>ROI</b>: center={center:.1f}, left={left:.1f}, right={right:.1f}, "
            f"half_width={half_width:.1f}, slope={slope:.4f}, "
            f"row_bounds={list(bounds)}"
        )

    def _sync_center_slider_value(self) -> None:
        if self._center_slider is None or self._left_slider is None or self._right_slider is None:
            return
        center = 0.5 * (float(self._left_slider.value) + float(self._right_slider.value))
        center = float(np.clip(center, self._center_slider.min, self._center_slider.max))
        if abs(float(self._center_slider.value) - center) > 1e-9:
            self._center_slider.value = center

    def _update_plot(self) -> None:
        if self._left_slider is None:
            return

        self._sync_roi_from_controls()
        cols, top, bottom = roi_boundary_rows(self.data.shape, roi=self.roi)
        spec = roi_weighted_column_mean(self.data, roi=self.roi)

        if self._summary_html is not None:
            self._summary_html.value = self._summary_text()

        self._updating = True
        self._sync_center_slider_value()
        self._updating = False

        if self._plot_backend == "plotly-output":
            self._render_plotly_output(cols=cols, top=top, bottom=bottom, spec=spec)
            return
        if self._figure is None:
            return

        self._updating = True
        with self._figure.batch_update():
            self._figure.layout.title = (
                f"{self.title} | slope={float(self.roi.get('slope_per_col', 0.0)):.4f}"
            )
            self._figure.data[1].x = cols
            self._figure.data[1].y = top
            self._figure.data[2].x = cols
            self._figure.data[2].y = bottom
            self._figure.data[3].x = np.arange(self.data.shape[1], dtype=float)
            self._figure.data[3].y = spec
        self._updating = False

    def _render_plotly_output(
        self,
        cols: np.ndarray,
        top: np.ndarray,
        bottom: np.ndarray,
        spec: np.ndarray,
    ) -> None:
        if self._plot_output is None:
            return
        import plotly.graph_objects as go
        from IPython.display import display
        from plotly.subplots import make_subplots

        disp, zmin, zmax = _display_image_and_limits(self.data)
        x_axis = np.arange(self.data.shape[1], dtype=float)
        self._updating = True
        with self._plot_output:
            self._plot_output.clear_output(wait=True)
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.1,
                row_heights=[0.68, 0.32],
                subplot_titles=("Tilted ROI Preview", "ROI Vertical Average Spectrum"),
            )
            fig.add_trace(
                go.Heatmap(
                    z=disp,
                    zmin=zmin,
                    zmax=zmax,
                    colorscale="magma",
                    colorbar=dict(title="mu"),
                    showscale=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=cols,
                    y=top,
                    mode="lines",
                    line=dict(color="#00e5ff", width=2),
                    name="ROI top",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=cols,
                    y=bottom,
                    mode="lines",
                    line=dict(color="#00e5ff", width=2),
                    fill="tonexty",
                    fillcolor="rgba(0,229,255,0.18)",
                    name="ROI bottom",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=spec,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    name="ROI avg",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.update_layout(
                template="plotly_white",
                title=f"{self.title} | slope={float(self.roi.get('slope_per_col', 0.0)):.4f}",
                height=760,
                margin=dict(t=70, l=60, r=30, b=50),
            )
            fig.update_xaxes(title="Column", row=1, col=1)
            fig.update_yaxes(
                title="Row",
                autorange="reversed",
                scaleanchor="x",
                scaleratio=1,
                row=1,
                col=1,
            )
            fig.update_xaxes(title="Column", row=2, col=1)
            fig.update_yaxes(title="Mean mu", row=2, col=1)
            display(fig)
        self._updating = False

    def _on_slider_change(self, change: dict) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._update_plot()

    def _on_center_change(self, change: dict) -> None:
        if self._updating or change.get("name") != "value":
            return
        if self._center_slider is None or self._left_slider is None or self._right_slider is None:
            return

        old_center = 0.5 * (float(self._left_slider.value) + float(self._right_slider.value))
        new_center = float(change["new"])
        delta = new_center - old_center
        if abs(delta) <= 1e-9:
            return

        left = float(self._left_slider.value) + delta
        right = float(self._right_slider.value) + delta
        max_row = float(self._center_slider.max)

        if left < 0.0:
            right -= left
            left = 0.0
        if right > max_row:
            left -= right - max_row
            right = max_row
        left = max(0.0, left)
        right = min(max_row, right)

        self._updating = True
        self._left_slider.value = left
        self._right_slider.value = right
        self._updating = False
        self._update_plot()

    def _on_save_clicked(self, _button) -> None:
        if self._save_status is None:
            return
        try:
            path_text = None if self._path_text is None else self._path_text.value.strip()
            saved = self.save(path=path_text or None)
            self._save_status.value = f"Saved ROI JSON: <code>{saved}</code>"
        except Exception as exc:  # pragma: no cover - notebook/UI path
            self._save_status.value = f"<span style='color:#b22222'>Save failed: {exc}</span>"

    def launch(self, show: bool = True):
        """Create and optionally display the interactive editor."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except Exception:
            raise ImportError(
                "TiltedBandROIEditor.launch() requires ipywidgets."
            )

        left, right, half_width = tilted_band_controls_from_roi(self.data.shape, roi=self.roi)
        center = 0.5 * (left + right)
        max_row = max(1.0, float(self.data.shape[0] - 1))

        slider_layout = widgets.Layout(width="25%")
        center_slider = widgets.FloatSlider(
            value=float(center),
            min=0.0,
            max=max_row,
            step=0.5,
            description="Center row:",
            continuous_update=True,
            readout_format=".1f",
            layout=slider_layout,
        )
        left_slider = widgets.FloatSlider(
            value=float(left),
            min=0.0,
            max=max_row,
            step=0.5,
            description="Left row:",
            continuous_update=True,
            readout_format=".1f",
            layout=slider_layout,
        )
        right_slider = widgets.FloatSlider(
            value=float(right),
            min=0.0,
            max=max_row,
            step=0.5,
            description="Right row:",
            continuous_update=True,
            readout_format=".1f",
            layout=slider_layout,
        )
        half_width_slider = widgets.FloatSlider(
            value=float(half_width),
            min=0.5,
            max=max(2.0, float(self.data.shape[0]) / 2.0),
            step=0.5,
            description="Half width:",
            continuous_update=True,
            readout_format=".1f",
            layout=slider_layout,
        )
        summary_html = widgets.HTML(value=self._summary_text())

        path_text = widgets.Text(
            value="" if self.save_path is None else str(self.save_path),
            description="ROI JSON:",
            layout=widgets.Layout(width="75%"),
        )
        save_button = widgets.Button(description="Save ROI JSON", button_style="")
        save_status = widgets.HTML()

        controls = widgets.HBox([center_slider, left_slider, right_slider, half_width_slider])
        save_row = widgets.HBox([path_text, save_button])
        backend_html = None
        help_html = widgets.HTML(
            value=(
                "Use <b>Center row</b> to move the whole ROI up and down. "
                "Use <b>Left row</b> and <b>Right row</b> to change the tilt, and "
                "<b>Half width</b> to shrink or widen the band."
            )
        )

        fig = None
        plot_widget = None
        self._plot_output = None
        self._plot_backend = None
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            disp, zmin, zmax = _display_image_and_limits(self.data)
            cols, top, bottom = roi_boundary_rows(self.data.shape, roi=self.roi)
            spec = roi_weighted_column_mean(self.data, roi=self.roi)
            x_axis = np.arange(self.data.shape[1], dtype=float)

            base_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.1,
                row_heights=[0.68, 0.32],
                subplot_titles=("Tilted ROI Preview", "ROI Vertical Average Spectrum"),
            )
            base_fig.add_trace(
                go.Heatmap(
                    z=disp,
                    zmin=zmin,
                    zmax=zmax,
                    colorscale="magma",
                    colorbar=dict(title="mu"),
                    showscale=True,
                ),
                row=1,
                col=1,
            )
            base_fig.add_trace(
                go.Scatter(
                    x=cols,
                    y=top,
                    mode="lines",
                    line=dict(color="#00e5ff", width=2),
                    name="ROI top",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            base_fig.add_trace(
                go.Scatter(
                    x=cols,
                    y=bottom,
                    mode="lines",
                    line=dict(color="#00e5ff", width=2),
                    fill="tonexty",
                    fillcolor="rgba(0,229,255,0.18)",
                    name="ROI bottom",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            base_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=spec,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    name="ROI avg",
                ),
                row=2,
                col=1,
            )
            base_fig.update_layout(
                template="plotly_white",
                title=f"{self.title} | slope={float(self.roi.get('slope_per_col', 0.0)):.4f}",
                height=760,
                showlegend=False,
                margin=dict(t=70, l=60, r=30, b=50),
            )
            base_fig.update_xaxes(title="Column", row=1, col=1)
            base_fig.update_yaxes(
                title="Row",
                autorange="reversed",
                scaleanchor="x",
                scaleratio=1,
                row=1,
                col=1,
            )
            base_fig.update_xaxes(title="Column", row=2, col=1)
            base_fig.update_yaxes(title="Mean mu", row=2, col=1)
            fig = go.FigureWidget(base_fig)
            plot_widget = fig
            self._plot_backend = "plotly"
        except Exception:
            plot_widget = widgets.Output()
            self._plot_output = plot_widget
            self._plot_backend = "plotly-output"
            backend_html = widgets.HTML(
                value=(
                    "Plotly FigureWidget is unavailable in this kernel. "
                    "Using a live Plotly preview instead."
                )
            )

        items = [plot_widget, controls, summary_html, save_row, save_status]
        if backend_html is not None:
            items.append(backend_html)
        items.append(help_html)
        ui = widgets.VBox(items)

        self._figure = fig
        self._widget = ui
        self._center_slider = center_slider
        self._left_slider = left_slider
        self._right_slider = right_slider
        self._half_width_slider = half_width_slider
        self._summary_html = summary_html
        self._path_text = path_text
        self._save_button = save_button
        self._save_status = save_status

        center_slider.observe(self._on_center_change, names="value")
        left_slider.observe(self._on_slider_change, names="value")
        right_slider.observe(self._on_slider_change, names="value")
        half_width_slider.observe(self._on_slider_change, names="value")
        save_button.on_click(self._on_save_clicked)

        self._update_plot()

        if show:
            display(ui)
            return self
        return ui


def select_tilted_band_roi(
    img: np.ndarray,
    initial_roi: Optional[dict] = None,
    title: str = "",
    save_path: str | Path | None = None,
    show: bool = True,
) -> TiltedBandROIEditor:
    """Convenience helper: create and optionally display a tilted-band ROI editor."""
    editor = TiltedBandROIEditor(
        img,
        initial_roi=initial_roi,
        title=title,
        save_path=save_path,
    )
    editor.launch(show=show)
    return editor
