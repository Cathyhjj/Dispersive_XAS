"""Web-based rectangular ROI selection with rotation and live spectrum preview."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

from .plotting import show_image, show_mask_overlay

__all__ = ["show_roi", "PgSpec", "select_rect_roi"]


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
