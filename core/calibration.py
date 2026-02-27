"""Pixel-to-energy calibration for DXAS."""

from __future__ import annotations

import csv
from typing import Optional

import numpy as np

from .utils import date_today, time_now

__all__ = [
    "calibrate_regression",
    "EDXAS_Calibrate",
]


def _plot_calibration(traces, title: str = "Calibration"):
    from ._display import show_lines

    show_lines(
        traces=traces,
        title=title,
        x_label="Energy (eV)",
        y_label="Absorption",
        show=True,
    )


def _fit_poly(train: np.ndarray, target: np.ndarray, order: int) -> np.ndarray:
    train = np.asarray(train, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if train.size != target.size:
        raise ValueError("train and target must have equal length.")
    if train.size < order + 1:
        raise ValueError(
            f"Need at least {order + 1} points for polynomial order {order}, got {train.size}."
        )
    return np.polyfit(train, target, deg=int(order))


def calibrate_regression(
    train_spec: np.ndarray,
    target_standard: np.ndarray,
    peaks_train: np.ndarray,
    peaks_target: np.ndarray,
    order: int = 1,
    sample_spec: Optional[np.ndarray] = None,
    show: bool = True,
) -> np.ndarray:
    """Calibrate a pixel-space spectrum to energy space with polynomial fitting."""
    train_spec = np.asarray(train_spec, dtype=float)
    target_standard = np.asarray(target_standard, dtype=float)

    train_idx = np.asarray(peaks_train, dtype=int).reshape(-1)
    target_idx = np.asarray(peaks_target, dtype=int).reshape(-1)

    train_pts = train_spec[0][train_idx]
    target_pts = target_standard[0][target_idx]
    coef = _fit_poly(train_pts, target_pts, order=order)

    new_x = np.polyval(coef, train_spec[0])
    calibrated = np.array([new_x, train_spec[1]])

    print(f"Polynomial coefficients (highest degree first): {coef}")

    if sample_spec is not None:
        sample_spec = np.asarray(sample_spec, dtype=float)
        sample_new_x = np.polyval(coef, sample_spec[0])
        calibrated_sample = np.array([sample_new_x, sample_spec[1]])
        if show:
            _plot_calibration(
                [
                    {"x": calibrated[0], "y": calibrated[1], "name": "calibrated standard"},
                    {"x": target_standard[0], "y": target_standard[1], "name": "target standard"},
                    {"x": calibrated_sample[0], "y": calibrated_sample[1], "name": "calibrated sample"},
                ],
                title=f"Order-{order} calibration",
            )
        return calibrated, calibrated_sample

    if show:
        _plot_calibration(
            [
                {"x": calibrated[0], "y": calibrated[1], "name": "calibrated standard"},
                {"x": target_standard[0], "y": target_standard[1], "name": "target standard"},
            ],
            title=f"Order-{order} calibration",
        )
    return calibrated


class EDXAS_Calibrate:
    """Polynomial calibration mapping pixel positions to energy."""

    def __init__(
        self,
        train_spec: np.ndarray,
        target_spec: np.ndarray,
        train: np.ndarray,
        target: np.ndarray,
        order: int = 2,
        show: bool = True,
        save_param: bool = True,
        **param,
    ):
        self.train_spec = np.asarray(train_spec, dtype=float)
        self.target_spec = np.asarray(target_spec, dtype=float)
        self.order = int(order)
        self.show = bool(show)

        self.train = np.asarray(train, dtype=float).reshape(-1)
        self.target = np.asarray(target, dtype=float).reshape(-1)
        self.coef = _fit_poly(self.train, self.target, order=self.order)

        predicted_train = np.polyval(self.coef, self.train)
        self.rmse = float(np.sqrt(np.mean((predicted_train - self.target) ** 2)))
        print(f"Calibration RMSE: {self.rmse:.6f} eV")

        energy = np.polyval(self.coef, self.train_spec[0])
        # Keep column 1 as calibrated energy for backward compatibility.
        self.new_x = np.column_stack((self.train_spec[0], energy))

        if self.show:
            _plot_calibration(
                [
                    {"x": self.new_x[:, 1], "y": self.train_spec[1], "name": "exp standard", "style": param},
                    {"x": self.target_spec[0], "y": self.target_spec[1], "name": "synchrotron spec", "style": param},
                ],
                title=f"Order-{self.order} calibration",
            )

        if save_param:
            self._write_params()

    def sample_spec(self, sample_spec: np.ndarray, **param) -> np.ndarray:
        """Apply the fitted calibration to a sample spectrum.

        Returns
        -------
        ndarray, shape (N, 2)
            Column 0 is the original pixel axis, column 1 is calibrated energy.
        """
        sample_spec = np.asarray(sample_spec, dtype=float)
        energy = np.polyval(self.coef, sample_spec[0])
        sample_new_x = np.column_stack((sample_spec[0], energy))
        if self.show:
            _plot_calibration(
                [{"x": sample_new_x[:, 1], "y": sample_spec[1], "name": "exp sample", "style": param}],
                title=f"Order-{self.order} calibrated sample",
            )
        return sample_new_x

    def _write_params(self) -> None:
        """Write calibration parameters to a text file."""
        filename = date_today() + "_fitting_parameters.txt"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Fitting Parameters"])
            writer.writerow([f"Date: {date_today()}"])
            writer.writerow([f"Time: {time_now()}"])
            writer.writerow([f"Polynomial order: {self.order}"])
            writer.writerow([f"RMSE (eV): {self.rmse:.8f}"])
            writer.writerow(["Polynomial coefficients (highest degree first)"])
            writer.writerow(self.coef.tolist())
