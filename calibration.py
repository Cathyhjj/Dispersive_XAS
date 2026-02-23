"""Pixel-to-energy calibration for DXAS.

Provides polynomial regression tools to map experimental pixel positions to
physical energy values (eV) using reference standard spectra.
"""

import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from .utils import date_today, time_now

__all__ = [
    "calibrate_regression",
    "EDXAS_Calibrate",
]


def calibrate_regression(
    train_spec: np.ndarray,
    target_standard: np.ndarray,
    peaks_train: np.ndarray,
    peaks_target: np.ndarray,
    order: int = 1,
    sample_spec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calibrate a pixel-space spectrum to energy space via linear regression.

    Parameters
    ----------
    train_spec : ndarray, shape (2, N)
        Experimental spectrum in pixel space.
    target_standard : ndarray, shape (2, M)
        Reference standard spectrum in energy space (eV).
    peaks_train : ndarray
        Pixel positions of matched feature peaks in the experimental spectrum.
    peaks_target : ndarray
        Corresponding energy positions in the standard spectrum.
    order : {1, 2}
        Polynomial degree for the regression (default: 1).
    sample_spec : ndarray or None
        If provided, apply the same calibration to this sample spectrum.

    Returns
    -------
    ndarray or tuple of ndarray
        Calibrated experimental spectrum, or
        ``(calibrated_standard, calibrated_sample)`` if *sample_spec* is given.
    """
    train = train_spec[0][peaks_train].reshape(-1, 1)
    target = target_standard[0][peaks_target].reshape(-1, 1)

    if order == 2:
        train = np.hstack((train, train ** 2))
        target = np.hstack((target, target ** 2))

    reg = linear_model.LinearRegression()
    reg.fit(train, target)
    print(
        f"slope: {reg.coef_[0, 0]:.6f} | "
        f"intercept: {reg.intercept_[0]:.4f} | "
        f"coefficients: {reg.coef_}"
    )

    if order == 1:
        new_x = reg.predict(train_spec[0].reshape(-1, 1))
    else:
        X = np.hstack(
            (train_spec[0].reshape(-1, 1), train_spec[0].reshape(-1, 1) ** 2)
        )
        new_x = reg.predict(X)

    calibrated = np.array([new_x[:, 0], train_spec[1]])

    plt.title(f"Order-{order} calibration")
    plt.plot(calibrated[0], calibrated[1], lw=3, alpha=0.8, label="calibrated spec")
    plt.plot(
        target_standard[0], target_standard[1], lw=3, alpha=0.6, label="target standard"
    )

    if sample_spec is not None:
        if order == 1:
            sample_new_x = reg.predict(sample_spec[0].reshape(-1, 1))
        else:
            X_s = np.hstack(
                (sample_spec[0].reshape(-1, 1), sample_spec[0].reshape(-1, 1) ** 2)
            )
            sample_new_x = reg.predict(X_s)
        calibrated_sample = np.array([sample_new_x[:, 0], sample_spec[1]])
        plt.plot(
            calibrated_sample[0],
            calibrated_sample[1],
            lw=3,
            alpha=0.8,
            label="calibrated sample",
        )
        plt.legend()
        return calibrated, calibrated_sample

    plt.legend()
    return calibrated


class EDXAS_Calibrate:
    """Polynomial energy calibration for EDXAS data.

    Fits a polynomial mapping from experimental pixel positions to physical
    energy (eV) using a set of matched feature points identified in a
    reference standard spectrum.

    Parameters
    ----------
    train_spec : ndarray, shape (2, N)
        Experimental spectrum in pixel space.
    target_spec : ndarray, shape (2, M)
        Reference standard spectrum in energy space (eV).
    train : ndarray
        Pixel positions of matched features in *train_spec*.
    target : ndarray
        Corresponding energy positions in *target_spec*.
    order : int
        Polynomial degree (default: 2).
    show : bool
        Plot the calibration overlay (default: ``True``).
    save_param : bool
        Write calibration parameters to a dated text file (default: ``True``).
    **param
        Extra keyword arguments forwarded to ``plt.plot``.

    Examples
    --------
    >>> cal = EDXAS_Calibrate(foil_spec, standard_spec, train_pts, target_pts, order=2)
    >>> cal_sample_x = cal.sample_spec(sample_spec)
    """

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
        self.train_spec = train_spec
        self.target_spec = target_spec
        self.order = order
        self.show = show

        self._poly = PolynomialFeatures(degree=self.order)
        train_tf = self._poly.fit_transform(train.reshape(-1, 1))
        target_tf = self._poly.fit_transform(target.reshape(-1, 1))

        self._reg = linear_model.LinearRegression()
        self._reg.fit(train_tf, target_tf)

        train_spec_tf = self._poly.fit_transform(train_spec[0].reshape(-1, 1))
        self.new_x = self._reg.predict(train_spec_tf)

        if self.show:
            plt.plot(self.new_x[:, 1], train_spec[1], label="exp standard", **param)
            plt.plot(target_spec[0], target_spec[1], label="synchrotron spec", **param)
            plt.legend()

        if save_param:
            self._write_params()

    def sample_spec(self, sample_spec: np.ndarray, **param) -> np.ndarray:
        """Apply the fitted calibration to a sample spectrum.

        Parameters
        ----------
        sample_spec : ndarray, shape (2, N)
            Sample spectrum in pixel space.
        **param
            Extra keyword arguments forwarded to ``plt.plot``.

        Returns
        -------
        ndarray, shape (N, order+1)
            Calibrated polynomial features; column 1 contains the energy axis.
        """
        sample_tf = self._poly.fit_transform(sample_spec[0].reshape(-1, 1))
        sample_new_x = self._reg.predict(sample_tf)
        if self.show:
            plt.plot(sample_new_x[:, 1], sample_spec[1], label="exp sample", **param)
            plt.legend()
        return sample_new_x

    def _write_params(self) -> None:
        """Write calibration parameters to a dated CSV-formatted text file."""
        filename = date_today() + "_fitting_parameters.txt"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Fitting Parameters"])
            writer.writerow([f"Date: {date_today()}"])
            writer.writerow([f"Time: {time_now()}"])
            writer.writerow([f"Regression order: {self.order}"])
            writer.writerow(["Regression coefficients"])
            writer.writerows(self._reg.coef_)
            writer.writerow(["Regression intercept"])
            writer.writerow(self._reg.intercept_)
