from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dispersive_XAS.core.roi import (  # noqa: E402
    build_roi_mask,
    make_tilted_band_roi,
    prepare_roi_weights,
    roi_weighted_column_mean,
)


def test_row_range_roi_matches_plain_column_mean():
    """A horizontal row-range ROI should match a direct column mean."""
    image = np.arange(4 * 5, dtype=float).reshape(4, 5)

    result = roi_weighted_column_mean(image, row_range=(1, 3))

    np.testing.assert_allclose(result, image[1:3, :].mean(axis=0))


def test_no_roi_uses_all_detector_rows():
    """Passing no ROI settings should process the full detector height."""
    image = np.arange(4 * 5, dtype=float).reshape(4, 5)

    result = roi_weighted_column_mean(image)

    np.testing.assert_allclose(result, image.mean(axis=0))


def test_tilted_band_roi_selects_pixels_per_column():
    """A tilted-band ROI should include expected rows at both detector edges."""
    roi = make_tilted_band_roi(
        shape=(6, 5),
        left_center_row=1.0,
        right_center_row=3.0,
        half_width=0.5,
    )

    mask = build_roi_mask((6, 5), roi=roi)

    assert mask.shape == (6, 5)
    assert mask[:, 0].sum() >= 1
    assert mask[:, -1].sum() >= 1
    assert mask[1, 0]
    assert mask[3, -1]


def test_tilted_band_weights_are_fractional_at_boundaries():
    """Tilted-band processing should anti-alias row-boundary weights."""
    roi = make_tilted_band_roi(
        shape=(8, 5),
        left_center_row=2.25,
        right_center_row=4.25,
        half_width=0.75,
    )

    _spec, _bounds, row_weights, col_weight_sum = prepare_roi_weights((8, 5), roi=roi)

    assert np.any((row_weights > 0.0) & (row_weights < 1.0))
    np.testing.assert_allclose(col_weight_sum, row_weights.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(col_weight_sum, np.full((1, 5), 1.5))


def test_tilted_band_column_mean_uses_fractional_vertical_weights():
    """A tilted ROI should still reduce vertically by detector column."""
    image = np.repeat(np.arange(8, dtype=float)[:, None], 5, axis=1)
    roi = make_tilted_band_roi(
        shape=image.shape,
        left_center_row=2.25,
        right_center_row=4.25,
        half_width=0.75,
    )

    result = roi_weighted_column_mean(image, roi=roi)
    expected = np.array([2.3333333, 2.6666667, 3.3333333, 3.6666667, 4.3333333])

    np.testing.assert_allclose(result, expected, atol=1e-6)
