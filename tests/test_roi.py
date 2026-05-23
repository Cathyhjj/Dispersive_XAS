from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dispersive_XAS.core.roi import (  # noqa: E402
    build_roi_mask,
    make_tilted_band_roi,
    roi_weighted_column_mean,
)


def test_row_range_roi_matches_plain_column_mean():
    """A horizontal row-range ROI should match a direct column mean."""
    image = np.arange(4 * 5, dtype=float).reshape(4, 5)

    result = roi_weighted_column_mean(image, row_range=(1, 3))

    np.testing.assert_allclose(result, image[1:3, :].mean(axis=0))


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
