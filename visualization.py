"""Interactive DXAS image visualisation using pyqtgraph.

``pyqtgraph`` (and a running Qt application) is required only when the
functions / classes in this module are **instantiated**, not at import time.
This allows the rest of the package to be imported in headless environments.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "show_roi",
    "PgSpec",
]


def show_roi(
    img: np.ndarray,
    m: np.ndarray,
    viewer: str = "pg",
    alpha: float = 0.1,
    **kwargs,
) -> None:
    """Overlay a boolean mask on an image.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        2-D image.
    m : ndarray, shape (H, W)
        Boolean mask.  Masked pixels are shown at full brightness;
        unmasked pixels are dimmed by *alpha*.
    viewer : {'plt', 'pg'}
        ``'plt'`` for matplotlib; ``'pg'`` for a pyqtgraph window.
    alpha : float
        Dimming factor for unmasked pixels (default: 0.1).
    **kwargs
        Extra arguments forwarded to the display call.
    """
    display = img * (alpha + m)
    if viewer == "plt":
        plt.imshow(display, **kwargs)
    elif viewer == "pg":
        import pyqtgraph as pg

        pg.image(display)


class PgSpec:
    """Interactive DXAS image viewer with linked ROI and spectral readout.

    Displays a 2-D absorption map (e.g., the *μx* image) in a
    ``pyqtgraph`` window with:

    * A histogram / LUT control for contrast adjustment.
    * A zoomed ROI sub-image panel.
    * A live spectral readout panel showing the column-averaged
      intensity within the selected ROI.

    Multiple ROIs (rectangular, circular, polygonal) can be added and
    queried for boolean masks.

    Parameters
    ----------
    data : ndarray, shape (H, W)
        2-D absorption image to display.
    title : str, optional
        Window title.

    Examples
    --------
    >>> viewer = PgSpec(mux_image, title="Ni foil")
    >>> viewer.add_roi_poly("main")
    >>> viewer.add_roi_rect("bg")
    >>> masks = viewer.getAllMasks()
    """

    def __init__(self, data: np.ndarray, title: str = "", **kwargs):
        import pyqtgraph as pg
        import skimage as sk

        self._pg = pg
        self._sk = sk

        # pyqtgraph displays rows from top to bottom, so we flip once here
        # and reverse the flip in getArrayRegion / getMask.
        self.data = data[::-1, :]
        self.roi_lst: dict = {}
        self.data_max = float(self.data.max())
        self._setup_window(title)

    # ------------------------------------------------------------------
    # Internal window construction
    # ------------------------------------------------------------------

    def _setup_window(self, title: str) -> None:
        pg = self._pg
        pg.setConfigOptions(imageAxisOrder="row-major")
        pg.mkQApp()

        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)

        # Main image panel (spans 3 rows)
        self.p1 = self.win.addPlot(rowspan=3)
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.p1.setAspectLocked()
        self.img.setImage(self.data)

        # Histogram / LUT control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.win.addItem(self.hist, rowspan=3)

        # Zoomed ROI sub-image
        self.win.nextRow()
        self.win.nextCol()
        self.p3 = self.win.addPlot(colspan=1)
        self.p3.setMaximumHeight(250)
        self.img_zoom = pg.ImageItem(levels=self.hist.getLevels())
        self.p3.addItem(self.img_zoom)

        # Spectral readout
        self.win.nextRow()
        self.win.nextCol()
        self.p2 = self.win.addPlot(colspan=1)
        self.p2.setMaximumHeight(250)

        self.win.resize(800, 800)
        self.win.show()

    def _update_plot(self, roi_object) -> None:
        selected = roi_object.getArrayRegion(self.data, self.img)
        self.img_zoom.setImage(selected)
        masked = np.ma.masked_equal(selected, 0)
        self.p2.plot(np.ma.mean(masked, axis=0), clear=True)

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def add_roi_rect(self, name: str = "rect") -> None:
        """Add a rectangular (rotatable) ROI."""
        pg = self._pg
        self.data_max += 5
        roi = pg.ROI([0, 0], [200, 200])
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addRotateHandle([0, 0], [0.5, 0.5])
        self.p1.addItem(roi)
        roi.setZValue(self.data_max)
        self.roi_lst[name] = roi
        roi.sigRegionChanged.connect(self._update_plot)
        self._update_plot(roi)

    def add_roi_circle(self, name: str = "circ") -> None:
        """Add a circular ROI."""
        pg = self._pg
        self.data_max += 5
        roi = pg.CircleROI([50, 50], size=50, radius=50)
        self.p1.addItem(roi)
        roi.setZValue(self.data_max)
        self.roi_lst[name] = roi
        roi.sigRegionChanged.connect(self._update_plot)
        self._update_plot(roi)

    def add_roi_poly(self, name: str = "poly") -> None:
        """Add a closed polygonal ROI."""
        pg = self._pg
        self.data_max += 5
        roi = pg.PolyLineROI(
            [[0, 0], [100, 0], [120, 120], [0, 100]], closed=True
        )
        self.p1.addItem(roi)
        roi.setZValue(self.data_max)
        self.roi_lst[name] = roi
        roi.sigRegionChanged.connect(self._update_plot)
        self._update_plot(roi)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def getArrayRegion(self, roi_name: str, show: bool = True) -> np.ndarray:
        """Return the image sub-array inside a named ROI.

        Parameters
        ----------
        roi_name : str
        show : bool
            Display via ``plt.imshow``.

        Returns
        -------
        ndarray
        """
        roi = self.roi_lst[roi_name]
        region = roi.getArrayRegion(self.data, self.img)[::-1, :]
        if show:
            plt.imshow(region)
        return region

    def getAngle(self, roi_name: str) -> float:
        """Return the rotation angle (degrees) of a named ROI."""
        angle = self.roi_lst[roi_name].angle()
        print(f"ROI '{roi_name}' angle: {angle:.2f} deg")
        return angle

    def getMask(self, roi_name: str, show: bool = True) -> np.ndarray:
        """Return a boolean mask for the named ROI.

        Supports rectangular (``pg.ROI``), circular (``pg.CircleROI``),
        and polygonal (``pg.PolyLineROI``) ROI types.

        Parameters
        ----------
        roi_name : str
        show : bool
            Display the masked image via ``plt.imshow``.

        Returns
        -------
        ndarray, shape (H, W), dtype bool
        """
        pg = self._pg
        sk = self._sk
        roi = self.roi_lst[roi_name]
        mask = np.zeros(self.data.shape)
        state = roi.getState()

        if type(roi) is pg.ROI:
            rr, cc = sk.draw.rectangle(
                np.array(state["pos"]),
                extent=np.array(state["size"]),
                shape=mask.shape,
            )
            mask[rr.astype(int), cc.astype(int)] = 1

        elif type(roi) is pg.CircleROI:
            center = np.array(state["pos"])
            size = np.array(state["size"])
            rr, cc = sk.draw.disk(center + size / 2, size[0] / 2, shape=mask.shape)
            mask[rr.astype(int), cc.astype(int)] = 1

        elif type(roi) is pg.PolyLineROI:
            polygon = np.array(state["pos"]) + np.array(state["points"])
            mask = sk.draw.polygon2mask(mask.shape, polygon)

        # Undo the display flip applied in __init__
        mask = mask.T[::-1, :]

        if show:
            plt.imshow(self.data[::-1, :] * (mask * 0.75 + 0.25))

        return mask

    def getAllMasks(self, show: bool = False, save: bool = False) -> dict:
        """Return boolean masks for all registered ROIs.

        Parameters
        ----------
        show : bool
            Display each mask individually.
        save : bool
            Reserved for future use.

        Returns
        -------
        dict
            ``{roi_name: bool ndarray}``
        """
        masks = {}
        for name in self.roi_lst:
            if show:
                plt.figure()
            masks[name] = self.getMask(name, show=show).astype(bool)
        return masks
