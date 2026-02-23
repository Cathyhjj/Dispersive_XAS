"""Miscellaneous utility functions for DXAS analysis."""

import datetime
import glob
import os

import numpy as np

__all__ = [
    "date_today",
    "time_now",
    "timestamp_convert",
    "color_gradient",
    "change_font_size",
    "binning",
    "make_gif",
    "make_video",
]


def date_today() -> str:
    """Return today's date as a YYYYMMDD string."""
    return datetime.datetime.today().strftime("%Y%m%d")


def time_now() -> str:
    """Return the current time as a HHMM string."""
    return datetime.datetime.now().strftime("%H%M")


def timestamp_convert(timestamp: float) -> str:
    """Convert a Unix timestamp to a human-readable datetime string.

    Parameters
    ----------
    timestamp : float
        Unix timestamp.

    Returns
    -------
    str
        Formatted datetime string ``YYYY-MM-DD HH:MM:SS``.
    """
    dt = datetime.datetime.fromtimestamp(timestamp)
    formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted)
    return formatted


def color_gradient(c1: str, c2: str, mix: float = 0) -> str:
    """Linearly interpolate between two colours.

    Parameters
    ----------
    c1, c2 : str
        Any matplotlib-recognised colour strings (hex, named, etc.).
        ``c1`` is returned at ``mix=0``; ``c2`` at ``mix=1``.
    mix : float
        Interpolation factor in [0, 1].

    Returns
    -------
    str
        Hex colour string.
    """
    import matplotlib as mpl

    rgb1 = np.array(mpl.colors.to_rgb(c1))
    rgb2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * rgb1 + mix * rgb2)


def change_font_size(size: int) -> None:
    """Update matplotlib font sizes globally.

    Parameters
    ----------
    size : int
        Font size applied to legend, axis labels, ticks, and title.
    """
    from matplotlib import pylab

    pylab.rcParams.update(
        {
            "legend.fontsize": size,
            "axes.labelsize": size,
            "axes.titlesize": size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "font.family": "calibri",
        }
    )


def binning(arr: np.ndarray, downscale: int = 2) -> np.ndarray:
    """Spatially bin a 2-D or 3-D array by averaging.

    Parameters
    ----------
    arr : ndarray, shape (H, W) or (N, H, W)
        Input array.
    downscale : int
        Binning factor applied to both spatial dimensions.

    Returns
    -------
    ndarray
        Binned array with spatial dimensions divided by *downscale*.
    """
    if arr.ndim == 3:
        n, h, w = arr.shape
        reshaped = arr.reshape(
            n,
            h // downscale,
            downscale,
            w // downscale,
            downscale,
        )
    elif arr.ndim == 2:
        h, w = arr.shape
        reshaped = arr.reshape(
            h // downscale,
            downscale,
            w // downscale,
            downscale,
        )
    else:
        raise ValueError("arr must be 2-D or 3-D, got shape %s" % str(arr.shape))
    return np.average(reshaped, axis=(-1, -3))


def make_gif(
    img_list=None,
    extension: str = "jpg",
    fps: int = 20,
    file_name: str = "movie.gif",
    **kwargs,
) -> None:
    """Save a list of images as an animated GIF.

    Parameters
    ----------
    img_list : list of ndarray or None
        Images to write.  If *None*, all ``*.{extension}`` files in the
        current directory are collected in sorted order.
    extension : str
        File extension to glob when *img_list* is ``None``.
    fps : int
        Frames per second.
    file_name : str
        Output filename (saved inside ``generated_gif/``).
    **kwargs
        Extra keyword arguments forwarded to ``imageio.mimsave``.
    """
    import imageio

    os.makedirs("generated_gif", exist_ok=True)
    if img_list is None:
        img_list = [imageio.imread(f) for f in sorted(glob.glob("*.%s" % extension))]
    imageio.mimsave(os.path.join("generated_gif", file_name), img_list, fps=fps, **kwargs)


def make_video(
    img_list=None,
    fps: int = 20,
    file_name: str = "video.mp4",
    extension: str = "jpg",
    **kwargs,
) -> None:
    """Save a list of images as a video file.

    Parameters
    ----------
    img_list : list of ndarray or None
        Frames to write.  If *None*, all ``*.{extension}`` files in the
        current directory are used.
    fps : int
        Frames per second.
    file_name : str
        Output filename (saved inside ``generated_gif/``).
    extension : str
        File extension to glob when *img_list* is ``None``.
    **kwargs
        Extra keyword arguments forwarded to ``imageio.get_writer``.
    """
    import imageio

    os.makedirs("generated_gif", exist_ok=True)
    writer = imageio.get_writer(
        os.path.join("generated_gif", file_name), fps=fps, **kwargs
    )
    if img_list is None:
        img_list = [imageio.imread(f) for f in sorted(glob.glob("*.%s" % extension))]
    for frame in img_list:
        writer.append_data(frame)
    writer.close()
