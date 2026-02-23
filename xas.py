import datetime
import glob
from . import h5io
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
# from tools import mda
import pandas as pd
import pyqtgraph as pg
import scipy.ndimage as nd
# import skimage.transform as tf
import skimage as sk
from pyqtgraph.Qt import QtCore, QtGui
from scipy import interpolate
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "date_today",
    "time_now",
    "timestamp_convert",
    "change_font_size",
    "show_roi",
    "intensity_at_energy",
    "spectrum_generate",
    "standard_spec",
    "norm_spec",
    "make_gif",
    "make_video",
    "raw_loading",
    "load_processed",
    "load_processed_scans",
    "pre_process",
    "pre_process_scan",
    "interpt_spec",
    "PgSpec",
    "peak_finder",
    "calibrate_regression",
    "spec_cropping",
    "spec_shaper",
    "spec_wrapper",
    "load_mda",
    "timestamp_convert",
]


def date_today():
    return datetime.datetime.today().strftime("%Y%m%d")


def time_now():
    return datetime.datetime.now().strftime("%H%M")


def timestamp_convert(timestamp):
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_date)
    return formatted_date


def change_font_size(size):
    from matplotlib import pylab

    params = {
        "legend.fontsize": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "font.family": "calibri",
    }
    pylab.rcParams.update(params)


def show_roi(img, m, viewer="pg", alpha=0.1, **kwargs):
    """
    m: mask
    viewer: "plt" or "pg" (pygtgraph viewer)
    """
    if viewer == "plt":
        plt.imshow(img * (alpha + m), **kwargs)
    elif viewer == "pg":
        pg.image(img * (alpha + m))


def spectrum_generate(crop_mux, mode="average", title="temp", show=True, **kwargs):
    ener_pnt = np.arange(crop_mux.shape[1])
    if mode == "sum":
        spectrum_mux = np.sum(crop_mux, axis=0)
    elif mode == "average":
        spectrum_mux = np.average(crop_mux, axis=0)

    spec_original = np.array([ener_pnt, spectrum_mux])
    if show:
        # plt.figure()
        plt.plot(ener_pnt, spectrum_mux, **kwargs)
        plt.title(title)
    return spec_original


def standard_spec(sample, norm=True, intp=False, pnts=3000):
    """
    These standard XAS are from BM 14 of Spring8, 400 ms
    sample: Ag, AgO, AgNO3, Pd, PdO, Pd_ESRF
            Ag_norm, AgO_norm, AgNO3_norm, Pd_norm, PdO_norm
            Ag_flat, AgO_flat, AgNO3_flat, Pd_flat, PdO_flat
    """
    samples = {
        "Ag_norm": "Ag_K_standard.dat.nor",
        "Ag2O_norm": "Ag2O_K_standard.dat.nor",
        "AgNO3_norm": "AgNO3_k_standard.txt.nor",
        "AgCl_norm": "Ag_chloride.xmu.nor",
        "PdO_norm": "PdO_K_standard.dat.nor",
        "Pd_norm": "Pd_K_standard.dat.nor",
        "Pd_ESRF_norm": "esrf_BM23_Pd.txt.nor",
        "Ag": "Ag_K_standard.dat",
        "Ag2O": "Ag2O_K_standard.dat",
        "AgNO3": "AgNO3_k_standard.txt",
        "Pd": "Pd_K_standard.dat",
        "PdO": "PdO_K_standard.dat",
        "Pt-L3": "PtFoil_XAFS_Pos15empty.0001",
        "Ag_flat": "Ag_K_standard.dat.nor.flat",
        "Ag2O_flat": "Ag2O_K_standard.dat.nor.flat",
        "AgNO3_flat": "AgNO3_k_standard.txt.nor.flat",
        "AgCl_flat": "Ag_chloride.xmu.nor.flat",
        "PdO_flat": "PdO_K_standard.dat.nor.flat",
        "Pd_flat": "Pd_K_standard.dat.nor.flat",
    }

    dataDir = os.path.dirname(__file__)
    if "norm" in sample:
        standard_file = os.path.join(
            dataDir, "standard_XAS", "normalized_bk_rm", samples[sample]
        )
        standard_spectrum = np.loadtxt(standard_file, usecols=(0, 1))
    elif "flat" in sample:
        standard_file = os.path.join(
            dataDir, "standard_XAS", "normalized_bk_rm", "flatened", samples[sample]
        )
        standard_spectrum = np.loadtxt(standard_file, usecols=(0, 3))
    else:
        standard_file = os.path.join(dataDir, "standard_XAS", samples[sample])
        standard_spectrum = np.loadtxt(standard_file)

    print("loading %s" % (standard_file))
    if intp:
        standard_spectrum = interpt_spec(standard_spectrum, pnts=pnts)
    if norm:
        standard_norm = standard_spectrum.copy()
        standard_norm = norm_spec(standard_norm)
        return standard_norm
    return standard_spectrum


def norm_spec(spectrum, x0=None, x1=None, show=False, **kwargs):
    """
    This function is used to normalize the intensity into the range from 0 to 1
    Parameter
    --------------
    spectrum: 2 dimentional spectrum data
    x0, x1: normalizing range
    show: default is False
    **kwargs for plot
    """
    spec = spec_shaper(spectrum)
    spec_intensity = spec[1]
    spec_energy = spec[0]

    if x1 is None:
        max_value = spec_intensity.max()
        min_value = spec_intensity.min()
    elif type(x0) is list:
        mask_energy_arr = np.ma.masked_inside(spec_energy, x0[0], x1[0])
        for i in range(len(x0)):
            mask_energy_arr *= np.ma.masked_inside(spec_energy, x0[i], x1[i])
        #        plt.plot(spec_energy[mask_energy_arr.mask],spec_intensity[mask_energy_arr.mask], "o", lw = 5, alpha = 0.3, )
        max_value = spec_intensity[mask_energy_arr.mask].max()
        min_value = spec_intensity[mask_energy_arr.mask].min()
    else:
        mask_energy_arr = np.ma.masked_inside(spec_energy, x0, x1)
        max_value = spec_intensity[mask_energy_arr.mask].max()
        min_value = spec_intensity[mask_energy_arr.mask].min()

    spec_intensity_norm = (spec_intensity - min_value) / (max_value - min_value)
    if show:
        plt.plot(spec_energy, spec_intensity_norm, **kwargs)

    spec_norm = spec_wrapper(spec_energy, spec_intensity_norm, output=spectrum.shape)
    return spec_norm


def raw_loading(folder, detector=None):
    """
    loading all the files in certain folder into array

    parameter
    ------------
    detector: default 'Ximea', or 'None'
    denoise_size: size for median filter, default = 3
    """
    try:
        file_raw = h5io.h5read("%s/*.h5" % (folder))
    except IOError:
        file_raw = np.array([h5io.h5read("%s" % (folder))])
    file_list = []

    if detector == "Ximea":
        key = "raw_data"
    elif detector == None:
        key = "var1"

    for kk in range(len(file_raw)):
        if detector == "Zyla":
            file_kk = np.array(
                file_raw[kk]["entry_0000"]["measurement"]["andor-zyla"]["data"][0],
                dtype=float,
            )
        else:
            file_kk = np.array(file_raw[kk][key], dtype=float)
    file_list.append(file_kk)
    file_array = np.array(file_list)

    return file_array


def load_mda(mda_file, show=True, save=True):
    file_name = glob.glob(mda_file)[0]
    if len(mda_file) == 0:
        print("cannot find the file")
    else:
        print("loading %s" % file_name)
    data = mda.readMDA(file_name)

    motor_name_all = []
    motor_values_all = []

    for i in range(len(data[1].p)):
        motor_n = i  # the i_th motor
        motor_name_i, motor_values_i = (
            data[1].p[motor_n].name.decode(),
            data[1].p[motor_n].data,
        )
        motor_name_all.append(motor_name_i)
        motor_values_all.append(motor_values_i)
        if show:
            plt.figure(i)
            plt.plot(motor_values_i)
            plt.title(motor_name_i)

    for i in range(len(data[1].d)):
        motor_n = i  # the i_th motor
        motor_name_i, motor_values_i = (
            data[1].d[motor_n].name.decode(),
            data[1].d[motor_n].data,
        )
        motor_name_all.append(motor_name_i)
        motor_values_all.append(motor_values_i)
        if show:
            plt.figure(i + 100)
            plt.plot(motor_values_i)
            plt.title(motor_name_i)
    time = data[1].time.decode()
    motor_values_all = np.asarray(motor_values_all)

    if save:
        # Saving to txt files
        savename = file_name.split(os.sep)[-1].replace("mda", "txt")
        savedata = pd.DataFrame(
            data=motor_values_all.transpose(), columns=motor_name_all
        )

        os.makedirs("mda2csv", exist_ok=True)
        f = open(os.path.join("mda2csv", "%s" % savename), "w")
        f.write("#" + time + "\n")
        savedata.to_csv(f, sep="\t", index=False, header=time)
        f.close()
    return savedata


def load_processed(folder):
    """
    load data generated by "pre_process" function
    ------------
    return dictionary
    """
    data = h5io.h5read(os.path.join(folder, "00_data.h5"))["var1"]
    flat = h5io.h5read(os.path.join(folder, "01_flat.h5"))["var1"]
    transmission = h5io.h5read(os.path.join(folder, "02_transmission.h5"))["var1"]
    mux = h5io.h5read(os.path.join(folder, "03_mux.h5"))["var1"]

    return {"data": data, "flat": flat, "transmission": transmission, "mux": mux}


def load_processed_scans(folder):
    """
    load data generated by "pre_process_scan" function
    ------------
    return dictionary
    """

    processed_data = {}
    all_lst = ["00_data", "01_flat", "02_transmission", "03_mux"]

    for file_name in all_lst:
        # print(f"loading {file_name} files")

        file_lst = glob.glob(os.path.join(folder, f"{file_name}", "*.h5"))
        print(file_lst)
        print(os.path.join(folder, f"{file_name}", "*.h5"))
        processed_data[f"{file_name[3:]}"] = np.asarray(
            [h5io.h5read(file_i)["var1"] for file_i in file_lst]
        )

    return processed_data


def pre_process_scan(
    data_darkcorr, flat_darkcorr, denoise_size=3, savedata=True, prefix="preproc"
):
    """
    parameter
    ------------
    for
    denoise_size: size for median filter, default = 3
    ------------
    return dictionary
    """

    # Denoising using Pepper and Salt Medien denoising filter
    data_darkcorr_denoise = nd.median_filter(data_darkcorr, size=denoise_size)
    flat_darkcorr_denoise = nd.median_filter(flat_darkcorr, size=denoise_size)

    # Make negative value into postive value -->0.001, in an ugly way
    flat_darkcorr_denoise[flat_darkcorr_denoise <= 0] = 0.0001
    data_darkcorr_denoise[data_darkcorr_denoise <= 0] = 0.0001

    # Calculate mux
    transmission = data_darkcorr_denoise / flat_darkcorr_denoise
    mux = -np.log(transmission)
    if savedata is True:
        directory = os.path.join(os.getcwd(), prefix)
        folder = ["00_data", "01_flat", "02_transmission", "03_mux"]
        for subfolder_i in folder:
            if not os.path.exists(os.path.join(directory, subfolder_i)):
                os.makedirs(os.path.join(directory, subfolder_i))
        # Save data in to h5 files
        for i in range(data_darkcorr.shape[0]):
            h5io.h5write(
                os.path.join(directory, folder[0], "scan_%05d_data.h5" % i),
                var1=data_darkcorr_denoise[i],
            )
            h5io.h5write(
                os.path.join(directory, folder[1], "scan_%05d_flat.h5" % i),
                var1=flat_darkcorr_denoise[i],
            )
            h5io.h5write(
                os.path.join(directory, folder[2], "scan_%05d_transmission.h5" % i),
                var1=transmission[i],
            )
            h5io.h5write(
                os.path.join(directory, folder[3], "scan_%05d_mux.h5" % i), var1=mux[i]
            )

    return {
        "data": data_darkcorr_denoise,
        "flat": flat_darkcorr_denoise,
        "transmission": transmission,
        "mux": mux,
    }


def pre_process(data, flat, dark=None, denoise_size=3, savedata=True, prefix="preproc"):
    """
    parameter
    ------------
    denoise_size: size for median filter, default = 3

    ------------
    return dictionary containing "data_processed",
                                 "flat_processed",
                                 "transmission",
                                 "mux"(absorption)
    """

    # Darkfield correction1
    if dark is not None:
        data_darkcorr = data - dark
        flat_darkcorr = flat - dark
    else:
        data_darkcorr = data
        flat_darkcorr = flat

    # Denoising using Pepper and Salt Medien denoising filter
    data_darkcorr_denoise = nd.median_filter(data_darkcorr, size=denoise_size)
    flat_darkcorr_denoise = nd.median_filter(flat_darkcorr, size=denoise_size)

    # Make negative value into postive value -->0.001, in an ugly way
    flat_darkcorr_denoise[flat_darkcorr_denoise <= 0] = 0.0001
    data_darkcorr_denoise[data_darkcorr_denoise <= 0] = 0.0001

    # Calculate mux
    transmission = data_darkcorr_denoise / flat_darkcorr_denoise
    mux = -np.log(transmission)
    if savedata is True:
        directory = os.path.join(os.getcwd(), date_today() + "_" + prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save data in to h5 files
        h5io.h5write(os.path.join(directory, "00_data.h5"), var1=data_darkcorr_denoise)
        h5io.h5write(os.path.join(directory, "01_flat.h5"), var1=flat_darkcorr_denoise)
        h5io.h5write(os.path.join(directory, "02_transmission.h5"), var1=transmission)
        h5io.h5write(os.path.join(directory, "03_mux.h5"), var1=mux)

    return {
        "data": data_darkcorr_denoise,
        "flat": flat_darkcorr_denoise,
        "transmission": transmission,
        "mux": mux,
    }


def ximea_correction(data, crop_slice=None, show=False):
    """
    ximea_correction(data, crop_slice)
    crop_slice: use np.s_[:,:]
    return data_corr_minus, data_corr_divide

    """
    if crop_slice == None:
        crop_slice = np.s_[25:475, : data.shape[1]]
    # crop region for correction
    corr_mask = np.zeros_like(data) + 0.2
    ximea_corr_roi = crop_slice
    corr_mask[ximea_corr_roi] = 1
    # offset values
    ximea_offset = np.average(nd.median_filter(data[ximea_corr_roi], size=3), axis=0)

    # correct
    data_corr_minus = data - ximea_offset
    data_corr_divide = data / ximea_offset
    if show:
        plt.imshow(data * corr_mask, vmin=400, vmax=1000)
        plt.figure()
        plt.plot(ximea_offset)
        plt.title("offset")
        pg.image(data_corr_minus.transpose())
        pg.image(data_corr_divide.transpose())
        pg.image(nd.median_filter(data_corr_minus, size=3))
        pg.image(nd.median_filter(data_corr_divide, size=3))

    return data_corr_minus, data_corr_divide


def intensity_at_energy(energy, spec_1d, E_eV):
    """
    look for one specific value in a spectrum
    Parameter
    --------------
    energy: in eV
    spec_1d: spectrum intensity
    E_eV: the energy you want to look at

    - return
    the intensity at energy E_eV
    """
    f = interpolate.interp1d(energy, spec_1d)
    xnew = E_eV
    return f(xnew)


def interpt_spec(spec, x_min=None, x_max=None, pnts=3000):
    """
    interpolate spectrum from certain range
    --------------
    - Parameter
    spec: spectrum data
    x_min: min data you want to interpolate
    x_max: max data you want to interpolate
    pnts : interpolate points, default is 3000

    - return
    interpolated spectrum, shape (2, n)
    """
    import scipy.interpolate as si

    spec = spec_shaper(spec)
    if x_min == None:
        x_min = spec[0].min()
        x_max = spec[0].max()
    new_x = np.linspace(x_min, x_max, pnts, endpoint=False)
    f = si.interp1d(spec[0], spec[1])
    new_y = f(new_x)
    return np.array([new_x, new_y])


class PgSpec(pg.PlotWidget):

    def __init__(self, data, title=None, **kwarg):
        super().__init__()
        self.data = data[::-1, :]  # pg.image shows image in a flipped way
        self.image(title)
        self.roi_lst = {}
        self.data_max = self.data.max()

    def updatePlot(self, roi_object):
        # global img, roi, data, p2
        selected = roi_object.getArrayRegion(self.data, self.img)
        self.img_zoom.setImage(selected)

        # mean = selected.mean(axis=0)
        masked_selected = np.ma.masked_equal(selected, 0)
        mean = np.ma.mean(masked_selected, axis=0)
        self.p2.plot(mean, clear=True)

    def image(self, title):
        pg.setConfigOptions(imageAxisOrder="row-major")
        pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot(rowspan=3)
        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.p1.setAspectLocked()
        # Generate image data
        self.img.setImage(self.data)  # image is flipped for some reason
        # set position and scale of image
        # img.scale(0.2, 0.2)
        # 20221021 the following line does not work
        # self.img.translate(-50, 0)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.win.addItem(self.hist, rowspan=3)

        self.win.nextRow()
        self.win.nextCol()
        self.p3 = self.win.addPlot(colspan=1)
        self.p3.setMaximumHeight(250)
        self.img_zoom = pg.ImageItem(levels=self.hist.getLevels())
        self.p3.addItem(self.img_zoom)
        # p3.setAspectLocked()

        # Another plot area for displaying ROI data
        self.win.nextRow()
        self.win.nextCol()
        self.p2 = self.win.addPlot(colspan=1)
        self.p2.setMaximumHeight(250)
        self.win.resize(800, 800)
        self.win.show()

    def add_roi_rect(self, name="rect"):
        self.data_max += 5
        self.roi_rect = pg.ROI([0, 0], [200, 200])
        self.roi_rect.addScaleHandle([1, 1], [0, 0])
        self.roi_rect.addRotateHandle([0, 0], [0.5, 0.5])
        self.p1.addItem(self.roi_rect)
        self.roi_rect.setZValue(self.data_max)

        self.roi_lst[name] = self.roi_rect
        self.roi_rect.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot(roi_object=self.roi_rect)

    def add_roi_circle(self, name="circ"):
        self.data_max += 5
        self.roi_circle = pg.CircleROI([50, 50], size=50, radius=50)
        self.p1.addItem(self.roi_circle)
        self.roi_circle.setZValue(self.data_max)
        self.roi_lst[name] = self.roi_circle
        self.roi_circle.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot(roi_object=self.roi_circle)

    def add_roi_poly(self, name="poly"):
        self.data_max += 5
        self.roi_poly = pg.PolyLineROI(
            [[0, 0], [100, 0], [120, 120], [0, 100]], closed=True
        )
        self.p1.addItem(self.roi_poly)
        self.roi_poly.setZValue(self.data_max)
        self.roi_lst[name] = self.roi_poly
        self.roi_poly.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot(roi_object=self.roi_poly)

    def getArrayRegion(self, roi_name, show=True):
        selected_roi = self.roi_lst[roi_name]
        array_region = selected_roi.getArrayRegion(self.data, self.img)[::-1, :]
        if show:
            plt.imshow(array_region)
        return array_region

    def getAngle(self, roi_name):
        selected_roi = self.roi_lst[roi_name]
        angle = selected_roi.angle()
        print(f"The ROI has rotated by {angle} deg.")
        return angle

    def getMask(self, roi_name=None, show=True):
        self.mask_all = {}
        selected_roi = self.roi_lst[roi_name]
        mask = np.zeros((self.data.shape))
        roi_state = selected_roi.getState()

        if type(selected_roi) == pg.ROI:
            rr, cc = sk.draw.rectangle(
                np.array(roi_state["pos"]),
                extent=np.array(roi_state["size"]),
                shape=mask.shape,
            )
            mask[rr.astype(int), cc.astype(int)] = 1

        elif type(selected_roi) == pg.CircleROI:
            center = np.array(roi_state["pos"])
            size = np.array(roi_state["size"])
            rr, cc = sk.draw.disk(center + size / 2, size[0] / 2, shape=mask.shape)
            mask[rr.astype(int), cc.astype(int)] = 1

        elif type(selected_roi) == pg.PolyLineROI:
            polygon_coordinates = np.array(roi_state["pos"]) + np.array(
                roi_state["points"]
            )
            mask = sk.draw.polygon2mask(mask.shape, polygon_coordinates)

        mask = mask.transpose()[::-1, :]
        self.mask_all = {roi_name: mask}

        if show:
            plt.imshow(self.data[::-1, :] * (mask * 0.75 + 0.25))
        return mask

    def getAllMasks(self, show=False, save=False):
        self.mask_lst = {}
        for key_i in self.roi_lst.keys():
            if show:
                plt.figure()
            mask_i = self.getMask(key_i, show=show)
            self.mask_lst[key_i] = mask_i.astype(bool)
            if save:
                # TODO
                pass
        return self.mask_lst


class pg_spec_old(object):

    def __init__(self, data, title=""):
        self.data = data
        self.image(title)

    def updatePlot(self):
        # global img, roi, data, p2
        selected = self.roi.getArrayRegion(self.data, self.img)
        self.p2.plot(selected.mean(axis=0), clear=True)
        self.img_zoom.setImage(selected)

    def image(self, title):
        pg.setConfigOptions(imageAxisOrder="row-major")
        pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot(rowspan=3)
        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.p1.setAspectLocked()
        # Generate image data
        self.img.setImage(self.data)
        # set position and scale of image
        # img.scale(0.2, 0.2)
        # 20221021 the following line does not work
        # self.img.translate(-50, 0)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [200, 200])
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addRotateHandle([0, 0], [0.5, 0.5])
        self.p1.addItem(self.roi)
        self.roi.setZValue(30)  # make sure ROI is drawn above image

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.win.addItem(self.hist, rowspan=3)

        self.win.nextRow()
        self.win.nextCol()
        self.p3 = self.win.addPlot(colspan=1)
        self.p3.setMaximumHeight(250)
        self.img_zoom = pg.ImageItem(levels=self.hist.getLevels())
        self.p3.addItem(self.img_zoom)
        # p3.setAspectLocked()

        # Another plot area for displaying ROI data
        self.win.nextRow()
        self.win.nextCol()
        self.p2 = self.win.addPlot(colspan=1)
        self.p2.setMaximumHeight(250)
        self.win.resize(800, 800)

        self.win.show()

        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()

    def getROI(self):
        return self.roi

    def getArraySlice(self):
        return self.roi.getArraySlice(self.data, self.img)[0]

    def getArrayRegion(self):
        selected = self.roi.getArrayRegion(self.data, self.img)
        # affine_slice = self.roi.getAffineSliceParams(self.data, self.img)
        # print("--shape--:\n" + str(affine_slice[0]) + "\n--vector--:\n" +  str(affine_slice[1]) + "\n--origin--:\n" + str(affine_slice[2]))
        return selected

    def getAffileSliceParams(self):
        affine_slice = self.roi.getAffineSliceParams(self.data, self.img)
        print(
            "--shape--:\n"
            + str(affine_slice[0])
            + "\n--vector--:\n"
            + str(affine_slice[1])
            + "\n--origin--:\n"
            + str(affine_slice[2])
        )
        return affine_slice

    def getAngle(self):
        angle = self.roi.angle()
        return angle

    def getArrayCoord(self, show=True):
        """
        still bugs and need to be fixed
        """
        slice_coordinates = self.roi.getArraySlice(self.data, self.img)
        print("slice_coordinates:", slice_coordinates)
        coordinates = [
            slice_coordinates[0][0].start,
            slice_coordinates[0][0].stop,
            slice_coordinates[0][1].start,
            slice_coordinates[0][1].stop,
        ]
        if show:
            plt.imshow(self.data)
            plt.scatter(coordinates[0], coordinates[1], color="y")
            plt.scatter(coordinates[0], coordinates[3], color="y")
            plt.scatter(coordinates[2], coordinates[1], color="y")
            plt.scatter(coordinates[2], coordinates[3], color="y")
        return coordinates


def peak_finder(
    spec,
    spec_min=None,
    spec_max=None,
    peak_n=None,
    prominence=0.01,
    filtering=False,
    show=True,
    **kwarg,
):
    """
    find peaks within an interval

    Parameters
    ----------
    spec : spectrum, will be converted to (n,2)
    spec_min, spec_max: the defined interval
    peak_n: the numbers of peaks indices that to be returned
    prominence: float, default = 0.01
    filtering: default = False, **kwarg: window_length, polyorder
    show: default = True

    Returns
    -------
    out : ndarray
    the indices of peaks, the x values of the peaks
    """
    from scipy import signal

    spec = spec_shaper(spec)
    spec_y = spec[1].copy()

    if spec_min is not None:
        interval = np.ma.masked_inside(spec[0], spec_min, spec_max)
        spec_y = spec[1] * interval.mask

    if filtering:
        spec_y = signal.savgol_filter(spec_y, **kwarg)

    peaks_target1, _target = signal.find_peaks(spec_y, prominence=prominence)
    peaks_target2, _target2 = signal.find_peaks(-spec_y, prominence=prominence)
    peaks_target = np.hstack([peaks_target1, peaks_target2])
    peaks_target.sort()

    if peak_n is not None:
        peaks_target = peaks_target[:peak_n]

    if show:
        plt.figure()
        plt.plot(spec[0], spec[1], label="original")
        plt.plot(spec[0], spec_y, label="filtered/masked")
        plt.plot(spec[0][peaks_target], spec[1][peaks_target], "o", color="red")
        plt.legend()
    return peaks_target, spec[0][peaks_target]


def find_edge_jump(spec, show=True, prominence=0.005):
    import scipy as sp

    """
    Parameter
    --------------    
    spec: spectrum with shape (2, -1)
    return: the index of edge jump point
    """
    x = spec[0]
    dx = x[1] - x[0]
    y = spec[1]
    spl = sp.interpolate.splrep(x, y)
    ddy = sp.interpolate.splev(x, spl, der=1)
    ddy_f = (
        sp.signal.savgol_filter(ddy, 11, 2) / sp.signal.savgol_filter(ddy, 11, 2).max()
    )
    peaks = sp.signal.find_peaks(ddy_f, prominence=prominence)[0]
    point0_index = peaks[np.argsort(-ddy[peaks])[0]]
    point1_index = peaks[np.argsort(-ddy[peaks])[1]]
    mid_point_index = point0_index // 2 + point1_index // 2

    if show:
        plt.plot(spec[0], spec[1])
        plt.plot(spec[0], ddy_f * 0.5)
        plt.hlines(
            np.zeros(3000),
            xmin=spec[0].min(),
            xmax=spec[0].max(),
            linestyles=":",
            alpha=0.3,
        )
        plt.scatter(spec[0][mid_point_index], spec[1][mid_point_index], color="r")
    return mid_point_index


def calibrate_regression(
    train_spec, target_standard, peaks_train, peaks_target, order=1, sample_spec=None
):
    # transform your train & target into (n,2) array
    train = train_spec[0][peaks_train].reshape(-1, 1)
    target = target_standard[0][peaks_target].reshape(-1, 1)
    if order == 2:
        train = np.hstack((train, train**2))
        target = np.hstack((target, target**2))

    regression = linear_model.LinearRegression()
    regression.fit(train, target)
    slope = regression.coef_[0, 0]
    intecept = regression.intercept_[0]
    print(
        "slope: %s \n intecept: %s \n coeficient:%s"
        % (slope, intecept, regression.coef_)
    )
    if order == 1:
        new_x = regression.predict(train_spec[0].reshape(-1, 1))
    elif order == 2:
        predict_x = np.hstack(
            (train_spec[0].reshape(-1, 1), train_spec[0].reshape(-1, 1) ** 2)
        )
        new_x = regression.predict(predict_x)
    calibrated_data = np.array((new_x[:, 0], train_spec[1]))
    # plt.figure()
    plt.title("%d order" % order)
    plt.plot(
        calibrated_data[0],
        calibrated_data[1],
        linewidth=3,
        alpha=0.8,
        label="calibrated spec",
    )
    plt.plot(
        target_standard[0],
        target_standard[1],
        linewidth=3,
        alpha=0.6,
        label="target standard",
    )
    # predict for sample using same calibration function
    if np.any(sample_spec) is not None:
        if order == 1:
            sample_new_x = regression.predict(sample_spec[0].reshape(-1, 1))
        elif order == 2:
            sample_predict_x = np.hstack(
                (sample_spec[0].reshape(-1, 1), sample_spec[0].reshape(-1, 1) ** 2)
            )
            sample_new_x = regression.predict(sample_predict_x)

        calibrated_sample_data = np.array((sample_new_x[:, 0], sample_spec[1]))
        # add plot
        plt.plot(
            calibrated_sample_data[0],
            calibrated_sample_data[1],
            linewidth=3,
            alpha=0.8,
            label="calibrated sample",
        )
        plt.legend()
        return calibrated_data, calibrated_sample_data
    return calibrated_data


def saveh5(data, file_name="temp", folder_name="_save_h5", date=True):
    """
    save h5 file into a specific folder
    Parameter
    --------------
    data:
    file_name: default is 'temp'
    folder_name: default is '_save_h5/'
    - return None
    """
    if date is True:
        date = date_today() + "_"
    else:
        date = ""
    directory = os.getcwd() + "/" + folder_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    h5io.h5write(directory + date + "%s.h5" % file_name, var1=data[:, ::-1])
    return None


def atten_slope_corr(element, data_x, E_threshld=0, show=True):
    """
    Parameter
    --------------
    element: atomic Z number,
    data_x: data energy array,
    E_threshld = 0: set values to 1 before this energy,

    """
    import xraylib

    energy = np.arange(23, 28, 0.005)
    density = xraylib.ElementDensity(element)
    cs_photo = np.array([xraylib.CS_Total(element, E) for E in energy])

    # Normalize
    cs_photo = cs_photo * density
    cs_photo_norm = (cs_photo - cs_photo.min()) / (cs_photo.max() - cs_photo.min())

    f = interpolate.interp1d(energy, cs_photo_norm)
    slope_corr = f(data_x / 1000)
    slope_corr[data_x < E_threshld] = 1
    if show:
        plt.figure()
        plt.plot(data_x, slope_corr)
    return slope_corr


def color_gradient(c1, c2, mix=0):
    """fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    Parameter
    --------------
    c1: color 1
    c2: color 2
    mix: float, mixture of color 1 and color 2
    """
    import matplotlib as mpl

    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)



class EDXAS_Calibrate(object):
    """
    Parameter
    --------------
    train_spec: experimental data
    target_spec: synchrotron data
    train: training peaks data
    target: synchrotron data
    order: 2
    plot: True
    save_param: True
    **kwargs for plot
    """

    def __init__(
        self,
        train_spec,
        target_spec,
        train,
        target,
        order=2,
        show=True,
        save_param=True,
        **param,
    ):

        self.train_spec = train_spec
        self.target_spec = target_spec
        self.train = train.reshape(-1, 1)
        self.target = target.reshape(-1, 1)
        self.order = order
        self.show = show
        self.polynomial_features = PolynomialFeatures(degree=self.order)
        self.train = self.polynomial_features.fit_transform(self.train)
        self.target = self.polynomial_features.fit_transform(self.target)
        self.regression = linear_model.LinearRegression()
        self.regression.fit(self.train, self.target)

        self.train_spec_tf = self.polynomial_features.fit_transform(
            self.train_spec[0].reshape(-1, 1)
        )
        self.new_x = self.regression.predict(self.train_spec_tf)

        if self.show:
            plt.plot(
                self.new_x[:, 1], self.train_spec[1], label="exp standard", **param
            )
            plt.plot(
                self.target_spec[0],
                self.target_spec[1],
                label="synchrotron spec",
                **param,
            )
            plt.legend()
        if save_param:
            self.write_to_file()

    # def regression(self, order = 2):
    def sample_spec(self, sample_spec, **param):
        sample_spec_tf = self.polynomial_features.fit_transform(
            sample_spec[0].reshape(-1, 1)
        )
        sample_new_x = self.regression.predict(sample_spec_tf)
        if self.show:
            plt.plot(sample_new_x[:, 1], sample_spec[1], label="exp sample", **param)
            plt.legend()
        return sample_new_x

    def write_to_file(self):
        import csv

        with open(date_today() + "_fitting_parameters.txt", "w") as csvoutput:
            txt = csv.writer(csvoutput)
            txt.writerow((["Fitting Parameters"]))
            txt.writerow(["Date: %s" % date_today()])
            txt.writerow(["Time: %s" % time_now()])
            txt.writerow(["Regression order: %f" % self.order])
            txt.writerow(["Regression coefficient"])
            txt.writerows(self.regression.coef_)
            txt.writerow(["Regression "])
            txt.writerow(self.regression.intercept_)


def spec_shaper(spectrum):
    """
    convert spectrum array into shape (2,n)
    """
    if spectrum.shape[-1] == 2:
        return spectrum.transpose()
    return spectrum


def spec_wrapper(energy, intensity, output=(2, -1)):
    """
    wrap energy and intensity to one spectrum (2,n) array
    --------------
    - Output: default (2,n)
    """
    spectrum = np.vstack((energy, intensity))
    if output[0] == 2:
        return spectrum
    return spectrum.transpose()


def make_gif(img_list=None, extension="jpg", fps=20, file_name="movie.gif", **kwargs):
    try:
        os.mkdir("generated_gif/")
    except:
        pass
    if img_list is None:
        images = []
        for filename in glob.glob("*.%s" % extension):
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join("generated_gif", file_name), images, **kwargs)
    else:
        imageio.mimsave(os.path.join("generated_gif", file_name), img_list, **kwargs)


def make_video(img_list=None, fps=20, file_name="video.mp4", extension="jpg", **kwargs):
    try:
        os.mkdir("generated_gif/")
    except:
        pass
    writer = imageio.get_writer(
        os.path.join("generated_gif", file_name), fps=fps, **kwargs
    )

    if img_list is None:
        video = [
            writer.append_data(
                imageio.imread(file_name) for file_name in glob.glob("*.%s" % extension)
            )
        ]
    else:
        video = [writer.append_data(img_i) for img_i in img_list]
    writer.close()


def binning(arr, downscale=2):
    if len(arr.shape) == 3:
        arr_down = arr.reshape(
            arr.shape[-3],
            arr.shape[-2] // downscale,
            downscale,
            arr.shape[-1] // downscale,
            downscale,
        )
    if len(arr.shape) == 2:
        arr_down = arr.reshape(
            arr.shape[-2] // downscale, downscale, arr.shape[-1] // downscale, downscale
        )
    arr_down = np.average(arr_down, axis=(-1, -3))
    return arr_down


def spec_cropping(spec, crop_E1, crop_E2, show=False):
    """
    cropping(spec, crop_E1, crop_E2, show = False)

    can remove abnormal data from fitting
    """
    spec = spec_shaper(spec)
    crop_mask = np.where((spec[0] > crop_E1) & (spec[0] < crop_E2))[0]
    right_index = np.arange(crop_mask[0], crop_mask[0] + crop_mask.shape[0])
    difference_index = crop_mask - right_index
    crop_mask_normal = crop_mask[difference_index == 0].copy()
    cropped_spec = spec[:, crop_mask_normal]
    if show:
        plt.plot(cropped_spec[0], cropped_spec[1])
    return cropped_spec


def spec_save(spec, crop_E1=None, crop_E2=None, save_name="cropped_spec", show=True):
    """
    spec_save(spec, crop_E1=None, crop_E2=None,
              save_name='cropped_spec', show=True):
    spec: one spectrum
    crop: True or False
    crop_E1: None
    crop_E2: None
    save_name: 'cropped_spec'
    show: True

    return spectrum being saved

    """
    spec = spec_shaper(spec)
    if crop_E1 is not None:
        spec = spec_cropping(spec, crop_E1, crop_E2, show=True)
    try:
        os.mkdir("saved_spec")
    except OSError:
        pass
    file_name = os.path.join("saved_spec", "%s_%s.dat" % (date_today(), save_name))
    print("Saving %s" % (file_name))
    np.savetxt(file_name, spec.transpose())
    if show:
        plt.plot(spec[0], spec[1])
    return spec


def find_edge_pnts(spec, y_pnts, edge_min=None, edge_max=None, show=True):
    spec = spec_shaper(spec)
    if edge_min is not None:
        interval = np.ma.masked_inside(spec[0], edge_min, edge_max)
        x_pnts = np.interp(y_pnts, spec[1][interval.mask], spec[0][interval.mask])
    else:
        x_pnts = np.interp(y_pnts, spec[1], spec[0])
    if show:
        plt.plot(spec[0], spec[1])
        plt.plot(x_pnts, y_pnts, "o")
    return x_pnts


class XAS_spec(object):
    """
    XAS data analysis workflow
    """

    def __init__(self, imgs, m=None):
        self.imgs = np.array(imgs).reshape(-1, imgs.shape[-2], imgs.shape[-1])
        self.imgs_rot = None
        if m is None:
            self.m = np.ones(self.imgs[0].shape, dtype=bool)
        self.m = m
        self.data_range = self.imgs.shape[0]
        self.norm = False

    def normalize_imgs(self, imgs=None, **kwargs):
        self.norm = True
        for i in range(self.data_range):
            if imgs is None:
                self.imgs[i] = sk.exposure.rescale_intensity(self.imgs[i], **kwargs)
            else:
                imgs[i] = sk.exposure.rescale_intensity(imgs[i], **kwargs)
                return imgs

    def rotate(self, rot_angle):
        print("rotating images and masks")
        self.rot_angle = np.atleast_1d(rot_angle)
        if len(self.rot_angle) < 2:
            self.imgs = nd.rotate(
                self.imgs, self.rot_angle[0], axes=(-2, -1), reshape=False
            )
            self.m = nd.rotate(
                self.m, self.rot_angle[0], axes=(-2, -1), reshape=False, order=0
            )
            self.m[self.m < 1] = 0
            self.m = self.m.astype(bool)
        else:
            self.imgs = np.array(
                [
                    nd.rotate(self.imgs[i], self.rot_angle[i], reshape=False)
                    for i in range(self.data_range)
                ]
            )
            m = self.m[0]
            self.m = np.array(
                [
                    nd.rotate(m, self.rot_angle[i], reshape=False, order=0)
                    for i in range(self.data_range)
                ],
                dtype=bool,
            )

    def gaussian_filtering(self, sigma=2):
        self.imgs = np.array(
            [
                nd.gaussian_filter(self.imgs[i], sigma=sigma)
                for i in range(self.data_range)
            ]
        )

    def median_filtering(self, size=3, **kwargs):
        self.imgs = np.array(
            [
                nd.median_filter(self.imgs[i], size=size, **kwargs)
                for i in range(self.data_range)
            ]
        )

    def bilateral_filtering(self, **kwargs):
        """
        This function uses skimage.restoration.denoise_bilateral
        (image, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, multichannel=False, *, channel_axis=None)
        """
        self.imgs = np.array(
            [
                sk.restoration.denoise_bilateral(
                    self.imgs[i], multichannel=False, **kwargs
                )
                for i in range(self.data_range)
            ]
        )

    def show(self, alpha=0.1):
        pg.image(self.imgs * (alpha + self.m))

    def spec_generate(self, show=True, **kwargs):
        if len(self.m) < 3:
            self.spec = np.array(
                [
                    spectrum_generate(
                        np.ma.array(
                            self.imgs[i],
                            mask=~self.m,
                        ),
                        show=show,
                        **kwargs,
                    )
                    for i in range(self.data_range)
                ]
            )
        else:
            self.spec = np.array(
                [
                    spectrum_generate(
                        np.ma.array(
                            self.imgs[i],
                            mask=~self.m[i],
                        ),
                        show=show,
                        **kwargs,
                    )
                    for i in range(self.data_range)
                ]
            )

    def spec_normalize(self, x1, x2, inpt_pnts, show=True):
        self.norm_x1 = x1
        self.norm_x2 = x2
        self.norm_spec = np.array(
            [
                interpt_spec(
                    norm_spec(self.spec[i], self.norm_x1, self.norm_x2, show=show),
                    pnts=inpt_pnts,
                )
                for i in range(self.data_range)
            ]
        )
        return self.norm_spec

    def spec_cropping(self, crop_E1, crop_E2, show=True):
        self.crop_E1 = crop_E1
        self.crop_E2 = crop_E2
        self.norm_spec = np.array(
            [
                spec_cropping(self.norm_spec[i], self.crop_E1, self.crop_E2, show=show)
                for i in range(self.data_range)
            ]
        )
        return self.norm_spec

    def find_edge(self, y_pnts, x1, x2, show=True):
        self.edge_x1 = x1
        self.edge_x2 = x2
        self.edge_pnts = [
            find_edge_pnts(
                self.norm_spec[i], y_pnts, self.edge_x1, self.edge_x2, show=show
            )
            for i in range(self.data_range)
        ]
        return self.edge_pnts

    def find_peaks(
        self,
        x1,
        x2,
        peak_n,
        show=True,
        filtering=True,
        window_length=3,
        polyorder=2,
        **kwargs,
    ):
        self.peak_x1 = x1
        self.peak_x2 = x2
        self.peaks = [
            peak_finder(
                self.norm_spec[i],
                self.peak_x1,
                self.peak_x2,
                peak_n=peak_n,
                show=show,
                filtering=filtering,
                window_length=window_length,
                polyorder=polyorder,
                **kwargs,
            )
            for i in range(self.data_range)
        ]
        self.train = [
            np.insert(self.peaks[i][1], 0, self.edge_pnts[i])
            for i in range(self.data_range)
        ]

    def generate_cali_spec(self, fit):
        self.cali_spec = self.norm_spec.copy()
        for i in range(len(fit)):
            self.cali_spec[i][0] = (
                fit[i].sample_spec(self.norm_spec[i])[:, 1].transpose()
            )
        return self.cali_spec
