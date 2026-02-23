import datetime

import numpy as np


def date_today():
    return datetime.datetime.today().strftime("%Y%m%d")


def time_now():
    return datetime.datetime.now().strftime("%H%M")


class Crystal(object):

    def __init__(
        self,
        energy,
        hkl=None,
        p=0,
        crystal="Si",
        divergence=2 / 1000.0,
        R=1e8,
        d_ML=None,
    ):
        """
        Creat a Crystal class

        Parameters
        ----------
        energy : floats
        hkl : list, default: [1,1,1]
        p : floats, default: 0
            Source-to-crystal distance [m]
        crystal: str, default: 'Si'
            Or 'Ge', 'ML'(multilayer), for ML choice, please specify d_ML
        divergence: floats, default: 2/1000.
            Full divergence of X-ray (NOT half divergence!)
        R: int, default : 1e8
        d_ML: floats
            d-spacing of multilayer [angstrom]


        """
        self.energy = energy
        self.wavelength = 12.3984428 / energy  # in angstrom
        self.hkl = hkl
        self.crystal = crystal
        self.divergence = divergence
        self.p = p
        self.R = R
        if self.crystal == "Si":
            self.lattice_cons = 5.431
        elif self.crystal == "Ge":
            self.lattice_cons = 5.65
        elif self.crystal == "ML":
            self.d = d_ML  # in angstrom
        self.printlst = {
            "00_header_top": "=" * 50,
            "01_Energy_hkl": "Energy is %.2f KeV and %s(hkl)is %s"
            % (self.energy, self.crystal, str(self.hkl)),
            "02_header_bottom": "=" * 50,
        }

    @property
    def theta0(self):
        if self.crystal != "ML":
            self.d = self.lattice_cons / np.sqrt(
                self.hkl[0] ** 2 + self.hkl[1] ** 2 + self.hkl[2] ** 2
            )  # in angstrom
        theta0 = np.arcsin(self.wavelength / (2.0 * self.d))
        self.printlst["03_theta0"] = "theta0 : %.3f rad, %.2f degree" % (
            theta0,
            np.rad2deg(theta0),
        )
        return theta0

    @property
    def energy_spread_flat(self):
        theta_Emin = self.theta0 + self.divergence / 2.0
        theta_Emax = self.theta0 - self.divergence / 2.0

        self.E_min = 12.39847 / (2 * self.d * np.sin(theta_Emin))
        self.E_max = 12.39847 / (2 * self.d * np.sin(theta_Emax))

        energy_spread = self.energy * self.divergence / np.tan(self.theta0)
        self.printlst["04_energy_spread_flat"] = (
            "Energy spread (flat crystal): %.2f eV" % (energy_spread * 1000)
        )
        return energy_spread, [theta_Emin, theta_Emax]

    def det2cen_calc(self, dety):
        """
        dy: distance of detector along y-direction (X-ray propagation direction), in mm
        """
        self.d_det2cen = dety * np.tan(2 * self.theta0)
        self.printlst["06_det2cen_distance"] = (
            "detector position to the center is %.3f mm" % (self.d_det2cen)
        )
        return self.d_det2cen

    def angle_to_energy_spread_calc(self, delta_theta):
        deltaE = delta_theta * 1 / np.tan(self.theta0) * self.energy
        print(
            "Energy spread for %.f microrad is: %.2f eV"
            % (delta_theta * 1e6, deltaE * 1000)
        )
        return deltaE

    @property
    def beam_size(self):
        beam_size = 2 * self.p * 1000 * np.tan(self.divergence / 2)  # + 50/1000 # in mm
        self.printlst["05_beam_size_transverse"] = (
            "Transverse beam size at a distance of %.3f m is %.3f mm"
            % (self.p, beam_size)
        )
        return beam_size

    def type_writer(self, curvature=False, crystal_rotation=False, sort=True):
        if sort:
            printlst = sorted(self.printlst)
        else:
            printlst = self.printlst
        for items in printlst:
            print(self.printlst[items])


class Laue_Crystal(Crystal):

    def __init__(
        self,
        energy,
        hkl=[1, 1, 1],
        p=0,
        crystal="Si",
        divergence=2 / 1000.0,
        R=1e8,
        surface_hkl=[1, 0, 0],
        Poisson_ratio=0.22,
        T=200 * 1e-6,
        condition="lower",
    ):
        super().__init__(energy, hkl, p, crystal, divergence, R)
        self.surface_hkl = surface_hkl
        self.Poisson_ratio = Poisson_ratio
        self.T = T
        self.condition = condition
        self.printlst["00_header_top_01"] = "-" * 19 + "LAUE CRYSTAL" + "-" * 19

    #     # def assy_angle(self):
    #     #     a = 1./self.hkl[0]
    #     #     b = 1./np.sqrt(2)
    #     #     assy_angle = np.pi * 0.5 - np.arctan(a/b)
    #     #     return assbby_angle

    @property
    def assy_angle(self):
        plane = np.array(self.hkl)
        normal = np.array(self.surface_hkl)
        assy_angle = np.pi / 2 - np.arccos(
            np.dot(plane, normal) / np.linalg.norm(plane) / np.linalg.norm(normal)
        )
        self.printlst["Laue01_assymetry_cut"] = (
            "Assymetric cut is : %.2f degree" % np.rad2deg(assy_angle)
        )
        return assy_angle

    @property
    def crystal_rotation(self):
        if self.condition == "upper":
            crystal_rotation = 0.5 * np.pi - (self.assy_angle + self.theta0)
        elif self.condition == "lower":
            crystal_rotation = 0.5 * np.pi - (self.assy_angle - self.theta0)
        self.printlst["Laue02_crystal_rotation"] = (
            "crystal rotation: %s case,  %.6f rad, %.3f degree \n"
            % (self.condition, crystal_rotation, np.rad2deg(crystal_rotation))
        )
        return crystal_rotation

    @property
    def foot_print(self):
        return self.beam_size / np.sin(self.crystal_rotation)  # in mm

    @property
    def energy_spread_bent(self):
        delta_bent_crystal = self.foot_print / 1000 / self.R  # in radians
        delta_theta = self.divergence + delta_bent_crystal
        deltaE = delta_theta * 1 / np.tan(self.theta0) * self.energy
        self.printlst["Laue03_energy_spread_bent"] = (
            "The energy spread for the crystal with bending radius of %.2f is %.2f eV"
            % (self.R, deltaE * 1000)
        )
        return deltaE

    @property
    def energy_spread_bent_esrf(self):
        delta_theta = (
            self.beam_size / 1000 / np.cos(self.theta0) / 2 * (1 / self.p - self.f_g)
        )  # beamsize in mm, this is only true for symmetric crystal
        deltaE = delta_theta * 1 / np.tan(self.theta0) * self.energy
        self.printlst["Laue04_energy_spread_bent_esrf"] = (
            "ESRF energy spread for the crystal with bending radius of %.2f is %.2f eV \n"
            % (self.R, deltaE * 1000)
        )
        return deltaE

    def curvature_calc(self, exp_spread=0):
        self.diff_spread = self.energy_spread_flat[0] - exp_spread
        self.diff_delta = self.diff_spread * np.tan(self.theta0) / self.energy
        self.crys_curvature = self.foot_print / self.diff_delta / 1000  # in m
        self.printlst["Laue05_01_diff_spread"] = (
            "-" * 50
            + "\n"
            + "difference between exp and theoretical is: %f eV"
            % (self.diff_spread * 1000)
        )
        self.printlst["Laue05_02_diff_delta"] = (
            "curvature induced delta is: %f mrad" % (self.diff_delta * 1000)
        )
        self.printlst["Laue05_03_beam_size"] = (
            "Beam size on the crystal at distance of %.2f m is: %.2f mm"
            % (self.p, self.beam_size)
        )
        self.printlst["Laue05_04_foot_print"] = (
            "footprint at rotation angle %.2f is: %.2f cm"
            % (self.crystal_rotation, self.foot_print)
        )
        self.printlst["Laue05_05_crys_curvature"] = (
            "Crystal curvature is: %.3f m" % self.crys_curvature + "\n" + "-" * 50
        )
        return self.crys_curvature

    def single_ray_focus_calc(self):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0
        self.f_p = (
            self.R
            * np.sin(2 * theta0)
            / (
                2 * np.sin(self.assy_angle + theta0)
                + (1 + self.Poisson_ratio)
                * np.sin(2 * self.assy_angle)
                * np.cos(self.assy_angle + theta0)
            )
        )
        print()
        self.printlst["Laue06_single_ray_focus"] = "Single-ray focus: %.5f" % self.f_p
        return self.f_p

    def geometric_focus_calc(self, assy_angle=None):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0

        if assy_angle is None:
            assy_angle = self.assy_angle
        self.f_g = np.cos(assy_angle - theta0) / (
            2 / self.R + np.cos(assy_angle + theta0) / self.p
        )
        self.printlst["Laue07_01_geometric_focus"] = "Geometric focus: %.5f" % self.f_g
        self.printlst["Laue07_02_geometric_focus"] = (
            "Geometric along incident ray direction: %.5f"
            % (self.f_g * np.cos(2 * self.theta0))
        )
        return self.f_g

    def Laue_size_calc(self, det2crys_d):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0

        # delta_bent_crystal = self.foot_print/1000/self.R
        # self.Laue_size = self.beam_size/1000 - ((det2crys_d - self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 - 0.5* self.divergence+ delta_bent_crystal)) + ((det2crys_d + self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 + 0.5* self.divergence - delta_bent_crystal))
        # vertical_x1 =  self.beam_size/1000/2 + (det2crys_d + self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 - self.divergence/2)
        # vertical_x2 =  -self.beam_size/1000/2 + (det2crys_d - self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 + self.divergence/2)
        # self.Laue_size = vertical_x1 - vertical_x2
        # print(vertical_x1)
        # print(vertical_x2)

        self.det_position = det2crys_d * np.tan(2 * self.theta0)
        # print('middle ray position:', det2crys_d * np.tan(2*self.theta0))
        exit_beam_size_vertical = (
            (np.cos(self.assy_angle - theta0) / np.cos(2 * theta0))
            * self.foot_print
            / 1000
        )
        # print(self.foot_print)
        # print(np.cos(self.assy_angle + theta0)/np.cos(2 * theta0))
        # print(self.exit_beam_size_vertical)
        self.Laue_size = (self.f_g - det2crys_d) / self.f_g * exit_beam_size_vertical
        self.printlst["Laue08_Laus_size"] = (
            "Laue beam size on the detector:%.2f mm \n" % (self.Laue_size * 1000)
        )
        return self.Laue_size

    def magic_condition_calc(self):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0
        magic_coeff = (
            (2 + self.Poisson_ratio) * np.tan(self.assy_angle)
            + self.Poisson_ratio * np.tan(theta0) * np.tan(self.assy_angle) ** 2
            + np.tan(self.assy_angle) ** 2 * np.tan(self.assy_angle)
            - np.tan(theta0)
        )
        self.printlst["Laue09_magic_condition_coff"] = (
            "Magic condition coeff:%.2f \n" % (magic_coeff)
        )
        return magic_coeff

    def Borrmann_calc(self):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0
        # thesis
        Borrmann_angle = (self.T / self.R) * (
            np.tan(self.assy_angle + theta0)
            + np.sin(self.assy_angle)
            * np.cos(self.assy_angle)
            * (1 + self.Poisson_ratio)
            - np.tan(theta0)
            * (
                np.cos(self.assy_angle) ** 2
                - self.Poisson_ratio * np.sin(self.assy_angle) ** 2
            )
        )
        # Borrmann_angle = (self.T/self.R) * (np.tan(self.assy_angle - theta0) + np.sin(self.assy_angle) * np.cos(self.assy_angle) * (1+self.Poisson_ratio) + np.tan(theta0)*(np.cos(self.assy_angle)**2 - self.Poisson_ratio * np.sin(self.assy_angle)**2))
        self.printlst["Laue10_01_Borrmann_fan"] = "Borrmann fan:%.3f microrad" % (
            Borrmann_angle * 1e6
        )
        self.printlst["Laue10_02_Borrmann_fan"] = (
            "Corresponding energy spread:%.3f eV"
            % (self.energy_resolution_calc(Borrmann_angle))
        )
        return Borrmann_angle

    def Borrmann_flat_calc(self):
        Borrmann_width = 2 * self.T * np.sin(self.theta0)  # in meter
        self.printlst["Laue10_00_Borrmann_fan"] = (
            "Borrmann fan (flat_crystal):%.3f microm" % (Borrmann_width * 1e6)
        )
        # deltaE = self.energy_resolution_calc(Borrmann_angle)
        return Borrmann_width

    def Borrmann_bent_calc(self):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0
        Borrmann_width = self.T * np.sin(2 * self.theta0) / np.cos(theta0)
        # print('Borrmann fan width (flat crystal): %.3f microm'%(Borrmann_width*1e6))
        # deltaE = self.energy_resolution_calc(Borrmann_angle)
        return Borrmann_width

    def rotation_spread(self, rotation_angle=0):
        rotation_angle = np.deg2rad(rotation_angle)
        new_theta0 = self.theta0 + rotation_angle
        rotation_spread = self.energy * (rotation_angle) / np.tan(new_theta0)
        self.printlst["Laue12_rotation_spread"] = "rotation spread is: %f eV" % (
            1000 * rotation_spread
        )
        return rotation_spread

    def energy_resolution_calc(self, angular_spread):
        E_spread = angular_spread * self.energy / np.tan(self.theta0)
        self.printlst["Laue13_energy_resolution"] = (
            "The energy spread for angular spread of %.03f microrad is: %.03f eV"
            % (angular_spread * 1e6, E_spread * 1000)
        )
        return E_spread


class Bragg_Crystal(Crystal):

    def __init__(
        self,
        energy,
        hkl=None,
        p=0,
        crystal="Si",
        d_ML=None,
        divergence=2 / 1000.0,
        R=1e8,
        assy_angle=0,
        condition="upper",
    ):
        super().__init__(energy, hkl, p, crystal, divergence, R, d_ML)
        self.printlst["00_header_top_01"] = "-" * 17 + " BRAGG CRYSTAL " + "-" * 17
        self.assy_angle = assy_angle
        self.condition = condition

    @property
    def crystal_rotation(self):
        if self.condition == "upper":
            crystal_rotation = self.assy_angle + self.theta0
        elif self.condition == "lower":
            crystal_rotation = self.assy_angle - self.theta0
        self.printlst["Laue02_Bragg_rotation"] = (
            "crystal rotation: %s case,  %.6f rad, %.3f degree \n"
            % (self.condition, crystal_rotation, np.rad2deg(crystal_rotation))
        )
        return crystal_rotation

    @property
    def foot_print(self):
        self.printlst["Laue03_foot_print"] = "footprint on the crystal: %.3f mm \n" % (
            self.beam_size / np.sin(self.crystal_rotation)
        )
        return self.beam_size / np.sin(self.crystal_rotation)  # in mm

    def energy_spread_bent_calc(self, foot_print_set=None):
        if foot_print_set is None:
            foot_print_set = self.foot_print
        else:
            self.printlst["Bragg04_01_energy_spread_bent"] = (
                "Set foot_print value: %.3f mm" % (foot_print_set)
            )
        delta_bent_crystal = foot_print_set / 1000 / self.R  # in radians
        #         delta_bent_crystal = 2 *np.arcsin(self.foot_print/1000/(2 * self.R))
        delta_theta = self.divergence - delta_bent_crystal
        deltaE = delta_theta * 1 / np.tan(self.theta0) * self.energy
        self.energy_spread_bent = deltaE
        self.printlst["Bragg04_delta_theta_bent"] = (
            "Delta theta of the bent crystal is %.3f mrad, %.3f deg"
            % (delta_theta * 1000, np.rad2deg(delta_theta))
        )
        self.printlst["Bragg04_energy_spread_bent"] = (
            "The energy spread for the crystal with bending radius of %.2f is %.5f eV"
            % (self.R, deltaE * 1000)
        )
        return deltaE

    @property
    def energy_spread_bent_esrf(self):
        self.geometric_focus_calc(self.assy_angle)
        delta_theta = (
            self.foot_print
            * np.sin(self.theta0)
            / 1000
            / 2
            * (1 / self.p - 1 / self.f_g)
        )  # beamsize in mm
        deltaE = delta_theta * 1 / np.tan(self.theta0) * self.energy
        self.printlst["Bragg04_energy_spread_bent_esrf"] = (
            "ESRF energy spread for the crystal with bending radius of %.2f is %.2f eV \n"
            % (self.R, deltaE * 1000)
        )
        return deltaE

    def curvature_calc(self, exp_spread=0):
        self.diff_spread = self.energy_spread_flat[0] - exp_spread
        self.diff_delta = self.diff_spread * np.tan(self.theta0) / self.energy
        self.crys_curvature = self.foot_print / self.diff_delta / 1000  # in m
        self.printlst["Laue05_01_diff_spread"] = (
            "-" * 50
            + "\n"
            + "difference between exp and theoretical is: %f eV"
            % (self.diff_spread * 1000)
        )
        self.printlst["Laue05_02_diff_delta"] = (
            "curvature induced delta is: %f mrad" % (self.diff_delta * 1000)
        )
        self.printlst["Laue05_03_beam_size"] = (
            "Beam size on the crystal at distance of %.2f m is: %.2f mm"
            % (self.p, self.beam_size)
        )
        self.printlst["Laue05_04_foot_print"] = (
            "footprint at rotation angle %.2f is: %.2f cm"
            % (self.crystal_rotation, self.foot_print)
        )
        self.printlst["Laue05_05_crys_curvature"] = (
            "Crystal curvature is: %.3f m" % self.crys_curvature + "\n" + "-" * 50
        )
        return self.crys_curvature

    def geometric_focus_calc(self, assy_angle=None):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0
        if assy_angle is None:
            assy_angle = self.assy_angle
        self.f_g = np.sin(theta0 - assy_angle) / (
            2 / self.R - np.sin(theta0 + assy_angle) / self.p
        )
        self.printlst["Bragg07_01_geometric_focus"] = (
            "Geometric focus: %.5f m" % self.f_g
        )
        self.printlst["Bragg07_02_geometric_focus"] = (
            "Geometric along incident ray direction: %.5f m"
            % (self.f_g * np.cos(2 * self.theta0))
        )
        return self.f_g

    def Bragg_size_calc(self, det2crys_d=None, dety=None):
        if self.condition == "upper":
            theta0 = self.theta0
        elif self.condition == "lower":
            theta0 = -self.theta0

        # delta_bent_crystal = self.foot_print/1000/self.R
        # self.Laue_size = self.beam_size/1000 - ((det2crys_d - self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 - 0.5* self.divergence+ delta_bent_crystal)) + ((det2crys_d + self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 + 0.5* self.divergence - delta_bent_crystal))
        # vertical_x1 =  self.beam_size/1000/2 + (det2crys_d + self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 - self.divergence/2)
        # vertical_x2 =  -self.beam_size/1000/2 + (det2crys_d - self.beam_size/1000/2 * 1/np.tan(self.crystal_rotation)) * np.tan(2*self.theta0 + self.divergence/2)
        # self.Laue_size = vertical_x1 - vertical_x2
        # print(vertical_x1)
        # print(vertical_x2)
        if dety is not None:
            det2crys_d = dety / np.cos(2 * self.theta0)
        self.det_position = det2crys_d * np.tan(2 * self.theta0)
        # print('middle ray position:', det2crys_d * np.tan(2*self.theta0))

        # compared to Laue, cos(rotation_angle) --> sin()
        # and no need to be divided by Cos(2theta)
        exit_beam_size_vertical = (
            np.sin(theta0 - self.assy_angle) * self.foot_print / 1000
        )

        # print(self.foot_print)
        # print(np.cos(self.assy_angle + theta0)/np.cos(2 * theta0))
        # print(self.exit_beam_size_vertical)
        self.Bragg_size = (self.f_g - det2crys_d) / self.f_g * exit_beam_size_vertical
        self.Bragg_size_vertical = self.Bragg_size / np.cos(2 * theta0)
        self.printlst["Laue08_01_Bragg_size"] = (
            "Bragg beam size perpendicular to travel: %.2f mm"
            % (self.Bragg_size * 1000)
        )
        self.printlst["Laue08_02_Bragg_size_vertical"] = (
            "Bragg beam size perpendicular to incident ray:%.2f mm \n"
            % (self.Bragg_size_vertical * 1000)
        )

        return self.Bragg_size, self.Bragg_size_vertical

    def rotation_spread(self, rotation_angle=0):
        rotation_angle = np.deg2rad(rotation_angle)
        new_theta0 = self.theta0 + rotation_angle
        rotation_spread = self.energy * (rotation_angle) / np.tan(new_theta0)
        self.printlst["Laue12_rotation_spread"] = "rotation spread is: %f eV" % (
            1000 * rotation_spread
        )
        return rotation_spread

    def energy_resolution_calc(self, pixel_size=75):
        self.energy_resolution = (
            self.energy_spread_bent * 1000 / (self.Bragg_size / (pixel_size / 1e6))
        )
        self.printlst["Laue13_energy_resolution"] = (
            "Energy resolution (pixel size = %.1f): %.2f eV/pixel"
            % (pixel_size, self.energy_resolution)
        )
        return self.energy_resolution


"""
# demo of Bragg crystal
if __name__ == '__main__':
    crystal =Bragg_Crystal(11,
                           hkl = [1,1,1],
                           p = 0.42743, 
                           crystal = 'Si', 
                           divergence = 1/1000., 
                           R = -0.5,
                           assy_angle=np.deg2rad(4.5),
                           condition = 'upper'
                          )
    crystal.theta0
    crystal.crystal_rotation;
    crystal.foot_print;
    crystal.energy_spread_flat
    crystal.energy_spread_bent_calc()
    crystal.energy_spread_bent_esrf
    # crystal.curvature_calc(exp_spread = 400)
    crystal.geometric_focus_calc()
    # crystal.Borrmann_calc()
    # crystal.Borrmann_bent_calc()
    # crystal.magic_condition_calc()
    crystal.Bragg_size_calc(det2crys_d = 2.5639)
    # crystal.rotation_spread(2)
    crystal.energy_resolution_calc(75)
    crystal.type_writer(sort=False)

# demo of general crystal
if __name__ == '__main__':
    crystal_111_para = {"hkl" :[1,1,1],
                        "p" : 0.43, 
                        "crystal" : "Si", 
                        "divergence" : 1/1000., 
                        "R" : -0.5}

    crystal = Crystal(11.5, **crystal_111_para)
    crystal.theta0;
    crystal.energy_spread_flat;
    crystal.beam_size;
    crystal.type_writer()

# demo of Laue crystal
if __name__ == '__main__':

    crystal =Laue_Crystal(24.5, 
                           hkl = [1,1,1],
                           surface_hkl=[1,0,0],
                           p = 3.8, 
                           crystal = 'Si', 
                           divergence = 4/1000., 
                           T = 0.0002, # m
                           R = -13,)
    crystal.assy_angle;
    crystal.crystal_rotation;
    crystal.energy_spread_bent
    crystal.curvature_calc(exp_spread = 400)
    crystal.single_ray_focus_calc()
    crystal.geometric_focus_calc()
    crystal.Borrmann_calc()
    crystal.Borrmann_bent_calc()
    crystal.magic_condition_calc()
    crystal.Laue_size_calc(det2crys_d = 0.4)
    crystal.rotation_spread(2)
    crystal.energy_resolution_calc(0.2)
    crystal.type_writer()    
"""
