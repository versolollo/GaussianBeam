import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt

plt.style.use("~/styling.mplstyle")


class GaussianBeam:
    def __init__(self, zr=2, k=None, wlength=None, z0=0):
        if k is None and wlength is None:
            raise Exception("one between k and wlength needs to be specified.")
        elif k is not None and wlength is not None:
            raise Exception(
                "Can't specify both k and wlength as they refer to the same quantity."
            )
        self.z0 = z0
        self.zr = zr
        if wlength is None:
            wlength = 2 * np.pi / k
        self.wlength = wlength
        self.omega = 3e8 * self.k

    @property
    def NAe2(self):
        angle = np.arctan(np.sqrt(self.wlength / (self.zr * np.pi)))
        return np.sin(angle)

    @NAe2.setter
    def NAe2(self, new_NAe2):
        angle = np.arcsin(new_NAe2)
        self.zr = self.wlength / (np.tan(angle) ** 2 * np.pi)

    @property
    def k(self):
        return 2 * np.pi / self.wlength

    @property
    def w0(self):
        return np.sqrt(2 * self.zr / self.k)

    @w0.setter
    def w0(self, w0_new):
        zr_new = w0_new**2 * self.k / 2
        self.zr = zr_new

    def q(self, z):
        q = z - self.z0 - 1j * self.zr
        return q

    def w(self, z):
        """half beam diameter"""
        return self.w0 * np.sqrt(1 + ((z - self.z0) / self.zr) ** 2)

    def width(self, z):
        """width of beam"""
        return 2 * self.w(z)

    def find_diameter(self, diameter):
        z = np.sqrt(((diameter / self.w0) ** 2 - 1)) * self.zr + self.z0

        if z == np.nan:
            print("Beam never has this diameter")
            return None
        return z

    def __call__(self, r, z, t=0):

        return np.exp(1j * (self.k * (z - self.z0) - self.omega * t)) * np.exp(
            1j * self.k * r**2 / (2 * self.q(z))
        )

    def copy(self):
        return GaussianBeam(
            zr=-np.imag(self.q), z0=self.z0, wlength=self.wlength
        )

    def apply_lens(self, f, z_lens):
        lens_matrix = np.array([[1, 0], [-1 / f, 1]])
        vec_temp = lens_matrix @ np.array([self.q(z_lens), 1])
        new_q = vec_temp[0] / vec_temp[1]
        return GaussianBeam(
            zr=-np.imag(new_q),
            k=self.k,
            z0=-(np.real(new_q)),  # z0=-(np.real(new_q) - z_lens),
        )

    def plot(self, ax=None, zstart=None, zend=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if zstart is None:
            zstart = self.z0 - 2 * self.zr
        if zend is None:
            zend = self.z0 + 2 * self.zr
        zarr = np.linspace(zstart, zend, 100)
        ax.fill_between(
            zarr,
            -self.width(zarr) / 2,
            self.width(zarr) / 2,
            alpha=0.3,
            **kwargs,
        )

    def fit_data(self, zvalues, diam_13_5, p0):
        rad_13_5 = diam_13_5 / 2

        def fit_func(z, z0, zr):
            w0 = np.sqrt(self.wlength * zr / np.pi) * np.sqrt(
                1 + (z - z0) ** 2 / zr**2
            )
            return w0

        fit = sp.optimize.curve_fit(fit_func, zvalues, rad_13_5, p0=p0)
        g = self.copy()
        g.z0 = fit[0][0]
        g.zr = fit[0][1]
        return g

    def _get_parameter_string(self):
        return (
            f"\n{self.z0:.2e}\t{self.w0:.2e}\t{self.zr:.2e}\t{self.NAe2:.2e}"
        )

    def __repr__(self):
        repr_str = f"Gaussian Laser Beam @ {self.wlength*1e9:.1f}nm\n---z0---\t---w0---\t---zr---\t--NAe2--"
        repr_str += self._get_parameter_string()
        return repr_str


class LaserBeam:
    def __init__(self, g: GaussianBeam):
        self.gaussian_beams = [g]
        self.z_partitions = []

    def apply_lens(self, f, z_lens):

        new_beam = self.gaussian_beams[-1].apply_lens(f, z_lens)
        self.z_partitions.append(z_lens)
        self.gaussian_beams.append(new_beam)

    def plot_beam(self, ax=None, zstart=None, zstop=None):
        if ax is None:
            ax = plt.gca()
        if len(self.z_partitions) != 0:
            for i_beam in range(len(self.z_partitions) + 1):
                beam = self.gaussian_beams[i_beam]
                if i_beam == 0:
                    next_part = self.z_partitions[i_beam]
                    if zstart is None:
                        zstart = next_part - 2 * beam.zr
                    beam.plot(ax=ax, zstart=zstart, zend=next_part)
                elif i_beam == len(self.z_partitions):
                    prev_part = self.z_partitions[i_beam - 1]
                    if zstop is None:
                        zstop = prev_part + 2 * beam.zr
                    beam.plot(ax=ax, zstart=prev_part, zend=zstop)
                else:
                    prev_part = self.z_partitions[i_beam - 1]
                    next_part = self.z_partitions[i_beam]
                    beam.plot(ax=ax, zstart=prev_part, zend=next_part)
        else:
            self.gaussian_beams[0].plot(ax=ax)

    def __repr__(self):
        repr_str = f"Gaussian Laser Beam @ {self.gaussian_beams[0].wlength}nm\n#\t---z0---\t---w0---\t---zr---\t--NAe2--"
        for i_b, b in enumerate(self.gaussian_beams):
            b_str = (
                f"\n{i_b}\t{b.z0:.2e}\t{b.w0:.2e}\t{b.zr:.2e}\t{b.NAe2:.2e}"
            )
            repr_str += b_str
        return repr_str

    def __getitem__(self, i):
        return self.gaussian_beams[i]


def fiber_coupler(lam=461e-9, NAe2=0.01, fc=7e-3):
    g = GaussianBeam(zr=1, wlength=lam)
    g.NAe2 = NAe2
    beam = LaserBeam(g)
    beam.add_lens(fc, fc)

    plt.figure()
    ax = plt.gca()
    beam.plot_beam(ax, zstart=0, zstop=30e-3)
    ax.set_xlabel(r"Axial Distance $[m]$")
    plt.show()
    print(beam)
    return beam


if __name__ == "__main__":
    """
    g = GaussianBeam(zr=5, wlength=461e-9)
    g.NAe2 = 0.07
    lbeam = LaserBeam(g)
    lbeam.add_lens(10e-3, 10e-3)
    lbeam.plot_beam(zstop=30e-3, zstart=0)
    lbeam"""

    g = GaussianBeam(zr=5, wlength=421e-9)
    g.NAe2 = 0.07
    lbeam = LaserBeam(g)
    lbeam.apply_lens(8e-3, 8e-3)
    # beam = fiber_coupler(NAe2=0.11, fc=6.2e-3)
    # beam = fiber_coupler(NAe2=0.011, fc=20e-3)
    beam = lbeam[1]
    zlist = np.linspace(0, 500e-3, 1000)
    wlist = beam.w(zlist) * 2

    plt.figure()
    plt.plot(zlist * 1000 / 25, wlist)
    plt.show()

# This is wrong
