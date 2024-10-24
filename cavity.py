import numpy as np
#from gaussian_beam import GaussianBeam


def cavity_beam_waist(wlength, L, R1, R2):
    prefactor = (wlength/np.pi)**2
    numerator = L * (R1 - L) * (R2 - L) * (R1 + R2 - L)
    denominator = (R1 + R2 - 2 * L)**2
    return prefactor * numerator / denominator

class Cavity:
    pass

class SymmetricCavity(Cavity):
    def __init__(self, L: float, R: float):
        self.L = L
      
        self.R = R
        
    def get_fundamental_mode(self, n):
        # beam shape
        zR = np.sqrt(self.L/2 * (self.R - self.L/2))

        # beam wavenumber
        k = 2 * np.arctan(self.L/(2*zR))/self.L + np.pi/self.L * n
        
        # phi
        phi = -np.pi/2 * n      
        
        def fundamental_mode(r, z):
            w0 = np.sqrt(2 * zR / k)
            w_z = w0 * np.sqrt((1 + (z/zR)**2))
            R_z = (zR**2 + z**2)/z
            fixed = w0/w_z / zR * np.exp(-(r/w_z)**2)
            oscillate = 2 * np.sin(k * z + phi - np.arctan2(z, zR) + k * r**2 / (2*R_z))
            return (fixed * oscillate)**2
        
        
        return fundamental_mode


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Arc
    L = 2e-6
    
    punit = 1e-6
    
    R=L
    c = SymmetricCavity(L=L, R=R)
    
    mode = c.get_fundamental_mode(n=11)
    
    zview = L/2
    xview = L/3
    r = np.linspace(-xview, xview, 1000)
    z = np.linspace(-zview, zview, 1000)
    
    rr, zz = np.meshgrid(r, z, indexing='ij')
    standing = mode(rr, zz)
    
    figsize = np.r_[1, xview/zview] * 6
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.pcolormesh(zz / punit, rr / punit, (standing), cmap='Reds')
    
    # add mirrors
    angle = 15
    linewidth=4
    m1 = Arc(np.r_[L/2-R, 0]/punit, width=2*R/punit, height=2*R/punit, theta1=-angle, theta2=angle, linewidth=linewidth)
    m2 = Arc(np.r_[-(L/2-R), 0]/punit, width=2*R/punit, height=2*R/punit, theta1=-(180 + angle), theta2=(180 + angle), linewidth=linewidth)
    
    ax.add_patch(m1)
    ax.add_patch(m2)
    
    #ax.set_xlim(-L/2/punit, L/2/punit)
    
    ax.set_xlabel('z-axis  [$\mathrm{\mu m}$]')
    ax.set_ylabel('x-axis  [$\mathrm{\mu m}$]')
    
    plt.show()
