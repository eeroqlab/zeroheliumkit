import numpy as np

from math import pi, sinh
from scipy.special import ellipk

from tabulate import tabulate

from .constants import speedoflight, mu_0, epsilon_0


GHz = 1e9
MHz = 1e6
kHz = 1e3

cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

class CPW_params():
    def __init__(self,
                 eps_substrate: float=11,
                 width: float=8*um,
                 gap: float=4*um,
                 height_substrate: float=0.5*mm,
                 sheet_inductance: float=0):
        
        self.eps_substrate = eps_substrate
        self.width = width
        self.gap = gap
        self.height_substrate = height_substrate
        self.sheet_inductance = sheet_inductance
        self.calculate_params()
        self.length = 0
        self.frequency = 0

    def calculate_params(self) -> None:
        """ see Simons "Coplanar Waveguides", page 20
        
        """

        a = self.width
        b = self.width + 2 * self.gap
        h = self.height_substrate

        k0 = float(a)/float(b)
        k0p = np.sqrt(1 - k0**2)

        k1 = sinh(pi * a/(4 * h)) / sinh(pi * b/(4 * h))
        k1p = np.sqrt(1 - k1**2)

        Ktwid = ellipk(k0p) * ellipk(k1)/(ellipk(k0) * ellipk(k1p))

        Cair = 4 * epsilon_0 * ellipk(k0)/ellipk(k0p)
        Cdiel = 2 * epsilon_0 * (self.eps_substrate - 1) * ellipk(k1)/ellipk(k1p)

        self.eps_eff =  1 + (self.eps_substrate - 1) * Ktwid / 2

        self.L = (mu_0/4) * ellipk(k0p)/ellipk(k0) + self.sheet_inductance / self.width
        self.C = Cair + Cdiel

        self.Z = np.sqrt(self.L/self.C)
        self.Z2 = 60 * np.pi / (np.sqrt(self.eps_eff) * (ellipk(k0)/ellipk(k0p) + ellipk(k1)/ellipk(k1p)))
        self.phase_velocity = 1/np.sqrt(self.L * self.C)
    
    def resonator_frequency(self,
                            length: float=1*cm,
                            resonator_type: float=0.5,
                            harmonic: int=0,
                            unit: float=GHz,
                            print_status: bool=True) -> float:


        if (resonator_type==0.25): 
            length_factor = 0.25 * (2*harmonic + 1)
        elif (resonator_type==0.5):                      
            length_factor = 0.5 * (harmonic + 1)
        else:
            raise Exception("choose 1/4 or 1/2 resonator type")
        
        bare_frequency = length_factor * self.phase_velocity/length
        parameters = {
                        "f0, GHz": round(bare_frequency/unit, 3),
                        "length, mm": round(length/mm, 2),
                        "width, um": round(self.width/um, 2),
                        "gap, um": round(self.gap/um, 2),
                        "eps ": round(self.eps_substrate),
                        "eps_eff": round(self.eps_eff, 2),
                        "impedance, Ohm": round(self.Z, 2),
                        "L, nH/m": round(self.L * 1e9, 3),
                        "C, pF/m":round(self.C* 1e12, 3)
                      }
        if print_status:
            print(tabulate([parameters.keys(), parameters.values()]))
        
        self.length = length
        self.frequency = bare_frequency
        #return bare_frequency/unit

    def resonator_length(self,
                         resonator_frequency: float=5*GHz,
                         resonator_type: float=0.5,
                         harmonic: int=0,
                         unit: float=mm,
                         print_status: bool=True) -> None:
        length = resonator_type * self.phase_velocity / resonator_frequency
        parameters = {
                        "f0, GHz": round(resonator_frequency/GHz, 3),
                        "length, mm": round(length/mm, 3),
                        "width, um": round(self.width/um, 2),
                        "gap, um": round(self.gap/um, 2),
                        "eps ": round(self.eps_substrate),
                        "eps_eff": round(self.eps_eff, 2),
                        "impedance, Ohm": round(self.Z, 2),
                        "L, nH/m": round(self.L * 1e9, 3),
                        "C, pF/m":round(self.C* 1e12, 3)
                      }
        if print_status:
            print(tabulate([parameters.keys(), parameters.values()]))

        self.length = length
        self.frequency = resonator_frequency
        #return length



class CPW_conductorbacked():
    def __init__(self,
                 eps_substrate: float=11,
                 width: float=8*um,
                 gap: float=4*um,
                 height_substrate: float=0.5*mm,
                 sheet_inductance: float=0):
        
        self.eps_substrate = eps_substrate
        self.width = width
        self.gap = gap
        self.height_substrate = height_substrate
        self.sheet_inductance = sheet_inductance
        self.calculate_params()
        self.length = 0
        self.frequency = 0

    def calculate_params(self) -> None:
        """ see Simons "Coplanar Waveguides", page 20
        
        """
        a = self.width
        b = self.width + 2 * self.gap
        h = self.height_substrate

        k0 = float(a)/float(b)
        k0p = np.sqrt(1 - k0**2)

        k3 = tanh(pi * a/(4 * h))/  tanh(pi * b/(4 * h))
        k3p = np.sqrt(1 - k3**2)

        Ctop_air = 2 * epsilon_0 * ellipk(k0) / ellipk(k0p)
        Cbottom_air = 2 * epsilon_0 * ellipk(k3) / ellipk(k3p)
        Cbottom_diel = 2 * epsilon_0 * self.eps_substrate * ellipk(k3) / ellipk(k3p)
        self.C = Ctop_air + Cbottom_diel

        Ltop = (mu_0/2) * ellipk(k0p)/ellipk(k0)
        Lbottom = (mu_0/2) * ellipk(k3p)/ellipk(k3)
        self.L = Ltop * Lbottom / (Ltop + Lbottom) + self.sheet_inductance / self.width

        self.phase_velocity = 1/np.sqrt(self.L * self.C)
        self.eps_eff =  (speedoflight / self.phase_velocity)**2
        self.eps_eff2 = self.C/(Ctop_air + Cbottom_air)
        self.Z = np.sqrt(self.L/self.C)

        
    
    def resonator_frequency(self,
                            length: float=1*cm,
                            resonator_type: float=0.5,
                            harmonic: int=0,
                            unit: float=GHz,
                            print_status: bool=True) -> float:


        if (resonator_type==0.25): 
            length_factor = 0.25 * (2*harmonic + 1)
        elif (resonator_type==0.5):                      
            length_factor = 0.5 * (harmonic + 1)
        else:
            raise Exception("choose 1/4 or 1/2 resonator type")
        
        bare_frequency = length_factor * self.phase_velocity/length
        parameters = {
                        "f0, GHz": round(bare_frequency/unit, 3),
                        "length, mm": round(length/mm, 2),
                        "width, um": round(self.width/um, 2),
                        "gap, um": round(self.gap/um, 2),
                        "eps ": round(self.eps_substrate),
                        "eps_eff": round(self.eps_eff, 2),
                        "impedance, Ohm": round(self.Z, 2),
                        "L, nH/m": round(self.L * 1e9, 3),
                        "C, pF/m":round(self.C* 1e12, 3)
                      }
        if print_status:
            print(tabulate([parameters.keys(), parameters.values()]))
        
        self.length = length
        self.frequency = bare_frequency
        #return bare_frequency/unit

    def resonator_length(self,
                         resonator_frequency: float=5*GHz,
                         resonator_type: float=0.5,
                         harmonic: int=0,
                         unit: float=mm,
                         print_status: bool=True) -> None:
        length = resonator_type * self.phase_velocity / resonator_frequency
        parameters = {
                        "f0, GHz": round(resonator_frequency/GHz, 3),
                        "length, mm": round(length/mm, 3),
                        "width, um": round(self.width/um, 2),
                        "gap, um": round(self.gap/um, 2),
                        "eps ": round(self.eps_substrate),
                        "eps_eff": round(self.eps_eff, 2),
                        "impedance, Ohm": round(self.Z, 2),
                        "L, nH/m": round(self.L * 1e9, 3),
                        "C, pF/m":round(self.C* 1e12, 3)
                      }
        if print_status:
            print(tabulate([parameters.keys(), parameters.values()]))

        self.length = length
        self.frequency = resonator_frequency
        #return length