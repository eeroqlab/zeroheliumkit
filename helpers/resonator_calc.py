from scipy import sqrt,pi, sinh
from scipy.special import ellipk

mu0 = 1.25663706e-6
eps0 = 8.85418782e-12
speedoflight = 299792458.0

def calculate_eps_eff_from_geometry(substrate_epsR: float,
                                    w: float,
                                    g: float,
                                    substrate_height: float) -> float:
    a = w
    b = w + 2 * g
    h = substrate_height
    k0 = float(a)/b
    k0p = sqrt(1 - k0**2)
    #k3 = tanh(pi*a/(4*h))/  tanh(pi*b/(4*h))
    k3 = sinh(pi * a/(4 * h)) / sinh(pi * b/(4 * h))
    k3p = sqrt(1 - k3**2)
    Ktwid = ellipk(k0p**2) * ellipk(k3**2)/(ellipk(k0**2) * ellipk(k3p**2))
    
    #return (1+substrate_epsR*Ktwid)/(1+Ktwid)
    return 1 + (substrate_epsR - 1) * Ktwid / 2

def calculate_eps_eff (phase_velocity: float) -> float:
    return (speedoflight/phase_velocity)**2

def calculate_impedance (w: float, 
                         g: float, 
                         eps_eff: float) -> float:
    #From Andreas' resonator paper or my thesis...agrees for values given in his paper
    k0 = float(w)/(w + 2 * g)
    k0p = sqrt(1 - k0**2)
    L = (mu0/4) * ellipk(k0p**2)/ellipk(k0**2)
    C = 4 * eps0 * eps_eff * ellipk(k0**2)/ellipk(k0p**2)
    Z = sqrt(L/C)
    
    return Z

def calculate_resonator_frequency(length: float,
                                  eps_eff: float,
                                  impedance: float,
                                  resonator_type: float=0.5,
                                  harmonic: int=0) -> float:
    
    phase_velocity = speedoflight/sqrt(eps_eff)

    if (resonator_type==0.25): 
        length_factor = 0.25 * (2*harmonic + 1)
    else:                      
        length_factor = 0.5 * (harmonic + 1)
    
    bare_frequency = 1e6 * length_factor * phase_velocity/length
    
    return 1e-9 * bare_frequency

def calculate_resonator_length(resonator_type: float,
                               effective_eps: float,
                               resonator_frequency: float) -> float:
    return resonator_type * speedoflight/(sqrt(effective_eps) * resonator_frequency * 1e9) * 1e6



