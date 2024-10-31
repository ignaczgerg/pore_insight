import numpy as np
import matplotlib.pyplot as plt

from fitting_models import MWCOFitting, PoreSizeDistribution, MolarVolumeRelation
from utils import StokesRadiusCalculator, DiffusivityCalculator, Solvent

def main():
    # Ruo-yu Fu (2023). Water 25C 0.890 cP https://doi.org/10.1016/j.desal.2022.116318
    molecular_weights = [92, 122, 150,180] #g mol-1
    rejections = [37.96,47.96,54.84,62.96] #%
    errors = np.zeros_like(rejections)

    solvent = Solvent.from_selection(1,298.15,0.89) #Water
    mwco_line = MWCOFitting(molecular_weights,rejections,errors)
    mw_range, fit_rej_A, _ = mwco_line.fit_curve('model_f')
    mw_range, fit_rej_B, _ = mwco_line.fit_curve('sigmoid')
    mw_range, fit_rej_C, _ = mwco_line.fit_curve('generalized_logistic')
    mw_range, fit_rej_D, _ = mwco_line.fit_curve('gompertz')
    #mw_range, fit_rej_E = mwco_line.fit_curve('double_sigmoid')

    # Molar Volumes    
    Vm_A, Vm_B = [], []
    for i in molecular_weights:
        x = MolarVolumeRelation.relation_vs_a(i)
        y = MolarVolumeRelation.relation_vs_b(i)
        Vm_A.append(x)
        Vm_B.append(y)

    # Ruo-yu Fu (2023). Water 25C 0.890 cP
    smiles = {
        'Glycerin': ('C(C(CO)O)O'),
        'Erythritol': ('C([C@H]([C@H](CO)O)O)O'),
        'Xylose': ('C1[C@H]([C@@H]([C@H](C(O1)O)O)O)O'),
        'Glucose': ('C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O')
    }
    
    Vm = []
    for i in smiles:
        Vc = MolarVolumeRelation.joback(smiles[i])
        x = 0.285 * (Vc)**1.048
        x = x * 1000
        Vm.append(x)

    print(Vm_A)
    print(Vm_B)
    print(Vm)

    #Diffusion coefficient
    D_A = []
    for i in Vm_A:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_A.append(D)
    print(D_A)

    D_B = []
    for i in Vm_B:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_B.append(D)
    print(D_B)

    D_j = []
    for i in Vm:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_j.append(D)
    print(D_j)

    #radius
    r_A = []
    for i in D_A:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r = r/(1e-9)
        r_A.append(r)
    print(r_A)

    r_B = []
    for i in D_B:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r = r/(1e-9)
        r_B.append(r)
    print(r_B)

    r_j = []
    for i in D_j:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r = r/(1e-9)
        r_j.append(r)
    print(r_j)

    #MWCO fitting to obtain the optimal parameters
    r_line = MWCOFitting(r_A,rejections,errors)
    _, _, rA_opt_B = r_line.fit_curve('sigmoid')
    _, _, rA_opt_C = r_line.fit_curve('generalized_logistic')
    _, _, rA_opt_D = r_line.fit_curve('gompertz')
    _, _, rA_opt_E = r_line.fit_curve('double_sigmoid')

    r_line = MWCOFitting(r_B,rejections,errors)
    _, _, rB_opt_B = r_line.fit_curve('sigmoid')
    _, _, rB_opt_C = r_line.fit_curve('generalized_logistic')
    _, _, rB_opt_D = r_line.fit_curve('gompertz')
    _, _, rB_opt_E = r_line.fit_curve('double_sigmoid')

    r_line = MWCOFitting(r_j,rejections,errors)
    _, _, rj_opt_B = r_line.fit_curve('sigmoid')
    _, _, rj_opt_C = r_line.fit_curve('generalized_logistic')
    _, _, rj_opt_D = r_line.fit_curve('gompertz')
    _, _, rj_opt_E = r_line.fit_curve('double_sigmoid')

    #Pore Size Distribution
    x = np.linspace(0.01, 1, 100)  # Random pore sizes
    # log-normal PDF
    avg_radius = 0.3
    std_dev = 0.1
    psd_A = PoreSizeDistribution.calculate_psd(x,avg_radius,std_dev)

    # Derivative Sigmoid
    psd_j_B = PoreSizeDistribution.derivative_sigmoid(x,*rj_opt_B)

    # Derivative Generalized Logistic
    psd_j_C = PoreSizeDistribution.derivative_generalized_logistic(x,*rj_opt_C)

    # Derivative Gompertz
    psd_j_D = PoreSizeDistribution.derivative_gompertz(x,*rj_opt_D)

    # Derivative Double Sigmoid
    psd_j_E = PoreSizeDistribution.derivative_double_sigmoid(x,*rj_opt_E)

    plt.plot(x,psd_A, color='r',label='PDF')
    plt.plot(x,psd_j_B, color='b',label='Derivative Sigmoid')
    plt.plot(x,psd_j_C, color='k',label='Derivative g(x)')
    plt.plot(x,psd_j_D, color='y',label='Derivative Gompertz')
    plt.plot(x,psd_j_E, color='g',label='Derivative Double Sigmoid')
    plt.xlabel('radius (nm)')
    plt.ylabel('(a.u.)')
    plt.title('PSD')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()