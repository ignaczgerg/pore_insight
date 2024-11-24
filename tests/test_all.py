import numpy as np
import matplotlib.pyplot as plt

from fitting_models import curve_fitting, PSD, MolarVolume, DistributionParameters
from utils import StokesRadiusCalculator, DiffusivityCalculator, Solvent

def main():
    # Rifan (2024), Table S2, M4. MetOH 25C 0.543 cP
    molecular_weights = [236.4,272.38,327.33,422.92,435.52,540.51,585.54,837.05,1017.65] #g mol-1
    rejections = [4.38,4.79,8.68,11.76,17.56,50.92,58.8,88.39,94.82] #%
    errors = [1.57,0.76,1.55,1.82,3.1,4.86,2.11,2.79,2.2] #%
    solvent = Solvent.from_selection(2,298.15,0.543) #MetOH
    
    mwco_line = curve_fitting(molecular_weights,rejections)
    mw_range, fit_rej_A, _, _, _ = mwco_line.fit_curve('boltzmann')
    mw_range, fit_rej_B, _, _, _ = mwco_line.fit_curve('sigmoid')
    mw_range, fit_rej_C, _, _, _ = mwco_line.fit_curve('generalized_logistic')
    mw_range, fit_rej_D, _, _, _ = mwco_line.fit_curve('gompertz')
    mw_range, fit_rej_E, _, _, _ = mwco_line.fit_curve('double_sigmoid')

    plt.errorbar(molecular_weights,rejections,errors,ls='none',marker='o', capsize=5, capthick=1, ecolor='black')
    plt.plot(mw_range,fit_rej_A, color='r',label='Boltzmann')
    plt.plot(mw_range,fit_rej_B, color='b',label='Sigmoid')
    plt.plot(mw_range,fit_rej_C, color='k',label='g(x)')
    plt.plot(mw_range,fit_rej_D, color='y',label='Gompertz')
    plt.plot(mw_range,fit_rej_E, color='g',label='Double Sigmoid')
    plt.xlabel('Molecular Weight (g gmol-1)')
    plt.ylabel('Rejection (%)')
    plt.title('MWCO curve')
    plt.legend()
    plt.show()

    # Molar Volumes    
    Vm_A, Vm_B = [], []
    for i in molecular_weights:
        x = MolarVolume.relation_Schotte(i)
        y = MolarVolume.relation_Wu(i)
        Vm_A.append(x)
        Vm_B.append(y)

    # Rifan (2024), Table S2, M4. MetOH 25C 0.543 cP
    smiles = {
        'Styrene Dimer': ('CC(C)(CC(=C)c1ccccc1)c2ccccc2'),
        'Estradiol': ('OC(C=C1)=CC2=C1[C@@]3([H])CC[C@]4(C)[C@@H](O)CC[C@@]4([H])[C@]3([H])CC2'),
        'Methyl Orange': ('[O-]S(C1=CC=C(/N=N/C2=CC=C(N(C)C)C=C2)C=C1)(=O)=O.[Na+]'),
        'Losartan': ('OCC1=C(N=C(N1CC2=CC=C(C3=C(C4=NN=NN4)C=CC=C3)C=C2)CCCC)Cl'),
        'Valsartan': ('CC([C@H](N(CC1=CC=C(C2=C(C3=NN=NN3)C=CC=C2)C=C1)C(CCCC)=O)C(O)=O)C'),
        'Oleuropein': ('OC1=C(O)C=C(CCOC(C[C@@]2([H])/C([C@H](OC3[C@@H]([C@H]([C@@H]([C@@H](CO)O3)O)O)O)OC=C2C(OC)=O)=C\C)=O)C=C1'),
        'Acid fuchsin': ('CC1=C/C(=C(/C2=CC(=C(C=C2)N)S(=O)(=O)O)\C3=CC(=C(C=C3)N)S(=O)(=O)[O-])/C=C(C1=N)S(=O)(=O)[O-].[Na+].[Na+]'),
        'Roxithromycin': ('CC[C@@H]1[C@@]([C@@H]([C@H](C(=NOCOCCOC)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O'),
        'Rose Bengal': ('O=C1OC2(C3=C(OC4=C2C=C(I)C([O-])=C4I)C(I)=C([O-])C(I)=C3)C5=C1C(Cl)=C(Cl)C(Cl)=C5Cl.[Na+].[Na+]')
    }
    
    Vm = []
    for i in smiles:
        x = MolarVolume.joback(smiles[i])
        Vm.append(x)

    print('\nMolar Volumes [cm3 gmol-1]')
    print('\n',Vm_A,'\n',Vm_B,'\n',Vm)

    #Diffusion coefficient
    D_A = []
    for i in Vm_A:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_A.append(D)

    D_B = []
    for i in Vm_B:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_B.append(D)

    D_j = []
    for i in Vm:
        D = DiffusivityCalculator.wilke_chang_diffusion_coefficient(i,solvent.molecular_weight,solvent.temperature,solvent.viscosity,solvent.alpha)
        D_j.append(D)

    print('\nDiffusion coefficients [cm2 s-1]')
    print('\n',D_A,'\n',D_B,'\n',D_j)

    #radius
    r_A = []
    for i in D_A:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r_A.append(r)

    r_B = []
    for i in D_B:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r_B.append(r)

    r_j = []
    for i in D_j:
        r = StokesRadiusCalculator.stokes_einstein_radius(i,solvent.temperature,solvent.viscosity)
        r_j.append(r)
    
    print('\nRadius [nm]')
    print('\n',r_A,'\n',r_B,'\n',r_j)

    # r curves fitting
    # With r_A (radii calculated with Schotte relation)
    r_line = curve_fitting(r_A,rejections,errors)
    r_range, fit_rA_A, _, low_rA_A, high_rA_A = r_line.fit_curve('boltzmann')
    r_range, fit_rA_B, rA_opt_B, low_rA_B, high_rA_B = r_line.fit_curve('sigmoid')
    r_range, fit_rA_C, rA_opt_C, low_rA_C, high_rA_C = r_line.fit_curve('generalized_logistic')
    r_range, fit_rA_D, rA_opt_D, low_rA_D, high_rA_D = r_line.fit_curve('gompertz')
    r_range, fit_rA_E, rA_opt_E, low_rA_E, high_rA_E = r_line.fit_curve('double_sigmoid')

    plt.plot(r_range,fit_rA_A, color='r',label='Boltzmann')
    plt.plot(r_range,fit_rA_B, color='b',label='Sigmoid')
    plt.plot(r_range,fit_rA_C, color='k',label='g(x)')
    plt.plot(r_range,fit_rA_D, color='y',label='Gompertz')
    plt.plot(r_range,fit_rA_E, color='g',label='Double Sigmoid')
    plt.xlabel('rs (nm)')
    plt.ylabel('Rejection (%)')
    plt.title('Radius curve - rA')
    plt.legend()
    plt.show()

    # With r_B (radii calculated with Wu relation)
    r_line = curve_fitting(r_B,rejections,errors)
    r_range, fit_rB_A, _, low_rB_A, high_rB_A = r_line.fit_curve('boltzmann')
    r_range, fit_rB_B, rB_opt_B, low_rB_B, high_rB_B = r_line.fit_curve('sigmoid')
    r_range, fit_rB_C, rB_opt_C, low_rB_C, high_rB_C = r_line.fit_curve('generalized_logistic')
    r_range, fit_rB_D, rB_opt_D, low_rB_D, high_rB_D = r_line.fit_curve('gompertz')
    r_range, fit_rB_E, rB_opt_E, low_rB_E, high_rB_E = r_line.fit_curve('double_sigmoid')

    plt.plot(r_range,fit_rB_A, color='r',label='Boltzmann')
    plt.plot(r_range,fit_rB_B, color='b',label='Sigmoid')
    plt.plot(r_range,fit_rB_C, color='k',label='g(x)')
    plt.plot(r_range,fit_rB_D, color='y',label='Gompertz')
    plt.plot(r_range,fit_rB_E, color='g',label='Double Sigmoid')
    plt.xlabel('rs (nm)')
    plt.ylabel('Rejection (%)')
    plt.title('Radius curve - rB')
    plt.legend()
    plt.show()    

    # With r_j (radii calculated with Joback)
    r_line = curve_fitting(r_j,rejections,errors)
    r_range, fit_rj_A, _, low_rj_A, high_rj_A = r_line.fit_curve('boltzmann')
    r_range, fit_rj_B, rj_opt_B, low_rj_B, high_rj_B = r_line.fit_curve('sigmoid')
    r_range, fit_rj_C, rj_opt_C, low_rj_C, high_rj_C = r_line.fit_curve('generalized_logistic')
    r_range, fit_rj_D, rj_opt_D, low_rj_D, high_rj_D = r_line.fit_curve('gompertz')
    #r_range, fit_rj_E, rj_opt_E, low_rj_E, high_rj_E = r_line.fit_curve('double_sigmoid')

    plt.plot(r_range,fit_rj_A, color='r',label='Boltzmann')
    plt.plot(r_range,fit_rj_B, color='b',label='Sigmoid')
    plt.plot(r_range,fit_rj_C, color='k',label='g(x)')
    plt.plot(r_range,fit_rj_D, color='y',label='Gompertz')
    #plt.plot(r_range,fit_rj_E, color='g',label='Double Sigmoid')
    plt.xlabel('rs (nm)')
    plt.ylabel('Rejection (%)')
    plt.title('Radius curve - rj')
    plt.legend()
    plt.show()

    # PSD curve
    # log-normal probability density function, first we need the distribution parameters
    x = np.linspace(0.01, 2, 100)  # Random pore sizes
    # With r_A
    rA_avg_A, rA_SD_A, rA_A = DistributionParameters.PDF(r_range,fit_rA_A,low_rA_A,high_rA_A) # Boltzmann
    rA_avg_B, rA_SD_B, rA_B = DistributionParameters.PDF(r_range,fit_rA_B,low_rA_B,high_rA_B) # Sigmoid
    rA_avg_C, rA_SD_C, rA_C = DistributionParameters.PDF(r_range,fit_rA_C,low_rA_C,high_rA_C) # g(x)
    rA_avg_D, rA_SD_D, rA_D = DistributionParameters.PDF(r_range,fit_rA_D,low_rA_D,high_rA_D) # Gompertz
    rA_avg_E, rA_SD_E, rA_E = DistributionParameters.PDF(r_range,fit_rA_E,low_rA_E,high_rA_E) # Double Sigmoid

    psd_rA_A = PSD.PDF(x,rA_avg_A,rA_SD_A)
    psd_rA_B = PSD.PDF(x,rA_avg_B,rA_SD_B)
    psd_rA_C = PSD.PDF(x,rA_avg_C,rA_SD_C)
    psd_rA_D = PSD.PDF(x,rA_avg_D,rA_SD_D)
    psd_rA_E = PSD.PDF(x,rA_avg_E,rA_SD_E)

    plt.plot(x,psd_rA_A, color='r',label='Boltzmann')
    plt.plot(x,psd_rA_B, color='b',label='Sigmoid')
    plt.plot(x,psd_rA_C, color='k',label='g(x)')
    plt.plot(x,psd_rA_D, color='y',label='Gompertz')
    plt.plot(x,psd_rA_E, color='g',label='Double Sigmoid')
    plt.xlabel('rp (nm)')
    plt.ylabel('(a.u.)')
    plt.title('PSD - rA')
    plt.legend()
    plt.show()

    # With r_B 
    rB_avg_A, rB_SD_A, rB_A = DistributionParameters.PDF(r_range,fit_rB_A,low_rB_A,high_rB_A) # Boltzmann
    rB_avg_B, rB_SD_B, rB_B = DistributionParameters.PDF(r_range,fit_rB_B,low_rB_B,high_rB_B) # Sigmoid
    rB_avg_C, rB_SD_C, rB_C = DistributionParameters.PDF(r_range,fit_rB_C,low_rB_C,high_rB_C) # g(x)
    rB_avg_D, rB_SD_D, rB_D = DistributionParameters.PDF(r_range,fit_rB_D,low_rB_D,high_rB_D) # Gompertz
    rB_avg_E, rB_SD_E, rB_E = DistributionParameters.PDF(r_range,fit_rB_E,low_rB_E,high_rB_E) # Double Sigmoid

    psd_rB_A = PSD.PDF(x,rB_avg_A,rB_SD_A)
    psd_rB_B = PSD.PDF(x,rB_avg_B,rB_SD_B)
    psd_rB_C = PSD.PDF(x,rB_avg_C,rB_SD_C)
    psd_rB_D = PSD.PDF(x,rB_avg_D,rB_SD_D)
    psd_rB_E = PSD.PDF(x,rB_avg_E,rB_SD_E)

    plt.plot(x,psd_rB_A, color='r',label='Boltzmann')
    plt.plot(x,psd_rB_B, color='b',label='Sigmoid')
    plt.plot(x,psd_rB_C, color='k',label='g(x)')
    plt.plot(x,psd_rB_D, color='y',label='Gompertz')
    plt.plot(x,psd_rB_E, color='g',label='Double Sigmoid')
    plt.xlabel('rp (nm)')
    plt.ylabel('(a.u.)')
    plt.title('PSD - rB')
    plt.legend()
    plt.show()

    # With r_j 
    rj_avg_A, rj_SD_A, rj_A = DistributionParameters.PDF(r_range,fit_rj_A,low_rj_A,high_rj_A) # Boltzmann
    print(rj_avg_A,rj_SD_A,rj_A)
    rj_avg_B, rj_SD_B, rj_B = DistributionParameters.PDF(r_range,fit_rj_B,low_rj_B,high_rj_B) # Sigmoid
    rj_avg_C, rj_SD_C, rj_C = DistributionParameters.PDF(r_range,fit_rj_C,low_rj_C,high_rj_C) # g(x)
    rj_avg_D, rj_SD_D, rj_D = DistributionParameters.PDF(r_range,fit_rj_D,low_rj_D,high_rj_D) # Gompertz

    #psd_rj_A = PSD.PDF(x,rj_avg_A,rj_SD_A)
    psd_rj_B = PSD.PDF(x,rj_avg_B,rj_SD_B)
    psd_rj_C = PSD.PDF(x,rj_avg_C,rj_SD_C)
    psd_rj_D = PSD.PDF(x,rj_avg_D,rj_SD_D)

    #plt.plot(x,psd_rj_A, color='r',label='Boltzmann')
    plt.plot(x,psd_rj_B, color='b',label='Sigmoid')
    plt.plot(x,psd_rj_C, color='k',label='g(x)')
    plt.plot(x,psd_rj_D, color='y',label='Gompertz')
    plt.xlabel('rp (nm)')
    plt.ylabel('(a.u.)')
    plt.title('PSD - rj')
    plt.legend()
    plt.show()  

if __name__ == "__main__":
    main()