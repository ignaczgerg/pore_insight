from dataclasses import dataclass
from typing import Dict
import numpy as np
from numpy import log, exp
from rdkit import Chem
from rdkit.Chem import Descriptors
from utils import rej_bounds, read_molecules
import scipy.stats as stats
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import root

class CurveModels:
    @staticmethod
    def lognormal_CDF(x,r_p,sigma):
        b = np.log(1 + (sigma / r_p) ** 2)
        return 0.5 + 0.5 * erf((np.log(x / r_p) + b / 2) / np.sqrt(2 * b))
    
    @staticmethod
    def boltzmann(x, a, b, c, d):
        return b + (a - b) / (1 + np.exp((x - c) / d))
    
    @staticmethod
    def sigmoid(x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    @staticmethod
    def generalized_logistic(x, A, K, B, Q, C, nu):
        return A + (K - A) / (C + Q * np.exp(-B * x))**(1 / nu)

    @staticmethod
    def gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    @staticmethod
    def double_sigmoid(x, K1, B1, M1, K2, B2, M2):
        return (K1 / (1 + np.exp(-B1 * (x - M1)))) + (K2 / (1 + np.exp(-B2 * (x - M2))))

class PSDModels:
    @staticmethod
    def derivative_boltzmann(x, a, b, c, d):
        """
        Derivative of the Boltzmann function.
        
        Parameters
        ----------
        x : array-like
            Input values.
        a, b, c, d : float
            Parameters of the Boltzmann function.
        
        Returns
        -------
        array-like
            Derivative of the Boltzmann function at each point in x.
        """
        exp_component = np.exp((x - c) / d)
        return (a - b) * exp_component / (d * (1 + exp_component) ** 2)

    @staticmethod
    def log_normal(x, avg_r, std_dev):
        """
        Calculates the pore size distribution using the log-normal probability density function.
        
        Parameters
        ----------
        x : array-like
            Pore sizes.
        avg_r : float
            Average pore size radius.
        std_dev : float
            Standard deviation of the pore size distribution.

        Returns
        -------
        psd : array-like
            Calculated pore size distribution.
        """
        b = np.log(1 + (std_dev / avg_r)**2)
        c1 = 1 / np.sqrt(2 * np.pi * b)
        return c1 * (1 / x) * np.exp(-((np.log(x / avg_r) + b / 2)**2) / (2 * b)) 

    @staticmethod
    def derivative_sigmoid(x, a, b, c):
        """
        Derivative of the Sigmoid function.
        
        Parameters
        ----------
        x : array-like
            Input values.
        a : float
            Steepness of the curve.
        b : float
            Midpoint of the curve.
        c : float
            Maximum value of the sigmoid.

        Returns
        -------
        array-like
            Derivative of the sigmoid at each point in x.
        """
        sigmoid_value = CurveModels.sigmoid(x, a, b, c)
        return a * sigmoid_value * (1 - sigmoid_value / c)

    @staticmethod
    def derivative_generalized_logistic(x, A, K, B, Q, C, nu):
        """
        Derivative of the Generalized Logistic function.

        Parameters
        ----------
        x : array-like
            Input values.
        A, K, B, Q, C, nu : float
            Parameters of the generalized logistic function.

        Returns
        -------
        array-like
            Derivative of the generalized logistic function at each point in x.
        """
        numerator = (K - A) * (-B) * Q * np.exp(-B * x)
        denominator = nu * (C + Q * np.exp(-B * x))**(1 / nu + 1)
        return - numerator / denominator

    @staticmethod
    def derivative_gompertz(x, a, b, c):
        """
        Derivative of the Gompertz function.

        Parameters
        ----------
        x : array-like
            Input values.
        a, b, c : float
            Parameters of the Gompertz function.

        Returns
        -------
        array-like
            Derivative of the Gompertz function at each point in x.
        """
        return a * b * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
    
    @staticmethod
    def derivative_double_sigmoid(x, K1, B1, M1, K2, B2, M2):
        """
        Derivative of the Double Sigmoid function.

        Parameters
        ----------
        x : array-like
            Input values.
        K1, B1, M1 : float
            Parameters for the first sigmoid function.
        K2, B2, M2 : float
            Parameters for the second sigmoid function.

        Returns
        -------
        array-like
            Derivative of the double sigmoid function at each point in x.
        """
        term1 = (K1 * B1 * np.exp(-B1 * (x - M1))) / ((1 + np.exp(-B1 * (x - M1)))**2)
        term2 = (K2 * B2 * np.exp(-B2 * (x - M2))) / ((1 + np.exp(-B2 * (x - M2)))**2)
        return term1 + term2
    

class DistributionModels:
    @staticmethod
    def rp_mode(x_dist,y_dist):
        """
        Identifies the peak (maximum point) in the PSD curve.

        Parameters
        ----------
        x_dist : array-like
            Pore size x values of the PSD curve.

        Returns
        ----------
        x_peak : float
            x value corresponding to the maximum y value.
        """
        i_max = np.argmax(y_dist)
        r_mode = round(x_dist[i_max],4) # rp_mode
        return r_mode
    
    @staticmethod
    def sigma_FWHM(x_dist,y_dist):
        """
        Calculates the log-normal standard deviation (σ*) using the Full Witdh at Half Maximum (FWHM) of the curve.

        Parameters
        ----------
        x_dist : array-like
            Pore size x values of the PSD curve.
        y_dist : array-like
            Density y values of the PSD curve.

        Returns
        ----------
        sigma : float
            log-normal standard deviation.
        """
        i_max = np.argmax(y_dist)
        y_peak = y_dist[i_max]
        y_half_max = y_peak / 2
        i_half_max = np.where(y_dist >= y_half_max)[0]
        FWHM = x_dist[i_half_max[-1]] - x_dist[i_half_max[0]]
        median = np.median(x_dist)
        mu = np.log(median)
        b = (np.sqrt(2) * np.log(2))**-1
        sigma = b * np.arcsinh(FWHM / (2 * np.exp(mu)))
        return sigma
    
    @staticmethod
    def rp_average(rp_mode,sigma):
        return np.exp( np.log(rp_mode) + sigma**2 )
    
class MolarVolume:
    @staticmethod
    def relation_Schotte(mol_or_mw):
        """
        Estimate the molar volume of a molecule based on its molecular weight using a linear relation (method a).
        Method a: Schotte et al. group contribution theory

        Parameters
        ----------
        mol_or_mw: rdkit Chem.Mol object or a float molecular weight value for a single molecule for which the molar volume will be estimated.

        Returns
        -------
        float

        References
        ----------
        - William Schotte, Prediction of the molar volume at the normal boiling point, The Chemical Engineering Journal, Volume 48, Issue 3, 1992, Pages 167-172, ISSN 0300-9467
        - https://doi.org/10.1016/0300-9467(92)80032-6
        """
        if isinstance(mol_or_mw,Chem.Mol):
            mw = Descriptors.MolWt(mol_or_mw)
        elif isinstance(mol_or_mw,(float,int)):
            mw = mol_or_mw
        else:
            raise ValueError("RDKit Mol object not found. Please provide a valid RDKit Mol object.")
        return 1.3348 * mw - 10.552

    @staticmethod
    def relation_Wu(mol_or_mw):
        """
        Estimate the molar volume of a molecule based on its molecular weight using a linear relation (method b).
        Method b: Wu et al. group contribution theory

        Parameters
        ----------
        mol_or_mw: rdkit Chem.Mol object or a float molecular weight value for a single molecule for which the molar volume will be estimated.

        Returns
        -------
        float

        References
        ----------
        - Albert X. Wu, Sharon Lin, Katherine Mizrahi Rodriguez, Francesco M. Benedetti, Taigyu Joo, Aristotle F. Grosz, Kayla R. Storme, Naksha Roy, Duha Syar, Zachary P. Smith, Revisiting group contribution theory for estimating fractional free volume of microporous polymer membranes, Journal of Membrane Science, Volume 636, 2021, 119526, ISSN 0376-7388
        - https://doi.org/10.1016/j.memsci.2021.119526
        """
        if isinstance(mol_or_mw,Chem.Mol):
            mw = Descriptors.MolWt(mol_or_mw)
        elif isinstance(mol_or_mw,(float,int)):
            mw = mol_or_mw
        else:
            raise ValueError("RDKit Mol object not found. Please provide a valid RDKit Mol object.")
        return 1.1353* mw + 3.8219

    @staticmethod
    def joback(mol: Chem.Mol):
        """
        Estimate thermodynamic properties of a molecule using the Joback method based on its SMILES notation.

        Parameters
        ----------
        SMILES : str
            SMILES string representing the molecular structure for which thermodynamic properties will be calculated.

        Returns
        -------
        Vm : float
            Estimated molar volume at the normal boiling point (in cubic entimeters per mole).
            The other returns are currently not implemented.

        Notes
        -----
        This method estimates several thermodynamic properties using group contribution values. The properties 
        are estimated based on functional groups identified in the molecule, such as the boiling point, critical 
        temperature, critical pressure, heat of formation, and Gibbs free energy of formation. These estimates 
        are calculated using the Joback method, which applies a group contribution technique with pre-defined 
        parameters for various functional groups.

        Steps performed:
        ----------------
        1. Parses the SMILES string and converts it into a molecular structure using RDKit.
        2. Identifies functional groups in the molecule based on SMARTS patterns and calculates their occurrences.
        3. Estimates the following thermodynamic properties:
        - Boiling point (TB) [°C]
        - Critical temperature (TC) [°C]
        - Critical pressure (PC) [MPa]
        - Molar volume (Vm) [cm³/mol]
        - Heat of formation (Hform) [kcal/mol]
        - Gibbs free energy of formation (Gform) [kcal/mol]
        - Ideal gas heat capacity (CPIG) [cal/mol-K]
        4. Converts units where necessary:
        - Temperatures from Kelvin to Celsius
        - Energy values from kJ/mol to kcal/mol
        - Heat capacity from J/mol-K to cal/mol-K
        5. Issues warnings when missing parameter values prevent the calculation of certain properties for specific functional groups.

        Group contribution parameters used are based on reference values for known functional groups and their respective
        contributions to the thermodynamic properties mentioned above.

        References
        ----------
        - Joback, K.G. and Reid, R.C. (1987) Estimation of Pure-Component Properties from Group-Contributions. Chemical Engineering Communications, 57, 233-243. 
        - https://en.wikipedia.org/wiki/Joback_method
        
        Example
        -------
        >>> joback('CCO')
        0.1065  # Estimated critical volume for ethanol
        
        """
        molecule = mol
        # Blank
        group = [0 for i in range(41)]
        SMARTS = [0 for i in range(41)]
        n = [0 for i in range(41)]

        # SMARTS Codes
        group[0] = '-CH3 (non-ring)'; SMARTS[0] = Chem.MolFromSmarts('[CX4H3]')
        group[1] = '-CH2- (non-ring)'; SMARTS[1] = Chem.MolFromSmarts('[!R;CX4H2]')
        group[2] = '>CH- (non-ring)'; SMARTS[2] = Chem.MolFromSmarts('[!R;CX4H]')
        group[3] = '>C< (non-ring)'; SMARTS[3] = Chem.MolFromSmarts('[!R;CX4H0]')
        group[4] = '=CH2 (non-ring)'; SMARTS[4] = Chem.MolFromSmarts('[CX3H2]')
        group[5] = '=CH- (non-ring)'; SMARTS[5] = Chem.MolFromSmarts('[!R;CX3H1;!$([CX3H1](=O))]')
        group[6] = '=C< (non-ring)'; SMARTS[6] = Chem.MolFromSmarts('[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]')
        group[7] = '=C= (non-ring)'; SMARTS[7] = Chem.MolFromSmarts('[$([CX2H0](=*)=*)]')
        group[8] = '≡CH (non-ring)'; SMARTS[8] = Chem.MolFromSmarts('[$([CX2H1]#[!#7])]') 
        group[9] = '≡C− (non-ring)'; SMARTS[9] = Chem.MolFromSmarts('[$([CX2H0]#[!#7])]') 
        group[10] = '−CH2− (ring)'; SMARTS[10] = Chem.MolFromSmarts('[R;CX4H2]')
        group[11] = '>CH- (ring)'; SMARTS[11] = Chem.MolFromSmarts('[R;CX4H]')
        group[12] = '>C< (ring)'; SMARTS[12] = Chem.MolFromSmarts('[R;CX4H0]')
        group[13] = '=CH- (ring)'; SMARTS[13] = Chem.MolFromSmarts('[R;CX3H1,cX3H1]')
        group[14] = '=C< (ring)'; SMARTS[14] = Chem.MolFromSmarts('[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]')
        group[15] = '-F'; SMARTS[15] = Chem.MolFromSmarts('[F]')
        group[16] = '-Cl'; SMARTS[16] = Chem.MolFromSmarts('[Cl]')
        group[17] = '-Br'; SMARTS[17] = Chem.MolFromSmarts('[Br]') 
        group[18] = '-I'; SMARTS[18] = Chem.MolFromSmarts('[I]')
        group[19] = '-OH (alcohol)'; SMARTS[19] = Chem.MolFromSmarts('[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]')
        group[20] = '-OH (phenol)'; SMARTS[20] = Chem.MolFromSmarts('[O;H1;$(O-!@c)]')
        group[21] = '-O- (non-ring)'; SMARTS[21] = Chem.MolFromSmarts('[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]')
        group[22] = '-O- (ring)'; SMARTS[22] = Chem.MolFromSmarts('[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]') 
        group[23] = '>C=O (non-ring)'; SMARTS[23] = Chem.MolFromSmarts('[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O')
        group[24] = '>C=O (ring)'; SMARTS[24] = Chem.MolFromSmarts('[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O')
        group[25] = 'O=CH- (aldehyde)'; SMARTS[25] = Chem.MolFromSmarts('[CH;D2;$(C-!@C)](=O)')
        group[26] = '-COOH (acid)'; SMARTS[26] = Chem.MolFromSmarts('[OX2H]-[C]=O')
        group[27] = '-COO- (ester)'; SMARTS[27] = Chem.MolFromSmarts('[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]')
        group[28] = '=O (other than above)'; SMARTS[28] = Chem.MolFromSmarts('[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]')
        group[29] = '-NH2'; SMARTS[29] = Chem.MolFromSmarts('[NX3H2]')
        group[30] = '>NH (non-ring)'; SMARTS[30] = Chem.MolFromSmarts('[NX3H1;!R]')
        group[31] = '>NH (ring)'; SMARTS[31] = Chem.MolFromSmarts('[#7X3H1;R]') 
        group[32] = '>N- (non-ring)'; SMARTS[32] = Chem.MolFromSmarts('[#7X3H0;!$([#7](~O)~O)]')
        group[33] = '-N= (non-ring)'; SMARTS[33] = Chem.MolFromSmarts('[#7X2H0;!R]')
        group[34] = '-N= (ring)'; SMARTS[34] = Chem.MolFromSmarts('[#7X2H0;R]')
        group[35] = '=NH'; SMARTS[35] = Chem.MolFromSmarts('[#7X2H1]')
        group[36] = '-CN'; SMARTS[36] = Chem.MolFromSmarts('[#6X2]#[#7X1H0]')
        group[37] = '-NO2'; SMARTS[37] = Chem.MolFromSmarts('[$([#7X3,#7X3+][!#8])](=[O])∼[O-]')
        group[38] = '-SH'; SMARTS[38] = Chem.MolFromSmarts('[SX2H]')
        group[39] = '-S- (non-ring)'; SMARTS[39] = Chem.MolFromSmarts('[#16X2H0;!R]')
        group[40] = '-S- (ring)'; SMARTS[40] = Chem.MolFromSmarts('[#16X2H0;R]')

        # Parameters: https://en.wikipedia.org/wiki/Joback_method
        # [Tc,Pc,Vc,Tb,Tm,Hform,Gform,a,b,c,d,Hfusion,Hvap,ηa,ηb]
        p = [
        [0.0141,    -0.0012,    65,     23.58,      -5.1,       -76.45,     -43.96,     1.95E+01,       -8.08E-03,      1.53E-04,       -9.67E-08,      0.908,      2.373,      548.29,     -1.719],
        [0.0189,    0,          56,     22.88,      11.27,      -20.64,     8.42,       -9.09E-01,      9.50E-02,       -5.44E-05,      1.19E-08,       2.59,       2.226,      94.16,      -0.199],
        [0.0164,    0.002,      41,     21.74,      12.64,      29.89,      58.36,      -2.30E+01,      2.04E-01,       -2.65E-04,      1.20E-07,       0.749,      1.691,      -322.15,    1.187],
        [0.0067,    0.0043,     27,     18.25,      46.43,      82.23,      116.02,     -6.62E+01,      4.27E-01,       -6.41E-04,      3.01E-07,       -1.46,      0.636,      -573.56,    2.307],
        [0.0113,    -0.0028,    56,     18.18,      -4.32,      -9.63,      3.77,       2.36E+01,       -3.81E-02,      1.72E-04,       -1.03E-07,      -0.473,     1.724,      495.01,     -1.539],
        [0.0129,    -0.0006,    46,     24.96,      8.73,       37.97,      48.53,      -8,             1.05E-01,       -9.63E-05,      3.56E-08,       2.691,      2.205,      82.28,      -0.242],
        [0.0117,    0.0011,     38,     24.14,      11.14,      83.99,      92.36,      -2.81E+01,      2.08E-01,       -3.06E-04,      1.46E-07,       3.063,      2.138,      'NA',       'NA'],
        [0.0026,    0.0028,     36,     26.15,      17.78,      142.14,     136.7,      2.74E+01,       -5.57E-02,      1.01E-04,       -5.02E-08,      4.72,       2.661,      'NA',       'NA'],
        [0.0027,    -0.0008,    46,     9.2,        -11.18,     79.3,       77.71,      2.45E+01,       -2.71E-02,      1.11E-04,       -6.78E-08,      2.322,      1.155,      'NA',       'NA'],
        [0.002,     0.0016,     37,     27.38,      64.32,      115.51,     109.82,     7.87,           2.01E-02,       -8.33E-06,      1.39E-09,       4.151,      3.302,      'NA',       'NA'],
        [0.01,      0.0025,     48,     27.15,      7.75,       -26.8,      -3.68,      -6.03,          8.54E-02,       -8.00E-06,      -1.80E-08,      0.49,       2.398,      307.53,     -0.798],
        [0.0122,    0.0004,     38,     21.78,      19.88,      8.67,       40.99,      -2.05E+01,      1.62E-01,       -1.60E-04,      6.24E-08,       3.243,      1.942,      -394.29,    1.251],
        [0.0042,    0.0061,     27,     21.32,      60.15,      79.72,      87.88,      -9.09E+01,      5.57E-01,       -9.00E-04,      4.69E-07,       -1.373,     0.644,      'NA',       'NA'],
        [0.0082,    0.0011,     41,     26.73,      8.13,       2.09,       11.3,       -2.14,          5.74E-02,       -1.64E-06,      -1.59E-08,      1.101,      2.544,      259.65,     -0.702],
        [0.0143,    0.0008,     32,     31.01,      37.02,      46.43,      54.05,      -8.25,          1.01E-01,       -1.42E-04,      6.78E-08,       2.394,      3.059,      -245.74,    0.912],
        [0.0111,    -0.0057,    27,     -0.03,      -15.78,     -251.92,    -247.19,    2.65E+01,       -9.13E-02,      1.91E-04,       -1.03E-07,      1.398,      -0.67,      'NA',       'NA'],
        [0.0105,    -0.0049,    58,     38.13,      13.55,      -71.55,     -64.31,     3.33E+01,       -9.63E-02,      1.87E-04,       -9.96E-08,      2.515,      4.532,      625.45,     -1.814],
        [0.0133,    0.0057,     71,     66.86,      43.43,      -29.48,     -38.06,     2.86E+01,       -6.49E-02,      1.36E-04,       -7.45E-08,      3.603,      6.582,      738.91,     -2.038],
        [0.0068,    -0.0034,    97,     93.84,      41.69,      21.06,      5.74,       3.21E+01,       -6.41E-02,      1.26E-04,       -6.87E-08,      2.724,      9.52,       809.55,     -2.224],
        [0.0741,    0.0112,     28,     92.88,      44.45,      -208.04,    -189.2,     2.57E+01,       -6.91E-02,      1.77E-04,       -9.88E-08,      2.406,      16.826,     2173.72,    -5.057],
        [0.024,     0.0184,     -25,    76.34,      82.83,      -221.65,    -197.37,    -2.81,          1.11E-01,       -1.16E-04,      4.94E-08,       4.49,       12.499,     3018.17,    -7.314],
        [0.0168,    0.0015,     18,     22.42,      22.23,      -132.22,    -105,       2.55E+01,       -6.32E-02,      1.11E-04,       -5.48E-08,      1.188,      2.41,       122.09,     -0.386],
        [0.0098,    0.0048,     13,     31.22,      23.05,      -138.16,    -98.22,     1.22E+01,       -1.26E-02,      6.03E-05,       -3.86E-08,      5.879,      4.682,      440.24,     -0.953],
        [0.038,     0.0031,     62,     76.75,      61.2,       -133.22,    -120.5,     6.45,           6.70E-02,       -3.57E-05,      2.86E-09,       4.189,      8.972,      340.35,     -0.35],
        [0.0284,    0.0028,     55,     94.97,      75.97,      -164.5,     -126.27,    3.04E+01,       -8.29E-02,      2.36E-04,       -1.31E-07,      'NA',       6.645,      'NA',       'NA'],
        [0.0379,    0.003,      82,     72.24,      36.9,       -162.03,    -143.48,    3.09E+01,       -3.36E-02,      1.60E-04,       -9.88E-08,      3.197,      9.093,      740.92,     -1.713],
        [0.0791,    0.0077,     89,     169.09,     155.5,      -426.72,    -387.87,    2.41E+01,       4.27E-02,       8.04E-05,       -6.87E-08,      11.051,     19.537,     1317.23,    -2.578],
        [0.0481,    0.0005,     82,     81.1,       53.6,       -337.92,    -301.95,    2.45E+01,       4.02E-02,       4.02E-05,       -4.52E-08,      6.959,      9.633,      483.88,     -0.966],
        [0.0143,    0.0101,     36,     -10.5,      2.08,       -247.61,    -250.83,    6.82,           1.96E-02,       1.27E-05,       -1.78E-08,      3.624,      5.909,      675.24,     -1.34],
        [0.0243,    0.0109,     38,     73.23,      66.89,      -22.02,     14.07,      2.69E+01,       -4.12E-02,      1.64E-04,       -9.76E-08,      3.515,      10.788,     'NA',       'NA'],
        [0.0295,    0.0077,     35,     50.17,      52.66,      53.47,      89.39,      -1.21,          7.62E-02,       -4.86E-05,      1.05E-08,       5.099,      6.436,      'NA',       'NA'],
        [0.013,     0.0114,     29,     52.82,      101.51,     31.65,      75.61,      1.18E+01,       -2.30E-02,      1.07E-04,       -6.28E-08,      7.49,       6.93,       'NA',       'NA'],
        [0.0169,    0.0074,     9,      11.74,      48.84,      123.34,     163.16,     -3.11E+01,      2.27E-01,       -3.20E-04,      1.46E-07,       4.703,      1.896,      'NA',       'NA'],
        [0.0255,    -0.0099,    'NA',   74.6,       'NA',       23.61,      'NA',       'NA',           'NA',           'NA',           'NA',           'NA',       3.335,      'NA',       'NA'],
        [0.0085,    0.0076,     34,     57.55,      68.4,       55.52,      79.93,      8.83,           -3.84E-03,      4.35E-05,       -2.60E-08,      3.649,      6.528,      'NA',       'NA'],
        ['NA',      'NA',       'NA',   83.08,      68.91,      93.7,       119.66,     5.69,           -4.12E-03,      1.28E-04,       -8.88E-08,      'NA',       12.169,     'NA',       'NA'],
        [0.0496,    -0.0101,    91,     125.66,     59.89,      88.43,      89.22,      3.65E+01,       -7.33E-02,      1.84E-04,       -1.03E-07,      2.414,      12.851,     'NA',       'NA'],
        [0.0437,    0.0064,     91,     152.54,     127.24,     -66.57,     -16.83,     2.59E+01,       -3.74E-03,      1.29E-04,       -8.88E-08,      9.679,      16.738,     'NA',       'NA'],
        [0.0031,    0.0084,     63,     63.56,      20.09,      -17.33,     -22.99,     3.53E+01,       -7.58E-02,      1.85E-04,       -1.03E-07,      2.36,       6.884,      'NA',       'NA'],
        [0.0119,    0.0049,     54,     68.78,      34.4,       41.87,      33.12,      1.96E+01,       -5.61E-03,      4.02E-05,       -2.76E-08,      4.13,       6.817,      'NA',       'NA'],
        [0.0019,    0.0051,     38,     52.1,       79.93,      39.1,       27.76,      1.67E+01,       4.81E-03,       2.77E-05,       -2.11E-08,      1.557,      5.984,      'NA',       'NA']
        ]

        #Number of functional groups
        for i in range(0, 41):
            n[i] = len(molecule.GetSubstructMatches(SMARTS[i]))

        #Calculations
        #Number of atoms
        molecule_with_Hs = Chem.AddHs(molecule)
        na = molecule_with_Hs.GetNumAtoms()

        #Molecular weight
        MW = Descriptors.MolWt(molecule)

        #Boiling point
        # TB = 198.2
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][3] == 'NA':
        #             print('Warning: There is not available parameter in boiling point calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             TB = TB + n[i] * p[i][3]

        #Critical temperature
        # TC1 = 0; TC2 = 0
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][0] == 'NA':
        #             print('Warning: There is not available parameter in critical temperature calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             TC1 = TC1 + n[i] * p[i][0]
        #             TC2 = TC2 + n[i] * p[i][0]
        #     TC = TB * (0.584 + 0.965 * TC1 - TC2**2)**-1

        #Critical pressure
        # PC1 = 0
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][1] == 'NA':
        #             print('Warning: There is not available parameter in critical pressure calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             PC1 = PC1 + n[i] * p[i][1]
        #     PC = (0.113 + 0.0032 * na - PC1)**-2
        
        #Critical volume
        VC = 17.5
        for i in range(0, 41):
            if n[i] != 0:
                if p[i][2] == 'NA':
                    print('\nWarning: There is not available parameter in critical volume calculation (Group %0.0f,'%i, group[i], ')')
                    print('\nRemark: Contribution from group %0.0f,'%i, group[i], 'will be discarded from calculations. Molar volume value will be affected.')
                else:
                    VC = VC + n[i] * p[i][2]

        #Heat of formation (ideal gas, 298 K)
        # Hform = 68.29
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][5] == 'NA':
        #             print('Warning: There is not available parameter in heat of formation calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             Hform = Hform + n[i] * p[i][5]

        #Gibbs energy of formation (ideal gas, 298 K)
        # Gform = 53.88
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][6] == 'NA':
        #             print('Warning: There is not available parameter in Gibbs energy of formation calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             Gform = Gform + n[i] * p[i][6]

        #Ideal gas heat capacity
        # a = 0; b = 0; c = 0; d = 0
        # for i in range(0, 41):
        #     if n[i] != 0:
        #         if p[i][7] == 'NA':
        #             print('Warning: There is not available parameter in ideal gas heat capacity calculation (Group %0.0f,'%i, group[i], ')')
        #         else:
        #             a = a + n[i] * p[i][7]
        #             b = b + n[i] * p[i][8]
        #             c = c + n[i] * p[i][9]
        #             d = d + n[i] * p[i][10]
                    
                    
        
        
        #Unit conversion
        # TB = TB - 273.15        # K to C
        # TC = TC - 273.15        # K to C
        # Hform = Hform / 4.1868   # kJ/mol to kcal/mol
        # Gform = Gform / 4.1868   # kJ/mol to kcal/mol

        #CPIG parameter estimation (
        # c1 = (a - 37.93) / 4.1868          # J/mol-K to cal/mol-K
        # c2 = (b + 0.21) / 4.1868           # J/mol-K to cal/mol-K
        # c3 = (c - 0.000391) / 4.1868       # J/mol-K to cal/mol-K
        # c4 = (d + 0.000000206) / 4.1868    # J/mol-K to cal/mol-K

        # C1 = c1 + 273.15 * c2 + 273.15**2 * c3 + 273.15**3 * c4
        # C2 = c2 + 2 * 273.15 * c3 + 3 * 273.15**2 * c4
        # C3 = c3 + 3 * 273.15 * c4
        # C4 = c4
        # C5 = 0
        # C6 = 0
        # C7 = 6.85
        # C8 = 826.85
        # C9 = 8.60543
        # C10 = ((C1 + C2 * C7 + C3 * C7**2 + C4 * C7**3) - C9) / 280**1.5
        # C11 = 1.5

        # CPIG = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11]

        #Empirical formula to calculate Molar Volume at normal boiling point [cm3 gmol-1]
        Vm = 0.285 * (VC)**1.048

        return Vm #[MW, TB, TC, PC, VC, Hform, Gform, CPIG]
    



class StokesRadiusCalculator:
    @staticmethod
    def stokes_einstein_radius(D,T,eta):
        """
        Calculate the radius of a particle from the diffusion coefficient using the Stokes-Einstein equation.

        Parameters:
        D (float): Diffusion coefficient (cm²/s)
        T (float): Absolute temperature (K)
        eta (float): Dynamic viscosity of the fluid (cP)

        Returns:
        float: Radius of the particle (nm)
        """
        # Boltzmann constant in J/K
        k_B = 1.380649e-23
        # Convert dynamic viscosity from cP to kg/(m·s) (or Pa·s)
        eta = eta*0.001
        
        # Convert diffusion coefficient from cm²/s to m²/s
        D = D * 1e-4
        
        # Calculate the radius
        r = k_B * T / (6 * np.pi * eta * D)
        r = r/(1e-9) # Convert radius from m to nm

        return r


class DiffusivityCalculator:
    @staticmethod
    def wilke_chang_diffusion_coefficient(molar_volume, molecular_weight, temp, viscosity, alpha):
        d = (7.4E-8) * temp * np.sqrt(alpha * molecular_weight) / (viscosity * (alpha * molar_volume)**0.6)
        return d



@dataclass
class Solvent:
    name: str
    molecular_weight: float
    viscosity: float
    alpha: float


class Solvents:
    _solvents: Dict[str, Solvent] = {
        "water": Solvent(name="water", molecular_weight=18.01528, viscosity=0.001, alpha=2.6),
        "methanol": Solvent(name="methanol", molecular_weight=32.042, viscosity=0.0006, alpha=1.9),
        "ethanol": Solvent(name="ethanol", molecular_weight=46.069, viscosity=0.0012, alpha=1.5),
        "ethyl_acetate": Solvent(name="ethyl_acetate", molecular_weight=88.106, viscosity=0.00046, alpha=1.0),
        "acetone": Solvent(name="acetone", molecular_weight=58.08, viscosity=0.0003, alpha=1.06),
        "toluene": Solvent(name="toluene", molecular_weight=92.141, viscosity=0.0006, alpha=1.0),
        "hexane": Solvent(name="hexane", molecular_weight=86.177, viscosity=0.00032, alpha=1.0),
        "heptane": Solvent(name="heptane", molecular_weight=100.204, viscosity=0.00042, alpha=1.0),
        "acetonitrile": Solvent(name="acetonitrile", molecular_weight=41.053, viscosity=0.00037, alpha=1.48),
        "anisol": Solvent(name="anisol", molecular_weight=108.14, viscosity=0.0012, alpha=1.02),
        "dioxane": Solvent(name="dioxane", molecular_weight=88.106, viscosity=0.0012, alpha=1.07),
        "pyridine": Solvent(name="pyridine", molecular_weight=79.101, viscosity=0.00095, alpha=1.24),
        "aniline": Solvent(name="aniline", molecular_weight=93.13, viscosity=0.00102, alpha=1.43),
        "propanol": Solvent(name="propanol", molecular_weight=60.096, viscosity=0.0022, alpha=1.5),
        "butyl_acetate": Solvent(name="butyl_acetate", molecular_weight=116.161, viscosity=0.000732, alpha=1.0),
        "other": Solvent(name="other", molecular_weight=None, viscosity=None, alpha=1.0),
    }

    @classmethod
    def get(cls, name: str) -> Solvent:
        if name not in cls._solvents:
            raise ValueError(f"Solvent '{name}' is not defined.")
        return cls._solvents[name]
    
class AimarMethods:
    @staticmethod
    def local_rejection(lambda_star, r_dimless):
        """
        Local rejection formula for a single pore of dimensionless radius r_dimless.
        The dimensionless solute radius is lambda_star = a / r*, where r* is the mean (log-normal) pore radius.
        
        If a > r (i.e. lambda_star > r_dimless), the pore is 'too small' and blocks the solute => R=1.
        Otherwise, use the expression [1 - (1 - a/r)^2]^2.
        """
        if lambda_star > r_dimless:
            return 1.0
        else:
            return (1.0 - (1.0 - lambda_star / r_dimless)**2) ** 2

    @staticmethod
    def lognormal_pdf(r_dimless, sigma):
        """
        The 'log-normal-like' weight function for pore radii in dimensionless form (r_dimless = r / r*).
        Only the shape matters here, so the normalization or amplitude is accounted for in the integrals.
        
        exp{ - [ln(r_dimless) / ln(sigma)]^2 }
        """
        return exp(-(log(r_dimless) / log(sigma))**2)