import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle

import random
from tqdm import tqdm

import imf

def cut_isochrone(MG, d, Av, f_ext_G, Teff, Mlimit=18):
    limit = MG + d + Av*f_ext_G(10**Teff)   
    w = limit<=(Mlimit+0.1)
    return w

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def d_to_m(d):
    m = 5*np.log10(d) - 5
    return m

def m_to_d(m):
    d = 10**(1+m/5)
    return d

def IMF_law(m):
    
    m1 = m[m<=0.08]
    m2 = m[(m>0.08)&(m<=0.5)]
    m3 = m[m>0.5]

    p1 = 0.3
    p2 = 1.3
    p3 = 2.3

    B = 0.5**(-p3) / 0.5**(-p2)
    A = B * (0.08**(-p2)/0.08**(-p1))


    value1 = A * m1**(-p1)
    value2 = B * m2**(-p2)
    value3 = m3**(-p3)
    norm = 12.297830547050037
    value = np.concatenate((value1, value2, value3))/norm
    return value

def load_photometric_errors():

    sss = pd.read_csv('Gaia_DR3_phot_uncertanties/LogErrVsMagSpline.csv')

    x_G = sss['knots_G'].values
    y_G = sss['coeff_G'].values

    y_G[3] = np.mean(y_G[0:4])
    x_G[:3]= np.nan
    y_G[:3]= np.nan
    x_G[-4:] = np.nan
    y_G[-4:] = np.nan

    x_BP= sss['knots_BP'].values
    y_BP= sss['coeff_BP'].values

    x_BP = x_BP[3:-16]
    y_BP = y_BP[3:-16]

    x_RP= sss['knots_RP'].values
    y_RP= sss['coeff_RP'].values

    x_RP = x_RP[3:-16]
    y_RP = y_RP[3:-16]

    f_G_err = interpolate.interp1d(x_G[3:-4], 10**y_G[3:-4], kind='quadratic',bounds_error=False, fill_value=np.mean(10**y_G[3:15]))
    f_BP_err= interpolate.interp1d(x_BP, 10**y_BP, kind='quadratic',bounds_error=False          , fill_value=np.mean(10**y_BP[:5]))
    f_RP_err= interpolate.interp1d(x_RP, 10**y_RP, kind='quadratic',bounds_error=False          , fill_value=np.mean(10**y_RP[:5]))

    unc_synthetic = pd.read_csv('Gaia_DR3_phot_uncertanties/uncertanties_synthetic.csv')

    sigma_b = np.zeros(unc_synthetic.G.values.shape[0])
    sigma_v = np.zeros(unc_synthetic.G.values.shape[0])
    sigma_r = np.zeros(unc_synthetic.G.values.shape[0])
    sigma_i = np.zeros(unc_synthetic.G.values.shape[0])

    for i in range(unc_synthetic.G.values.shape[0]):
        sigma_b[i] = np.mean([unc_synthetic['P50(∆B)'].values[i] - unc_synthetic['P16(∆B)'].values[i], unc_synthetic['P84(∆B)'].values[i] - unc_synthetic['P50(∆B)'].values[i]])*10**(-3)
        sigma_v[i] = np.mean([unc_synthetic['P50(∆V)'].values[i] - unc_synthetic['P16(∆V)'].values[i], unc_synthetic['P84(∆V)'].values[i] - unc_synthetic['P50(∆V)'].values[i]])*10**(-3)
        sigma_r[i] = np.mean([unc_synthetic['P50(∆R)'].values[i] - unc_synthetic['P16(∆R)'].values[i], unc_synthetic['P84(∆R)'].values[i] - unc_synthetic['P50(∆R)'].values[i]])*10**(-3)
        sigma_i[i] = np.mean([unc_synthetic['P50(∆I)'].values[i] - unc_synthetic['P16(∆I)'].values[i], unc_synthetic['P84(∆I)'].values[i] - unc_synthetic['P50(∆I)'].values[i]])*10**(-3)

    sigma_b[:6] = np.full(6, np.mean(sigma_b[:6]))

    f_b_err = interpolate.interp1d(unc_synthetic.G.values, sigma_b, kind='quadratic', bounds_error=False, fill_value=0)
    f_v_err = interpolate.interp1d(unc_synthetic.G.values, sigma_v, kind='quadratic', bounds_error=False, fill_value=0)
    f_r_err = interpolate.interp1d(unc_synthetic.G.values, sigma_r, kind='quadratic', bounds_error=False, fill_value=0)
    f_i_err = interpolate.interp1d(unc_synthetic.G.values, sigma_i, kind='quadratic', bounds_error=False, fill_value=0)

    return f_G_err, f_BP_err, f_RP_err, f_b_err, f_v_err, f_r_err, f_i_err

def extract_masses(IMF, MASSES,f_IMF, Nstars, Mcluster):

    #22-Oct-2024

    Mmin = MASSES.min()
    if Nstars == None:
        IMF = imf.Kroupa
        selected_imf = IMF(mmin=0.03, mmax=MASSES.max())

        mass_ext = imf.make_cluster(Mcluster, massfunc=selected_imf, silent=True)
        mass_ext = mass_ext[mass_ext>=Mmin]

        while ((Mcluster-mass_ext.sum())>(Mcluster*10**(-4))):
            mass_ext = np.append(mass_ext,imf.make_cluster(Mcluster-mass_ext.sum(), massfunc=selected_imf, silent=True))
            mass_ext = mass_ext[mass_ext>=Mmin]
        
    if Mcluster == None:

        rand = np.random.uniform(0,1,Nstars)
        ext = f_IMF(rand)

        mass_ext = ext
    
    return mass_ext

def load_ext_curves(Av):
    with open('Extintion/G/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_G = pickle.load(file)
    with open('Extintion/BP/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_BP = pickle.load(file)
    with open('Extintion/RP/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_RP = pickle.load(file)
    with open('Extintion/u/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_u = pickle.load(file)
    with open('Extintion/b/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_b = pickle.load(file)
    with open('Extintion/v/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_v = pickle.load(file)
    with open('Extintion/r/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_r = pickle.load(file)
    with open('Extintion/i/f_av%1.2f.pkl' % Av, 'rb') as file:
        f_ext_i = pickle.load(file)
    return f_ext_G, f_ext_BP, f_ext_RP, f_ext_u, f_ext_b, f_ext_v, f_ext_r, f_ext_i

import pandas as pd

class Isochrone:
    def __init__(self, FeH, logage, Av, distance):
        self.FeH = FeH
        self.logage = logage
        self.Av = Av
        self.dMod = d_to_m(distance)

        self.load_data()
        self.load_extinction_curve()
        self.load_photometric_errors()

    def load_data(self):
        # Load data from the corresponding CSV file
        filename = f'PARSEC/ALL/{self.FeH:.2f}_{self.logage:.2f}.csv'
        df = pd.read_csv(filename)
        
        # Extract necessary columns
        self.MASSES = df.Mini.values
        self.Mact = df.Mass.values
        self.MG = df.G.values
        self.MG_BP = df.BP.values
        self.MG_RP = df.RP.values
        self.Teff = df.logTe.values
        
        self.Mu = df.u.values
        self.Mb = df.b.values
        self.Mv = df.v.values
        self.Mr = df.r.values
        self.Mi = df.i.values

    def load_extinction_curve(self):
        # Load extinction curves based on Av
        self.f_ext_G, self.f_ext_BP, self.f_ext_RP, \
        self.f_ext_u, self.f_ext_b, self.f_ext_v, \
        self.f_ext_r, self.f_ext_i = load_ext_curves(self.Av)

    def load_photometric_errors(self):
        # Load photometric errors
        self.f_G_err, self.f_BP_err, self.f_RP_err, \
        self.f_b_err, self.f_v_err, self.f_r_err, \
        self.f_i_err = load_photometric_errors()

    def filter_isochrone(self):
        # Apply filter to keep only stars brighter than G=20
        w = cut_isochrone(self.MG, self.dMod, self.Av, self.f_ext_G, self.Teff, Mlimit=20)
        
        # Update attributes based on the filter
        self.MASSES = self.MASSES[w]
        self.Mact = self.Mact[w]
        self.MG = self.MG[w]
        self.MG_BP = self.MG_BP[w]
        self.MG_RP = self.MG_RP[w]
        self.Teff = self.Teff[w]
        self.Mu = self.Mu[w]
        self.Mb = self.Mb[w]
        self.Mv = self.Mv[w]
        self.Mr = self.Mr[w]
        self.Mi = self.Mi[w]

    def compute_apparent_magnitude(self):

        self.MG    = self.MG    + self.Av*self.f_ext_G(10**self.Teff)  + self.dMod
        self.MG_BP = self.MG_BP + self.Av*self.f_ext_BP(10**self.Teff) + self.dMod
        self.MG_RP = self.MG_RP + self.Av*self.f_ext_RP(10**self.Teff) + self.dMod

        self.Mu = self.Mu #+ Av*self.f_ext_G(10**self.Teff)   + d
        self.Mb = self.Mb + self.Av*self.f_ext_b(10**self.Teff) + self.dMod
        self.Mv = self.Mv + self.Av*self.f_ext_v(10**self.Teff) + self.dMod
        self.Mr = self.Mr + self.Av*self.f_ext_r(10**self.Teff) + self.dMod
        self.Mi = self.Mi + self.Av*self.f_ext_i(10**self.Teff) + self.dMod

    def add_extiction(self):
        self.MG    = self.MG    + self.Av*self.f_ext_G(10**self.Teff) 
        self.MG_BP = self.MG_BP + self.Av*self.f_ext_BP(10**self.Teff)
        self.MG_RP = self.MG_RP + self.Av*self.f_ext_RP(10**self.Teff)

        self.Mu = self.Mu #+ Av*self.f_ext_G(10**self.Teff)   + d
        self.Mb = self.Mb + self.Av*self.f_ext_b(10**self.Teff)
        self.Mv = self.Mv + self.Av*self.f_ext_v(10**self.Teff)
        self.Mr = self.Mr + self.Av*self.f_ext_r(10**self.Teff)
        self.Mi = self.Mi + self.Av*self.f_ext_i(10**self.Teff)


    def get_data(self):
        # Returns the filtered data as a dictionary
        return pd.DataFrame({
            'Mint': self.MASSES,
            'Mact': self.Mact,
            'MG': self.MG,
            'MBP': self.MG_BP,
            'MRP': self.MG_RP,
            'Teff': self.Teff,
            'Mu': self.Mu,
            'Mb': self.Mb,
            'Mv': self.Mv,
            'Mr': self.Mr,
            'Mi': self.Mi
        })

    def plot_isochrone(self, color='BP-RP', magnitude='G', absolute=False, c='k', cut=20.1):
        
        if absolute==False:
            self.filter_isochrone()
            self.compute_apparent_magnitude()
            iso = self.get_data()

            plt.ylabel(magnitude)
        else:
            self.add_extiction()
            iso = self.get_data()
            plt.ylabel('ABS' + magnitude)
        
        color_ = iso['M' + color.split('-')[0]] - iso['M' + color.split('-')[1]]
        magnitude_ = iso['M' + magnitude]
        
        w = magnitude_< cut

        plt.plot(color_[w], magnitude_[w], c=c)
        plt.gca().invert_yaxis()
        
        
        plt.xlabel(color)
        