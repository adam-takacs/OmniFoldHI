# Simple closure test in which ibu is performed in 2-dimensions 
# on independent Gaussian distributions.

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages

from ibu import MChistogram, MChistogram2d, MChistogramdd, ibu_1d_HI, ibu_dd_HI

# Generating Measured and Truth events (Effect and Cause).
Nev = 10**4 # number of events
par_c = [[0, 2], 
         [0, 2]] # Gaussian parameters
evs_c = [np.random.normal(*i, Nev) for i in par_c]
par_e = [7, 3]
evs_e = [np.array([i + np.random.normal(*par_e) for i in evs]) for evs in evs_c]
evs_e = [np.array([(i if (-10<=i<=10) else -11) for i in obs]) for obs in evs_e]
# Defining the bins.
BinList_c = [[-10, 10, 10],
             [-10, 10, 15]]
bins_c    = [np.linspace(xmin,xmax,Nbins+1) for xmin,xmax,Nbins in BinList_c]
BinList_e = [[-10, 10, 20],
             [-10, 10, 25]]
bins_e    = [np.linspace(xmin,xmax,Nbins+1) for xmin,xmax,Nbins in BinList_e]
probability=True

# Generating MC events.
Nev_mc = 10**5 # number of events
par_c_mc = [[0, 2], 
            [0, 2]] # MC parameters (doesn't hvat to be the same as Truth!)
temp_c   = [np.random.normal(*i, Nev_mc) for i in par_c_mc]
# MC generated events know about Trash and Fake.
evs_c_mc = [np.array([(i if (-10<=i<=10) else -11) for i in obs]) for obs in temp_c]
temp_e   = [np.array([i + np.random.normal(*par_e) for i in obs]) for obs in temp_c]
evs_e_mc = [np.array([(i if (-10<=i<=10) else -11) for i in obs]) for obs in temp_e]
# Defining MC bins.
BinList_c_mc = [[-10, 10, 10],
                [-10, 10, 15]]
val_fake_mc  = -11
bins_c_mc    = [np.linspace(xmin,xmax,Nbins+1) for xmin,xmax,Nbins in BinList_c_mc]
BinList_e_mc = [[-10, 10, 20],
                [-10, 10, 25]]
val_trash_mc = -11
bins_e_mc    = [np.linspace(xmin,xmax,Nbins+1) for xmin,xmax,Nbins in BinList_e_mc]

# Multi-dim ibu is performed here.
testdd = ibu_dd_HI(evs_e_mc, evs_c_mc, [bins_e_mc,bins_c_mc], -11, -11, \
                 evs_e=evs_e, bins_e=bins_e, out_probability=probability, \
                 ibu_itnum=10)
posdd_err = testdd.ibu_err()

# Bin width will be needed in various shapes.
dbin_c = [np.diff(j) for j in posdd_err[-1]]
for i in range(len(dbin_c)-1):
    if i==0: dbin_c_outer = dbin_c[i]
    dbin_c_outer = np.expand_dims(dbin_c_outer, axis=-1) * dbin_c[i+1]


with PdfPages('test_ibu_dd.pdf') as pdf:
    
    for i in range(len(bins_e)):
        # Loop through the observables. 

        # Measured histograms.
        meas = MChistogram(evs_e[i].tolist(), bins_e[i], density=probability)

        # 1d ibu is performed here.
        test = ibu_1d_HI(evs_e_mc[i].tolist(), evs_c_mc[i].tolist(), [bins_e_mc[i],bins_c_mc[i]], -11, -11, \
                    evs_e=evs_e[i].tolist(), bins_e=bins_e[i], out_probability=probability, \
                    ibu_itnum=10, \
                    smoothing=False, smoothing_method='gauss')
        pos_err = test.ibu_err()    

        # Plotting.
        fig = plt.figure()
        ax  = fig.add_subplot()

        bmid_c = bins_c[i][:-1] + np.diff(bins_c[i])/2
        plt.plot(bmid_c, (1 if probability else Nev)*norm(*par_c[i]).pdf(bmid_c),'C0:',label='Truth')
        plt.errorbar(meas[-1][:-1]+np.diff(meas[-1])/2, meas[0], 
                    yerr=meas[1], xerr=np.diff(meas[-1])/2, fmt='o', c='C1', label='Measure')
        plt.errorbar(pos_err[-1][:-1]+np.diff(pos_err[-1])/2, pos_err[0], 
                    yerr=pos_err[1], xerr=np.diff(pos_err[-1])/2, fmt='o', c='C3', label='ibu 1d')
        
        # Projecting the multi-dim posterior to 1d.
        axis = tuple([j for j in range(len(posdd_err[-1])) if j!=i])
        ax.errorbar(posdd_err[-1][i][:-1]+np.diff(posdd_err[-1][i])/2,  (posdd_err[0] * dbin_c_outer).sum(axis=axis)/dbin_c[i],  (posdd_err[1] * dbin_c_outer).sum(axis=axis)/dbin_c[i], np.diff(posdd_err[-1][i])/2, fmt='.', c='C2', label=r'ibu dd')

        ax.legend()
        ax.set_xlabel(r'$C_{%i}$' %(i+1), fontsize=18)
        ax.set_ylabel(r'$x(C_{%i})$' %(i+1),fontsize=18)
        plt.tight_layout()   
        pdf.savefig()
        plt.close()