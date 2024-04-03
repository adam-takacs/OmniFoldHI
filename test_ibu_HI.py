# Simple closure test in which ibu is performed in 2-dimensions 
# on independent Gaussian distributions.

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages

from ibu import MChistogram, MChistogram2d, MChistogramdd, ibu_1d_HI, ibu_dd_HI

# READING THE DATASETS: hardest jet in the event [ptjet, thg, ktg]
print('Reading in datasets.')
# data_1: Pythia8 default tune + thermal background.
dat_c = np.loadtxt("datasets/Pythia_hardestJet_250GeV_Sig.out")
dat_e = np.loadtxt("datasets/Pythia_hardestJet_250GeV_Dat.out")
dat_e_strip = np.array([i for i in dat_e if all(([250, 0, 0]<=i) & (i<=[np.inf, 0.4, 60]))])
data_1 = {'evs_c_mc_ptjet': [(i if (250<=i)    else -1) for i in dat_c.T[0]], 
          'evs_c_mc_thg':   [(i if (0<=i<=0.4) else -1) for i in dat_c.T[1]], 
          'evs_c_mc_ktg':   [(i if (0<=i<=60)  else -1) for i in dat_c.T[2]],
          'evs_e_mc_ptjet': [(i if (250<=i)    else -1) for i in dat_e.T[0]], 
          'evs_e_mc_thg':   [(i if (0<=i<=0.4) else -1) for i in dat_e.T[1]], 
          'evs_e_mc_ktg':   [(i if (0<=i<=60)  else -1) for i in dat_e.T[2]],
          'evs_c_ptjet':    [(i if (250<=i)    else -1) for i in dat_c.T[0]], 
          'evs_c_thg':      [(i if (0<=i<=0.4) else -1) for i in dat_c.T[1]], 
          'evs_c_ktg':      [(i if (0<=i<=60)  else -1) for i in dat_c.T[2]],
          'evs_e_ptjet':    [i for i in dat_e_strip.T[0]], 
          'evs_e_thg':      [i for i in dat_e_strip.T[1]], 
          'evs_e_ktg':      [i for i in dat_e_strip.T[2]]}
# data_2: Herwig7 default tune + thermal background
dat_c = np.loadtxt("datasets/Herwig_hardestJet_250GeV_Sig.out")
dat_e = np.loadtxt("datasets/Herwig_hardestJet_250GeV_Dat.out")
dat_e_strip = np.array([i for i in dat_e if all(([250, 0, 0]<=i) & (i<=[np.inf, 0.4, 60]))])
data_2 = {'evs_c_mc_ptjet': [(i if (250<=i)    else -1) for i in dat_c.T[0]], 
          'evs_c_mc_thg':   [(i if (0<=i<=0.4) else -1) for i in dat_c.T[1]], 
          'evs_c_mc_ktg':   [(i if (0<=i<=60)  else -1) for i in dat_c.T[2]],
          'evs_e_mc_ptjet': [(i if (250<=i)    else -1) for i in dat_e.T[0]], 
          'evs_e_mc_thg':   [(i if (0<=i<=0.4) else -1) for i in dat_e.T[1]], 
          'evs_e_mc_ktg':   [(i if (0<=i<=60)  else -1) for i in dat_e.T[2]],
          'evs_c_ptjet':    [(i if (250<=i)    else -1) for i in dat_c.T[0]], 
          'evs_c_thg':      [(i if (0<=i<=0.4) else -1) for i in dat_c.T[1]], 
          'evs_c_ktg':      [(i if (0<=i<=60)  else -1) for i in dat_c.T[2]],
          'evs_e_ptjet':    [i for i in dat_e_strip.T[0]], 
          'evs_e_thg':      [i for i in dat_e_strip.T[1]], 
          'evs_e_ktg':      [i for i in dat_e_strip.T[2]]}
print('Done.')

print('Defining unfolding procedure.')
# choose what is MC and Data in this context
synthetic, nature = data_1, data_2

# list of observables to study
obs_multifold = ['JetPT', 'Thg', 'Ktg']
# Define histograms
nbins = [15, 20, 15]
xlim  = [(250, 1000), (0, 0.445), (0, 64.29)]
posterior_is_probability = True

# a dictionary to hold information about the observables
obs = {}
# the jet pt and histogram style information
obs.setdefault('JetPT', {}).update({
    'evs': lambda dset, ptype: dset[ptype + '_ptjet'],
    'nbins_e': nbins[0], 'xlim_e': xlim[0],
    'nbins_c': nbins[0], 'xlim_c': xlim[0],
    'bins_e': np.linspace(*xlim[0], nbins[0]+1),
    'bins_c': np.linspace(*xlim[0], nbins[0]+1),
    'nbins_e_mc': nbins[0], 'xlim_e_mc': xlim[0],
    'nbins_c_mc': nbins[0], 'xlim_c_mc': xlim[0],
    'bins_e_mc': np.linspace(*xlim[0], nbins[0]+1),
    'bins_c_mc': np.linspace(*xlim[0], nbins[0]+1),
    'xlim': (250, 800), 'ylim': (-5e-4, 1e-2),
    'xlabel': r'$p_T$ [GeV]', 'symbol': r'$p_T$ [GeV]',
    'ylabel': r'$1/N_{ev}\, dN/dp_T$ [1/GeV]',
    'ibu_smoothing': False,
    'ibu_itnum': 10,
})
# the dynamically groomed angle and histogram style information
obs.setdefault('Thg', {}).update({
    'evs': lambda dset, ptype: dset[ptype + '_thg'],
    'nbins_e': nbins[1], 'xlim_e': xlim[1],
    'nbins_c': nbins[1], 'xlim_c': xlim[1],
    'bins_e': np.linspace(*xlim[1], nbins[1]+1),
    'bins_c': np.linspace(*xlim[1], nbins[1]+1),
    'nbins_e_mc': nbins[1], 'xlim_e_mc': xlim[1],
    'nbins_c_mc': nbins[1], 'xlim_c_mc': xlim[1],
    'bins_e_mc': np.linspace(*xlim[1], nbins[1]+1),
    'bins_c_mc': np.linspace(*xlim[1], nbins[1]+1),
    'xlim': (0, 0.4), 'ylim': (-0.5, 12),
    'xlabel': r'$\theta_g$', 'symbol': r'$\theta_g$',
    'ylabel': r'$1/N_{ev}\, dN/d\theta_g$',
    'ibu_smoothing': False,
    'ibu_itnum': 10,
})
# the dynamically groomed transverse momentum and histogram style information
obs.setdefault('Ktg', {}).update({
    'evs': lambda dset, ptype: dset[ptype + '_ktg'],
    'nbins_e': nbins[2], 'xlim_e': xlim[2],
    'nbins_c': nbins[2], 'xlim_c': xlim[2],
    'bins_e': np.linspace(*xlim[2], nbins[2]+1),
    'bins_c': np.linspace(*xlim[2], nbins[2]+1),
    'nbins_e_mc': nbins[2], 'xlim_e_mc': xlim[2],
    'nbins_c_mc': nbins[2], 'xlim_c_mc': xlim[2],
    'bins_e_mc': np.linspace(*xlim[2], nbins[2]+1),
    'bins_c_mc': np.linspace(*xlim[2], nbins[2]+1),
    'xlim': (0, 60), 'ylim': (-0.005, 0.12),
    'xlabel': r'$k_{t,g}$', 'symbol': r'$k_{t,g}$',
    'ylabel': r'$1/N_{ev}\, dN/dk_{t,g}$ [1/GeV]',
    'ibu_smoothing': False,
    'ibu_itnum': 10,
})
print('Done.')

print('Performing iterative Bayesian unfolding is 1d.')
# calculate quantities to be stored in obs
for obkey,ob in obs.items():
              
    # perform iterative Bayesian unfolding in 1d
    ob['ibu_class'] = ibu_1d_HI(ob['evs'](synthetic, 'evs_e_mc'), ob['evs'](synthetic, 'evs_c_mc'), [ob['bins_e_mc'],ob['bins_c_mc']], val_trash=-1, val_fake=-1, \
                      evs_e=ob['evs'](nature, 'evs_e_mc'), bins_e=ob['bins_e'], out_probability=posterior_is_probability, \
                      ibu_itnum=ob['ibu_itnum'], \
                      smoothing=ob['ibu_smoothing'], smoothing_method='rolling', smoothing_param=8)
    #ob['ibu'] = ob['ibu_class'].ibu()  
    ob['ibu_err'] = ob['ibu_class'].ibu_err(prior_err_itnum=5, meas_err_itnum=5, flat_prior=False)
    
    # get the histograms of GEN, DATA, and TRUTH level observables
    ob['H_c'] = MChistogram(ob['evs'](nature, 'evs_c'), bins=ob['bins_c'], probability=posterior_is_probability)
    ob['H_e'] = MChistogram(ob['evs'](nature, 'evs_e'), bins=ob['bins_e'], density=posterior_is_probability)
    ob['H_c_mc'] = ob['ibu_class'].p_c_mc
    ob['H_e_mc'] = ob['ibu_class'].p_e_mc
    
    print('Done with', obkey)

# perform iterative Bayesian unfolding in multi-sim
print('Preforming iterative Bayesian unfolding in multi-dimensions.')
evs_e_mc  = [obs['JetPT']['evs'](synthetic, 'evs_e_mc'), obs['Thg']['evs'](synthetic, 'evs_e_mc'), obs['Ktg']['evs'](synthetic, 'evs_e_mc')]
evs_c_mc  = [obs['JetPT']['evs'](synthetic, 'evs_c_mc'), obs['Thg']['evs'](synthetic, 'evs_c_mc'), obs['Ktg']['evs'](synthetic, 'evs_c_mc')]
bins_e_mc = [obs['JetPT']['bins_e_mc'], obs['Thg']['bins_e_mc'], obs['Ktg']['bins_e_mc']]
bins_c_mc = [obs['JetPT']['bins_c_mc'], obs['Thg']['bins_c_mc'], obs['Ktg']['bins_c_mc']]
evs_e     = [obs['JetPT']['evs'](nature, 'evs_e'), obs['Thg']['evs'](nature, 'evs_e'), obs['Ktg']['evs'](nature, 'evs_e')]
bins_e    = [obs['JetPT']['bins_e'], obs['Thg']['bins_e'], obs['Ktg']['bins_e']]
bins_c    = [obs['JetPT']['bins_c'], obs['Thg']['bins_c'], obs['Ktg']['bins_c']]

ibudd_class = ibu_dd_HI(evs_e_mc, evs_c_mc, [bins_e_mc,bins_c_mc], -1, -1, \
                      evs_e=evs_e, bins_e=bins_e, out_probability=posterior_is_probability, \
                      ibu_itnum=5)
posdd_err = ibudd_class.ibu_err(prior_err_itnum=5, meas_err_itnum=5)
print('Done.')

# PLOTTING
print('Exporing the unfolded distributions to pdf.')
fontsize = 14

fig, [ax0, ax1] = plt.subplots(2, len(obs), gridspec_kw={'height_ratios': (3,1), 'hspace': 0}, figsize=(6*len(obs),7))

ax0[-1].text(63.5, 0.0145, r'Truth/Measured: Herwig7 + thermal bkg.', color='gray', fontsize=fontsize, rotation=-90)
ax0[-1].text(60.5, 0.006, r'Smeared/Generated: Pythia8 + thermal bkg.', color='gray', fontsize=fontsize, rotation=-90)

# Multi-dim bin volumes.
dbin_c = [np.diff(j) for j in posdd_err[-1]]
for i in range(len(dbin_c)-1):
    if i==0: dbin_c_outer = dbin_c[i]
    dbin_c_outer = np.expand_dims(dbin_c_outer, axis=-1) * dbin_c[i+1]

for i,(obkey,ob) in enumerate(obs.items()):
    # Loop through observables

    # plot the "data" histogram of the observable
    ax0[i].plot(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2, ob['H_c'][0], 'C0', label='Truth')
    ax0[i].fill_between(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2, ob['H_c'][0], color='C0', alpha=0.3)
    ax0[i].errorbar(ob['H_e'][-1][:-1]+np.diff(ob['H_e'][-1])/2, ob['H_e'][0], ob['H_e'][1], xerr=np.diff(ob['H_e'][-1])/2, fmt='o', c='k', label='Measured')
    # plot the IBU distribution
    ax0[i].errorbar((ob['ibu_err'][-1][:-1]+np.diff(ob['ibu_err'][-1])/2)[1:], ob['ibu_err'][0][1:], ob['ibu_err'][1][1:], xerr=np.diff(ob['ibu_err'][-1][1:])/2, fmt='o', c='C3', label='ibu 1d')
    # plot the IBUdd distribution
    axis = tuple([j for j in range(len(posdd_err[-1])) if j!=i])
    ax0[i].errorbar(posdd_err[-1][i][:-1]+np.diff(posdd_err[-1][i])/2,  (posdd_err[0] * dbin_c_outer).sum(axis=axis)/dbin_c[i],  (posdd_err[1] * dbin_c_outer).sum(axis=axis)/dbin_c[i], np.diff(posdd_err[-1][i])/2, fmt='o', c='C2', label=r'ibu 3d')
    # plot the ratios
    ax1[i].set_xlim(ob['xlim'])
    ax1[i].set_xlabel(ob['xlabel'], fontsize=fontsize)
    ax1[i].set_ylabel('Ratio to\nTruth', fontsize=fontsize)
    ax1[i].tick_params(axis='both',which='both', right=True, top=True, bottom=True, direction='in', labelsize=fontsize)
    ax1[i].set_ylim(0.5, 1.5)
    ax1[i].plot(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2, ob['H_c'][0]/ob['H_c'][0], ob['H_c'][0]/ob['H_c'][0], 'C0')
    ax1[i].fill_between(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2, 1+ob['H_c'][1]/ob['H_c'][0], 1-ob['H_c'][1]/ob['H_c'][0], color='C0', alpha=0.3)
    ax1[i].errorbar(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2, ob['ibu_err'][0][1:]/ob['H_c'][0], ob['ibu_err'][1][1:]/ob['H_c'][0], xerr=np.diff(ob['ibu_err'][-1][1:])/2, fmt='o', c='C3')
    ax1[i].errorbar(ob['H_c'][-1][:-1]+np.diff(ob['H_c'][-1])/2,  ((posdd_err[0] * dbin_c_outer).sum(axis=axis)/dbin_c[i])[1:]/ob['H_c'][0],  ((posdd_err[1] * dbin_c_outer).sum(axis=axis)/dbin_c[i])[1:]/ob['H_c'][0], np.diff(ob['ibu_err'][-1][1:])/2, fmt='o', c='C2', label=r'ibu 3d')
    
    ax0[i].set_xlim(ob['xlim'])
    ax0[i].set_ylim(ob['ylim'])
    ax0[i].set_ylabel(ob['ylabel'], fontsize=fontsize)
    ax0[i].tick_params(axis='both',which='both', right=True, top=True, bottom=True, labelbottom=False, direction='in', labelsize=fontsize)
    ax0[i].legend(fontsize=fontsize)
    
plt.tight_layout()
plt.savefig('test_ibu_HI.pdf')

print('Done, program is ended.')