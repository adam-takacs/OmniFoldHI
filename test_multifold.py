import numpy as np
import copy
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pylab as plt

from omnifold import omnifold

def plot_omn (obs, obs_multifold, weights=[], save_file=''):

    hist_style = {
        'gen': {'label':'gen', 'color':'royalblue'},
        'sim': {'label':'sim', 'color':'orange'},
        'tru': {'label':'tru', 'color':'green'},
        'dat': {'label':'dat', 'color':'k'},
        'omn': {'label':'omn', 'color':'crimson'}
    }
    for stkey, st in hist_style.items():
        st.update({'fmt':'-'}) 

    fig = plt.figure(figsize=(len(obs_multifold)*7,7))

    gs_main = fig.add_gridspec(1, len(obs_multifold))

    for i, key in enumerate(obs_multifold):

        gs = gs_main[i].subgridspec(2,1, height_ratios=(5,2), hspace=0)
        ax  = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)

        ob = obs[key]
        ax.errorbar(ob['midbins_raw'], ob['obs_dat_hist'], ob['obs_dat_hist_std'], ob['bin_raw_std'], **hist_style['dat'])
        ax.errorbar(ob['midbins_sig'], ob['obs_tru_hist'], ob['obs_tru_hist_std'], ob['bin_sig_std'], **hist_style['tru'])
        ax.errorbar(ob['midbins_sig'], ob['obs_gen_hist'], ob['obs_gen_hist_std'], ob['bin_sig_std'], **hist_style['gen'])
        ax.errorbar(ob['midbins_raw'], ob['obs_sim_hist'], ob['obs_sim_hist_std'], ob['bin_raw_std'], **hist_style['sim'])

        if len(weights)>0:

            # plot omnifold
            hist_sim = np.histogram(ob['obs_sim'], bins=ob['bins_raw'], weights = weights[-1, 0])[0]
            hist_gen = np.histogram(ob['obs_gen'], bins=ob['bins_sig'], weights = weights[-1, 1])[0]
            ax.errorbar(ob['midbins_sig'], hist_gen, **hist_style['omn'])
            #ax.errorbar(ob['midbins_raw'], hist_sim, **hist_style['omn'], ls='--')

            axr.errorbar(ob['midbins_sig'], (ob['obs_tru_hist']+.01)/(ob['obs_tru_hist']+.01), **hist_style['tru'])
            axr.errorbar(ob['midbins_sig'], (hist_gen+.01)/(ob['obs_tru_hist']+.01), **hist_style['omn'])

        ax.set_ylabel(ob['ylabel'])
        axr.set_xlabel(ob['xlabel'])
        axr.set_ylabel('ratio to truth')
        #axr.set_ylim(0,2)
        axr.set_ylim(.5,1.5)

        if i==0:
            ax.legend(frameon=False, ncol=2)

        for axis in [ax,axr]:
            axis.minorticks_on()
            axis.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    if save_file:
        plt.savefig(save_file)

def read_mcpair (datafile_sig, datafile_raw):
    
    dataset = {}

    # add signal events
    with open(datafile_sig,'r') as f:
        obs_labels = f.readline().strip('#').split()
        obs_vals = np.loadtxt(datafile_sig)
        print(obs_labels)

    for i,val in enumerate(obs_labels):
        dataset['sig_'+val] = obs_vals.T[i]

    # add raw events
    with open(datafile_raw,'r') as f:
        obs_labels = f.readline().strip('#').split()
        obs_vals = np.loadtxt(datafile_raw)

    for i,val in enumerate(obs_labels):
        dataset['raw_'+val] = obs_vals.T[i]

    return dataset

def complete_obs (obs, synthetic, nature):

    # Calculate quantities to be stored in obs
    for obkey,ob in obs.items():
        
        # calculate observable for GEN, SIM, DATA, and TRUE
        ob['obs_gen'], ob['obs_sim'] = ob['func'](synthetic, 'sig'), ob['func'](synthetic, 'raw')
        ob['obs_tru'], ob['obs_dat'] = ob['func'](nature, 'sig'), ob['func'](nature, 'raw')
        
        # setup bins
        ob['bins_sig']     = np.linspace(ob['xlim_sig'][0], ob['xlim_sig'][1], ob['nbins_sig']+1)
        ob['bins_raw']     = np.linspace(ob['xlim_raw'][0], ob['xlim_raw'][1], ob['nbins_raw']+1)
        ob['midbins_sig']  = (ob['bins_sig'][:-1] + ob['bins_sig'][1:])/2
        ob['midbins_raw']  = (ob['bins_raw'][:-1] + ob['bins_raw'][1:])/2
        ob['binwidth_sig'] = ob['bins_sig'][1] - ob['bins_sig'][0]  # assuming linear binning
        ob['binwidth_raw'] = ob['bins_raw'][1] - ob['bins_raw'][0]
        ob['bin_sig_std']  = ob['binwidth_sig']/2  # assuming linear binning
        ob['bin_raw_std']  = ob['binwidth_raw']/2

        # get the histograms
        ob['obs_gen_hist'] = np.histogram(ob['obs_gen'], bins=ob['bins_sig'] )[0]
        ob['obs_sim_hist'] = np.histogram(ob['obs_sim'], bins=ob['bins_raw'] )[0]
        ob['obs_tru_hist'] = np.histogram(ob['obs_tru'], bins=ob['bins_sig'] )[0]
        ob['obs_dat_hist'] = np.histogram(ob['obs_dat'], bins=ob['bins_raw'] )[0]

        # get the standard deviations
        ob['obs_gen_hist_std'] = np.sqrt(np.histogram(ob['obs_gen'], bins=ob['bins_sig'] )[0])
        ob['obs_sim_hist_std'] = np.sqrt(np.histogram(ob['obs_sim'], bins=ob['bins_raw'] )[0])
        ob['obs_tru_hist_std'] = np.sqrt(np.histogram(ob['obs_tru'], bins=ob['bins_sig'] )[0])
        ob['obs_dat_hist_std'] = np.sqrt(np.histogram(ob['obs_dat'], bins=ob['bins_raw'] )[0])
        
        print('Done with', obkey)

# Choose datasets to use as synthetic and nature
data_path = 'datasets/'
synthetic = read_mcpair(data_path+'Pythia_hardestJet_250GeV_Sig.out', data_path+'Pythia_hardestJet_250GeV_Dat.out')
nature    = read_mcpair(data_path+'Herwig_hardestJet_250GeV_Sig.out', data_path+'Herwig_hardestJet_250GeV_Dat.out')

# Figure path
fig_path = ''

# Create dictionary w/ observables and related info
observables = {}

# Observable pt_jet
observables.setdefault('pt_jet', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_pt_jet'],
    'nbins_sig': 10, 'nbins_raw': 10,
    'xlim_sig': (200,800), 'xlim_raw': (200,800),
    'xlabel': 'pt_jet',
    'ylabel': 'dN/dpt_jet'
})

# Observable thg
observables.setdefault('thg', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_thg'],
    'nbins_sig': 20, 'nbins_raw': 20,
    'xlim_sig': (0,.4), 'xlim_raw': (0,.4),
    'xlabel': 'thg',
    'ylabel': 'dN/dthg'
})

# Observable ktg
observables.setdefault('ktg', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_ktg'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,60), 'xlim_raw': (0,60),
    'xlabel': 'ktg',
    'ylabel': 'dN/dktg'
})

# Create observables dictionary
complete_obs(observables, synthetic, nature)

# Observables to unfold
obs_multifold = ['pt_jet', 'thg', 'ktg']

# training data
dummy_phi = -1
theta_gen = np.asarray([observables[obkey]['obs_gen'] for obkey in obs_multifold]).T
theta_sim = np.asarray([observables[obkey]['obs_sim'] for obkey in obs_multifold]).T
theta_dat = np.asarray([observables[obkey]['obs_dat'][observables[obkey]['obs_dat']!=dummy_phi] for obkey in obs_multifold]).T

# Multifodl iterative reweighing
weights = omnifold(theta_gen, theta_sim, theta_dat,
                   model=0, itnum=4, dummy_phi=dummy_phi)

plot_omn(observables, obs_multifold, weights, save_file=fig_path+'test.pdf')