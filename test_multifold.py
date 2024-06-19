import numpy as np
import copy
import os
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pylab as plt

from omnifold import omnifold_weights, omnifold_weights2

def plot_omn (obs, obs_multifold, weights, weights2=[], weights_syst=[], it=-1, save_file='', std_scale=1, only_uncertainty=False):

    '''
    Parameters
    ----------
    obs              : dictionary, all the observables' info
    obs_multifold    : list, keys of observables to plot
    weights          : array, omnifold output
                       array, array of several omnifold outputs for stat uncert. (if weights2 is not present)
    weights2         : array, omnifold2 output
    weights_syst     : array, array of several omnifold outputs for syst uncert.
    it               : int, iteration to show
    save_file        : string, filename
    std_scale        : float, scale factor applied to the uncertainty
    only_uncertainty : bool, if True, show only uncertainty

    Returns
    -------
    axes
    '''

    # Style options
    hist_style = {
        'gen':      {'label':'gen', 'lw':1,  'fmt':'-', 'color':'royalblue'},
        'sim':      {'label':'sim', 'lw':1,  'fmt':'-', 'color':'orange'},
        'tru':      {'label':'tru', 'lw':1,  'fmt':'-', 'color':'green'},
        'dat':      {'label':'dat', 'lw':1,  'fmt':'-', 'color':'k'},
        'omn':      {'label':'omn', 'lw':.5, 'fmt':'-', 'color':'crimson'},
        'omn_syst': {'label':'sys uncert.', 'color':'crimson', 'alpha':.3}
    }

    # Create figure and grid
    fig = plt.figure(figsize=(14,10/3*int(np.ceil(len(obs_multifold)/3))))
    gs_main = fig.add_gridspec(int(np.ceil(len(obs_multifold)/3)), 3)

    # Plot for each observable
    axes = []
    for i, key in enumerate(obs_multifold):

        # Add observable subpolot, append to axes
        gs = gs_main[i].subgridspec(2,1, height_ratios=(5,2), hspace=0)
        ax  = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        axes.append(ax)

        # Plot original histos
        ob = obs[key]
        ax.errorbar(ob['midbins_raw'], ob['obs_dat_hist'], ob['obs_dat_hist_std']*std_scale, ob['bin_raw_std'], **hist_style['dat'])
        ax.errorbar(ob['midbins_sig'], ob['obs_tru_hist'], ob['obs_tru_hist_std']*std_scale, ob['bin_sig_std'], **hist_style['tru'])
        ax.errorbar(ob['midbins_sig'], ob['obs_gen_hist'], ob['obs_gen_hist_std']*std_scale, ob['bin_sig_std'], **hist_style['gen'])
        ax.errorbar(ob['midbins_raw'], ob['obs_sim_hist'], ob['obs_sim_hist_std']*std_scale, ob['bin_raw_std'], **hist_style['sim'])

        # Plot raito of truth
        tru_tru     = (ob['obs_tru_hist']+.01)/(ob['obs_tru_hist']+.01)
        #tru_tru_std = (ob['obs_tru_hist_std'])/(ob['obs_tru_hist']+.01) * np.sqrt(2)
        tru_tru_std = (ob['obs_tru_hist_std'])/(ob['obs_tru_hist']+.01)
        axr.errorbar(ob['midbins_sig'], tru_tru, **hist_style['tru'])
        axr.fill_between(ob['midbins_sig'], tru_tru-tru_tru_std, tru_tru+tru_tru_std, color='green', alpha=.4)

        # Compute histo and stat uncertainty
        if len(weights2):
            hist_omn     = np.histogram(ob['obs_gen'], bins=ob['bins_sig'], weights = weights[it, 1])[0]
            hist_omn_std = np.sqrt( np.histogram(ob['obs_gen'], bins=ob['bins_sig'], weights = weights2[it, 1]**2)[0] )
        else:
            hist_omn_arr = np.array( [np.histogram(ob['obs_gen'], bins=ob['bins_sig'], weights=w[it,1])[0] for w in weights] )
            hist_omn     = np.mean(hist_omn_arr, axis=0)
            hist_omn_std = np.std(hist_omn_arr, axis=0)
        omn_tru      = tru_tru if only_uncertainty else (hist_omn+.01)/(ob['obs_tru_hist']+.01)
        #omn_tru_std  = np.sqrt( hist_omn_std**2 + (ob['obs_tru_hist_std']*(hist_omn+.01)/(ob['obs_tru_hist']+.01))**2 ) / (ob['obs_tru_hist']+.01)
        omn_tru_std  = (hist_omn_std+.01)/(ob['obs_tru_hist']+.01)

        # Compute and plot syst uncertaity
        if len(weights_syst):

            # Compute uncertaity
            hist_omn_syst_arr = np.array( [np.histogram(ob['obs_gen'], bins=ob['bins_sig'], weights=w[it,1])[0] for w in weights_syst] )
            hist_omn_syst_std = np.std(hist_omn_syst_arr, axis=0)
            #omn_syst_tru_std  = np.sqrt( hist_omn_syst_std**2 + (ob['obs_tru_hist_std']*(hist_omn_syst+.01)/(ob['obs_tru_hist']+.01))**2 ) / (ob['obs_tru_hist']+.01)
            omn_syst_tru_std  = (hist_omn_syst_std+.01)/(ob['obs_tru_hist']+.01)

            # Plot uncertaity as boxes
            ax.bar(ob['midbins_sig'], 2*hist_omn_syst_std, 2*ob['bin_sig_std'], hist_omn-hist_omn_syst_std, **hist_style['omn_syst'])
            axr.bar(ob['midbins_sig'], 2*omn_syst_tru_std, 2*ob['bin_sig_std'], omn_tru-omn_syst_tru_std, **hist_style['omn_syst'])
        
        # Plot histo and stat uncertaity
        ax .errorbar(ob['midbins_sig'], hist_omn, hist_omn_std*std_scale, ob['bin_sig_std'], **hist_style['omn'], capsize=2)
        axr.errorbar(ob['midbins_sig'], omn_tru, omn_tru_std, ob['bin_sig_std'], **hist_style['omn'], capsize=2)

        # Axis stuff
        ax.set_ylabel(ob['ylabel'])
        axr.set_xlabel(ob['xlabel'])
        axr.set_ylabel('ratio to truth')
        #axr.set_ylim(0,2)
        axr.set_ylim(.8,1.2)
        for axis in [ax,axr]:
            axis.minorticks_on()
            axis.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    # Axis stuff
    axes[0].legend(frameon=False, ncol=2)
    plt.tight_layout()

    # Save plot
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    return axes

def read_sigraw (datafile_sig, datafile_raw):

    def get_labels (filename):
        file = open(filename,'r')
        labels_line = file.readline().strip('#').split()
        file.close()
        return [ label.strip(',') for label in labels_line if '[' not in label ]

    dataset = {}

    # add signal events
    sig_labels = get_labels(datafile_sig)
    sig_values = np.loadtxt(datafile_sig)
    for i,label in enumerate(sig_labels):
        dataset['sig_'+label] = sig_values.T[i]
        
    # add raw events
    raw_labels = get_labels(datafile_raw)
    raw_values = np.loadtxt(datafile_raw)
    for i,label in enumerate(raw_labels):
        dataset['raw_'+label] = raw_values.T[i]

    return dataset

def associate_data (observables, synthetic, nature):

    # Calculate quantities to be stored in obs
    for label, obs in observables.items():
        
        # calculate obs for GEN, SIM, DATA, and TRUE
        obs['obs_gen'], obs['obs_sim'] = obs['func'](synthetic, 'sig'), obs['func'](synthetic, 'raw')
        obs['obs_tru'], obs['obs_dat'] = obs['func'](nature, 'sig'), obs['func'](nature, 'raw')
        
        # setup bins
        obs['bins_sig']     = np.linspace(obs['xlim_sig'][0], obs['xlim_sig'][1], obs['nbins_sig']+1)
        obs['bins_raw']     = np.linspace(obs['xlim_raw'][0], obs['xlim_raw'][1], obs['nbins_raw']+1)
        obs['midbins_sig']  = (obs['bins_sig'][:-1] + obs['bins_sig'][1:])/2
        obs['midbins_raw']  = (obs['bins_raw'][:-1] + obs['bins_raw'][1:])/2
        obs['binwidth_sig'] = obs['bins_sig'][1] - obs['bins_sig'][0]  # assuming linear binning
        obs['binwidth_raw'] = obs['bins_raw'][1] - obs['bins_raw'][0]
        obs['bin_sig_std']  = obs['binwidth_sig']/2  # assuming linear binning
        obs['bin_raw_std']  = obs['binwidth_raw']/2

        # get the histograms
        obs['obs_gen_hist'] = np.histogram(obs['obs_gen'], bins=obs['bins_sig'] )[0]
        obs['obs_sim_hist'] = np.histogram(obs['obs_sim'], bins=obs['bins_raw'] )[0]
        obs['obs_tru_hist'] = np.histogram(obs['obs_tru'], bins=obs['bins_sig'] )[0]
        obs['obs_dat_hist'] = np.histogram(obs['obs_dat'], bins=obs['bins_raw'] )[0]

        # get the standard deviations
        obs['obs_gen_hist_std'] = np.sqrt(np.histogram(obs['obs_gen'], bins=obs['bins_sig'] )[0])
        obs['obs_sim_hist_std'] = np.sqrt(np.histogram(obs['obs_sim'], bins=obs['bins_raw'] )[0])
        obs['obs_tru_hist_std'] = np.sqrt(np.histogram(obs['obs_tru'], bins=obs['bins_sig'] )[0])
        obs['obs_dat_hist_std'] = np.sqrt(np.histogram(obs['obs_dat'], bins=obs['bins_raw'] )[0])
        
        print('Done with', label)

    return

# Choose datasets to use as synthetic and nature
data_path = 'datasets/'
synthetic = read_sigraw(data_path+'Pythia_hardestJet_250GeV_Sig.out', data_path+'Pythia_hardestJet_250GeV_Dat.out')
nature    = read_sigraw(data_path+'Herwig_hardestJet_250GeV_Sig.out', data_path+'Herwig_hardestJet_250GeV_Dat.out')

# Create dictionary w/ observables and related info
observables = {}

observables.setdefault('pt_jet', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_pt_jet'],
    'nbins_sig': 10, 'nbins_raw': 10,
    'xlim_sig': (200,800), 'xlim_raw': (200,800),
    'xlabel': 'pt_jet',
    'ylabel': 'dN/dpt_jet'
})

observables.setdefault('thg', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_thg'],
    'nbins_sig': 20, 'nbins_raw': 20,
    'xlim_sig': (0,.4), 'xlim_raw': (0,.4),
    'xlabel': 'thg',
    'ylabel': 'dN/dthg'
})

observables.setdefault('ktg', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_ktg'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,60), 'xlim_raw': (0,60),
    'xlabel': 'ktg',
    'ylabel': 'dN/dktg'
})

observables.setdefault('m_jet', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_m_jet'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,100), 'xlim_raw': (0,100),
    'xlabel': 'm_jet',
    'ylabel': 'dN/dm_jet'
})

observables.setdefault('n_jet', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_n_jet'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,100), 'xlim_raw': (0,100),
    'xlabel': 'n_jet',
    'ylabel': 'dN/dn_jet'
})

observables.setdefault('m_sd', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_m_sd'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,100), 'xlim_raw': (0,100),
    'xlabel': 'm_sd',
    'ylabel': 'dN/dm_sd'
})

observables.setdefault('n_sd', {}).update({
    'func': lambda dset, ptype: dset[ptype+'_n_sd'],
    'nbins_sig': 15, 'nbins_raw': 15,
    'xlim_sig': (0,100), 'xlim_raw': (0,100),
    'xlabel': 'n_sd',
    'ylabel': 'dN/dn_sd'
})

# Create observables dictionary
associate_data(observables, synthetic, nature)

# Choose observables to unfold
obs_multifold = ['pt_jet', 'thg', 'ktg','m_jet','n_jet','m_sd','n_sd']

# Training data
dummy_phi = -1
events_gen = np.asarray([observables[label]['obs_gen'] for label in obs_multifold]).T
events_sim = np.asarray([observables[label]['obs_sim'] for label in obs_multifold]).T
events_dat = np.asarray([observables[label]['obs_dat'][observables[label]['obs_dat']!=dummy_phi] for label in obs_multifold]).T

# Estimation of systemathic uncertaity
if os.path.exists('weights_syst.npy'):
    weights_syst = np.load('weights_syst.npy')
else:
    print('Estimating systematic uncertaity.')
    weights_syst = []
    np.random.seed(8)
    for i in range(5):
        print(i+1)
        weights = omnifold_weights(events_gen, events_sim, events_dat,
                                   weights_syn=[], weights_nat=[],
                                   model=0, iterations=4, dummy_phi=dummy_phi, rand_seed=np.random.randint(100))
        weights_syst.append(weights)
    weights_syst = np.array(weights_syst)
    np.save('weights_syst.npy', weights_syst)

# Unfolding and estimation of statistical uncertaity due to statistical uncertaity in data
# 1. variation of dat events
if os.path.exists('weights_stat1.npy'):
    weights_stat1 = np.load('weights_stat1.npy')
else:
    weights_stat1 = []
    for i in range(5):
        print(i+1)
        np.random.seed(i)
        weights = omnifold_weights(events_gen, events_sim, events_dat,
                                   weights_syn=[], weights_nat=np.random.poisson(1, size=len(events_dat)),
                                   iterations=4, model=0, dummy_phi=dummy_phi, rand_seed=1)
        weights_stat1.append(weights)
    weights_stat1 = np.array(weights_stat1)
    np.save('weights_stat1.npy', weights_stat1)

###
# 2. learn square of weights
if os.path.exists('weights_stat2.npy'):
    weights_stat2 = np.load('weights_stat2.npy')
else:
    weights_stat2  = omnifold_weights(events_gen, events_sim, events_dat,
                                      weights_syn=[], weights_nat=[],
                                      iterations=4, model=0, dummy_phi=dummy_phi, rand_seed=1)
    np.save('weights_stat2.npy', weights_stat2)

if os.path.exists('weights2_stat2.npy'):
    weights2_stat2 = np.load('weights2_stat2.npy')
else:
    weights2_stat2 = omnifold_weights2(events_gen, events_sim, events_dat,
                                       weights_syn=[], weights_nat=[],
                                       iterations=4, model=0, dummy_phi=dummy_phi, rand_seed=1)                                
    np.save('weights2_stat2.npy', weights2_stat2)

# Visualize
plot_omn(observables, obs_multifold, weights=weights_stat1, weights2=[], weights_syst=weights_syst,
         std_scale=1, save_file='multifolded_obs_stat1.pdf', only_uncertainty=False)

plot_omn(observables, obs_multifold, weights=weights_stat2, weights2=weights2_stat2, weights_syst=weights_syst,
         std_scale=1, save_file='multifolded_obs_stat2.pdf', only_uncertainty=False)