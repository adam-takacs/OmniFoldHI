import numpy as np

def MChistogram(data, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of 1d data: dN/dbin. 
        It also returns with uncertainty assuming Normal distribution
        usual for event generator plots.
       
        Parameters
        ----------
        data : array, input data.
        bins : array, bin edges.
        w    : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binsize.
              
        Returns
        -------
        [y, yerr, bins]
        
        Use plt.errorbar(bins[:-1]+np.diff(bins)/2,y,yerr) for plotting.
    '''
    if w is None: w = np.ones(len(data))
    dx    = np.diff(bins)
    y     = np.histogram(data, bins=bins, weights=w,    density=False)[0]
    y_err = np.histogram(data, bins=bins, weights=w**2, density=False)[0]

    if density: 
        return [y/sum(y)/dx, np.sqrt(y_err)/sum(y)/dx, bins]
    if probability:
        return [y/sum(w)/dx, np.sqrt(y_err)/sum(w)/dx, bins]
    if dividebin:
        return [y/dx,        np.sqrt(y_err)/dx,        bins]
    return [y, np.sqrt(y_err), bins] 

def MChistogram2d(datax, datay, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of 2d data: dN/dxdy. 
        It also returns with uncertainty assuming the normal distributed 
        bincounts (typical for MC generated smaples).
       
        Parameters
        ----------
        datax : array, input data1.
        datay : array, input data2.
        bins  : (array, array), containing x and y bin edges.
        w     : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binsize.
              
        Returns
        -------
        [z, zerr, xbins, ybins]
        
        Use pcolormesh(X,Y,Z) for plotting.
    '''
    if w is None: w = np.ones(len(datax))
    z, x, y        = np.histogram2d(datax, datay, bins=bins, weights=w,    density=False)
    zerr, dum, dum = np.histogram2d(datax, datay, bins=bins, weights=w**2, density=False)
    
    if density:
        return [z/np.outer(np.diff(x),np.diff(y))/sum(z), np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y))/sum(z), x, y]
    if probability:
        return [z/np.outer(np.diff(x),np.diff(y))/sum(w), np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y))/sum(w), x, y]
    if dividebin:
        return [z/np.outer(np.diff(x),np.diff(y)),        np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y)),        x, y]
    return [z, np.sqrt(zerr), x, y]

def MChistogramdd(data, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of N-dim data: dN / dx1*dx2*...*dxn. 
        It also returns with uncertainty assuming the normal distributed bincounts 
        typical for MC generated smaples.
       
        Parameters
        ----------
        data : [data1, data2, ..., dataN], list of array input data.
        bins : [bins1, bins2, ..., binsN], containing bin edges.
        w    : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binwidth.
              
        Returns
        -------
        [z, zerr, bins]
    '''
    if len(data) != len(bins): 
        return print('Error data and bins are incompatible', len(data), len(bins))
    
    if w is None: w = np.ones(len(data[0]))
    z    = np.histogramdd(data, bins=bins, weights=w,    density=False)[0]
    zerr = np.histogramdd(data, bins=bins, weights=w**2, density=False)[0]
    
    z_sum = 1
    if density:     z_sum = z.sum()
    if probability: z_sum = w.sum()
    
    if dividebin:
        bindiff = [np.diff(i) for i in bins]
        for i in range(len(data)):
            shape    = np.ones(len(bins), int)
            shape[i] = len(bins[i]) - 1
            z    = z / bindiff[i].reshape(shape)
            zerr = np.sqrt(zerr) / bindiff[i].reshape(shape)
    z    /= z_sum
    zerr /= z_sum
    
    return [z, zerr, bins]

class ibu_1d_HI:
    def __init__(self, evs_e_mc, evs_c_mc, bins_mc, val_trash=-1, val_fake=-1, \
                 evs_e=None, bins_e=None, out_probability=False, \
                 ibu_itnum=20, smoothing=True, smoothing_method='rolling', smoothing_param=None):
        ''' Iterative Bayesian Unfolding in 1d.
            Parameters
            ----------
            evs_e_mc : list, list of mc effects.
            evs_c_mc : list, list of mc causes.
            bins_mc  : (array, array), list of mc bins.
            val_trash : double, value of fake in causes.
            val_fake  : double, value of trash in effects.
            evs_e  : list, list of effects.
            bins_e : array, bins of effects.
            out_probability : bool, posterior is probability or bincount.
            ibu_itnum : int, iteration number.
            smoothing : bool, smoothing the posterior between iterations.
            smoothing_method : string, smoothing method.
            
            Functions
            ---------
            lam(evs_e, evs_c, bins), response matrix P(E| C).
            ibu_smoothing(posterior), smoothing routine.
            ibu(measure, prior), ibu algorithm.
            ibu_err(prior_err, measure_err, prior_err_itnum, measure_err_itnum), ibu algorithm with uncertainty.
        '''
        # Monte Carlo inputs
        self.evs_e_mc = evs_e_mc
        self.evs_c_mc = evs_c_mc
        self.bins_mc  = bins_mc
        self.val_trash = val_trash
        self.val_fake  = val_fake
        if (self.val_trash > self.bins_mc[0][0] or self.val_fake  > self.bins_mc[1][0]): 
            return print("Error : wrong trash or fake bin!")
        self.bins_mc_new = [np.append(self.val_trash, self.bins_mc[0]), 
                            np.append(self.val_fake, self.bins_mc[1])]
        self.p_e_mc  = MChistogram(evs_e_mc, bins=self.bins_mc_new[0], probability=True) # MC effect P(Ej | MC) with Trash.
        self.p_c_mc  = MChistogram(evs_c_mc, bins=self.bins_mc_new[1], probability=True) # MC cause P(Ci | MC) with Fake.
        self.p_ec_mc = self.lam(evs_e_mc, evs_c_mc, self.bins_mc) # MC response P(Ej | Ci, MC) with Trash and Fake bins.
        
        # Measurement input
        self.evs_e  = evs_e
        self.bins_e = bins_e
        self.out_probability = out_probability
        if self.evs_e != None:
            self.x_e = MChistogram(self.evs_e, bins=self.bins_e)
            
        # IBU parameters
        self.ibu_itnum = ibu_itnum
        self.smoothing = smoothing
        self.smoothing_method = smoothing_method
        
    def lam(self, evs_e, evs_c, bins):
        ''' Response matrix, normalized to being probability including fakes and trash.
            Algorithm
            ---------
            lam[j,i] = P(Ej | Ci) = P(Ci, Ej) / P(Ci).
            
            Parameters
            ----------
            evs_e : list, causes (fake is included).
            evc_c : list, effects (trash is included).
            bins  : [bins_e, bins_c].
            val_trash : value of the trash (should be the smallest bin).
            val_fake  : value of the fake (should be the smallest bin).

            Returns
            -------
            lam : [p_ec, p_ec_err, bins_e_new, bins_c_new].

            Plotting
            --------
            x-axis: cause, y-axis: effect.
            X, Y = np.meshgrid(lam[3], lam[2])
            im   = plt.pcolormesh(X, Y, lam[0])
        '''
        # Introduce trash and fake bins as new 0th bins.
        bins_new = [np.append(self.val_trash, bins[0]), np.append(self.val_fake, bins[1])]
        # Constructing lam using the joint probability.
        lam = MChistogram2d(evs_e, evs_c, bins=bins_new, probability=True)
        H_c = MChistogram(evs_c, bins=bins_new[1], probability=True)
        lam[0] /= H_c[0][None,:] + 1e-50
        return lam
    
    def ibu_smoothing(self, posterior, smoothing_method=None, smoothing_param=3):
        ''' Posterior smoothing. Different options are available and the 
            method should reflect the unfolding problem. 
            
            Parameters
            ----------
            posterior : [pos_val, bins]..
            method    : smoothing method different options,
                'rolling' : rolling average,
                'polyfit' : polynomial fit,
                'gaussfit' : Gaussian fit,
                'powerfit' : power-law fit.
                
            Returns
            -------
            pos_smooth : [hist, bins]
        '''
        pos, bins  = posterior
        mid_bins   = bins[:-1]+np.diff(bins)/2
        pos_smooth = pos
        if smoothing_method == None: smoothing_method=self.smoothing_method
        if smoothing_param == None: smoothing_param=self.smoothing_param
        if smoothing_method == 'rolling':
            # Rolling average with a given box size.
            # Skip 0th bin as it is the Trash.
            # Rolling average works well if the histogram has many bins
            # and the historam is smooth (no peaks)!
            box_size = min(max(3, smoothing_param), len(pos[1:]))
            box = np.ones(box_size)/box_size
            pos_smooth[1:] = np.convolve(pos[1:], box, mode='same')
        if smoothing_method == 'polyfit':
            # Higher order polynomial fit. 
            # Skip 0th bin as it is the Trash.
            poly_coeff = poly.polyfit(mid_bins[1:], pos[1:], deg=max(smoothing_param, int(len(pos[1:])/2)))
            pos_smooth[1:] = poly.polyval(mid_bins[1:], poly_coeff)
        if smoothing_method == 'gaussfit':
            # Gaussian fit.
            # Skip 0th bin as it is the Trash.
            def gauss(x, a, x0, sig): 
                return a*np.exp(-(x-x0)**2/(2*sig**2))
            popt = curve_fit(gauss, mid_bins[1:], pos[1:], maxfev=2000)[0]
            pos_smooth[1:] = np.array([gauss(i, *popt) for i in mid_bins[1:]])
        if smoothing_method == 'powerfit':
            # Power law fit.
            # Skip 0th bin as it is the Trash.
            def power(x, x0, a, b): 
                return (x/x0)**(a+b*x)
            popt = curve_fit(power, mid_bins[1:], pos[1:])[0]
            pos_smooth[1:] = np.array([power(i, *popt) for i in mid_bins[1:]])
            
        return [pos_smooth, bins]
    
    def ibu(self, measure=None, prior=None):
        ''' Iterative Bayesian Unfolding in 1D (D'Adostini 95').
            It includes additional notion of Trash and Fakes.
            Algorithm
            ---------
            x(Ci) = sum_j P(Ci | Ej) * x(Ej), 
            where P(Ci | Ej) = P(Ej | Ci) * P(Ci) / P(Ej).
            P(Ci) is the prior, and P(Ej) = sum_i P(Ej | Ci) * P(Ci).
            If x(Ej) doesn't include the trash bin, efficiency is considered.
            
            Parameters
            ----------
            meas  : x(Ej) = [hist, hist_err, bins] measured histogram. Contain Trash bin or not.
            prior : P(Ci) = [hist, hist_err, bins] prior histogram.
            itnum : iteration number.

            Returns
            -------
            pos : x(Ci) = [hist, bins] posterior distribution.

            Plotting
            --------
            plt.plot(pos[-1][:-1]+np.diff(pos[-1])/2, pos[0])
        '''
        
        # Prior: P(Ci).
        # It can be arbitrary and sources uncertainty. More the 
        # iteration, less the dependence on the prior.
        if prior != None: 
            pri, pri_err, bins_c = prior
        else:
            # Method 1. assuming the MC cause as a prior.
            pri, pri_err, bins_c = self.p_c_mc
            # Method 2. assuming flat cause as a prior.
            #pri = np.ones(len(bins_c)-1)/(len(bins_c)-1)
            
        # Measured histogram: x(Ej).
        # It is given and sources uncertainty.
        if measure != None:
            meas, meas_err, bins_e = measure
        else:
            meas, meas_err, bins_e = self.x_e
        
        pos = [pri]
        # Iteration loop of the unfolding.
        for it in range(self.ibu_itnum):
            # Inverted MC response with the iterated prior: 
            # th_ij = P(Ci | Ej) = P(Ej | Ci) * P(Ci) / P(Ej),
            # where P(Ej) = sum_i=0 P(Ej | Ci) * P(Ci).
            th  = self.p_ec_mc[0] * pos[-1][None,:]
            # When summing Ci, bin width is considered.
            th /= (th * np.diff(bins_c)[None,:]).sum(axis=1)[:,None] + 1e-50
            th  = th.T
                        
            if th.shape[1] != meas.shape[0]: 
                # Efficiency is handled when there is no Trash bin in the measurement.
                # x(Ci) = sum_j=1 P(Ci | Ej) * x(Ej) * dEj + P(Ci | E0) * x(E0) * dE0
                #       = sum_j=1 P(Ci | Ej) * x(Ej) * dEj + P(E0 | Ci) * x(Ci) * dE0
                #       = sum_j=1 P(Ci | Ej) * x(Ej) / (1 - P(E0 | Ci) * dE0)
                # We include the efficiency factor in P(Ci | Ej).
                eps = 1. - self.p_ec_mc[0][0,:]*np.diff(self.p_e_mc[-1])[0] + 1e-50
                th /= eps[:,None]
                # Trimming the Trash bin of th_ij.
                th = th[:,1:]
                
            # Posterior.
            # x(Ci) = sum_j P(Ci | Ej) * x(Ej)
            # When summing Ej, bin width is considered.
            xC = (th * meas[None,:] * np.diff(bins_e)[None,:]).sum(axis=1)
            
            if self.smoothing and it != self.ibu_itnum-1: 
                # Smoothing the posterior distribution before reiterating.
                # The algorithm is sensitive to smoothing and therefore one 
                # should choose the option that works best for the given 
                # unfolding problem.
                xC = self.ibu_smoothing([xC, bins_c])[0]
            
            if self.out_probability: 
                # Normalize the posterior to probability if needed.
                # Note, that sum_i xC = 1 including the Fake!
                xC /= sum(xC * np.diff(bins_c)) + 1e-50
                
            pos.append(xC)
            ## TODO: Early stopping is when the posterior barely changes.
            #if abs(pos[-1]-pos[-2]).sum()/(pos[-1]-pos[-2]).std()<1e-5: break
        
        return [pos[-1], bins_c]
    
    def ibu_err(self, prior_err_itnum=3, meas_err_itnum=3, flat_prior=False):
        ''' Estimating uncertainties for ibu.
            
            Algorithm
            ---------
            P(Ci | MC) each event assumed to have Poisson distribution.
            
            Parameters
            ----------
            prior_err_itnum : number of iteration for prior statistical uncertainty.
            meas_err_itnum  : number of iteration for measure statistical uncertainty.
            flat_prior : bool, model uncertainty of the prior.

            Returns
            -------
            pos : x(Ci) = [hist, hist_err, bins] posterior distribution.

            Plotting
            --------
            plt.errorbar(pos[-1][:-1]+np.diff(pos[-1])/2, pos[0], yerr=pos[1], xerr=np.diff(pos[-1])/2)
        '''

        pos_err = [self.ibu()[0]]
        
        for i in range(prior_err_itnum):
            # Prior statistical uncertainty
            # Sampling several equally possible prior probability.
            ## TODO: the sampling through the histogram itself for speeding up the algorithm.
            wp  = np.random.poisson(1, size=len(self.evs_c_mc))
            pri = MChistogram(self.evs_c_mc, bins=self.p_c_mc[-1], w=wp, probability=True)
            for j in range(meas_err_itnum):
                # Measurememnt uncertainty.
                # Sampling several equally possible measurements.
                wm   = np.random.poisson(1, size=len(self.evs_e))
                meas = MChistogram(self.evs_e, bins=self.bins_e, w=wm)
                pos  = self.ibu(measure=meas, prior=pri)
                pos_err.append(pos[0])
                if flat_prior: 
                    # Estimating model uncertainty of the prior by changing it ot be flat.
                    pri[0] = np.ones(len(self.p_c_mc[-1])-1)/(len(self.p_c_mc[-1])-1)
                    pos = self.ibu(measure=meas, prior=pri)
                    pos_err.append(pos[0])     

        # Calculating the mean and the standard deviation.
        pos_mean = np.mean(np.asarray(pos_err), axis=0)
        pos_var  = np.std(np.asarray(pos_err), axis=0)
        
        return [pos_mean, pos_var, self.p_c_mc[-1]]
    
class ibu_dd_HI:
    def __init__(self, evs_e_mc, evs_c_mc, bins_mc, val_trash=-1, val_fake=-1, \
                 evs_e=None, bins_e=None, out_probability=False, \
                 ibu_itnum=5):
        ''' Iterative Bayesian Unfolding in multi dimensions (D'Agostini 95)
            Parameters
            ----------
            evs_e_mc  = [arr_evs_e1, arr_evs_e2, ...] : list of mc effects.
            evs_c_mc  = [arr_evs_c1, arr_evs_c2, ...] : list of mc causes.
            bins_mc   = [bins_e_mc, bins_e_mc]  : list of mc bins.
            val_trash = -1 : value of fake in causes .
            val_fake  = -1 : value of trash in effects.
            evs_e     = [arr_evs_e1, arr_evs_e2, ...] : list of effects.
            bins_e    = [bins_e1, bins_e2, ...] : list of effect bins.
            out_probability = False : posterior is bincount or probability.
            ibu_itnum = 5 : number of ibu iterations.
            
            Functions
            ---------
            lam(evs_e, evs_c, bins), response matrix P(E| C).
            ibu(measure, prior), ibu algorighm.
            ibu_err(prior_err, measure_err, prior_err_itnum, measure_err_itnum), ibu aldorithm with uncertainty.
        '''
        # Monte Carlo inputs.
        self.evs_e_mc  = evs_e_mc
        self.evs_c_mc  = evs_c_mc
        self.bins_mc   = bins_mc
        
        # Trash and Fake bins will be placed below the user bins. 
        # Check if the values are below the bin ranges.
        if ( any([val_trash > min(j) for j in self.bins_mc[0]]) or 
             any([val_fake  > min(i) for i in self.bins_mc[1]]) ): 
            return print('Error in Trash or Fake bin value.')
        # Append the MC bins with the Trash and Fake bins.
        self.bins_mc_new = [[np.append(val_trash, j) for j in self.bins_mc[0]],
                            [np.append(val_fake,  i) for i in self.bins_mc[1]]]
        
        self.p_e_mc  = MChistogramdd(self.evs_e_mc, bins=self.bins_mc_new[0], probability=True) # MC effect P(Ej | MC) with Trash.
        self.p_c_mc  = MChistogramdd(self.evs_c_mc, bins=self.bins_mc_new[1], probability=True) # MC cause P(Ci | MC) with Fake.
        self.p_ec_mc = self.lam([self.evs_e_mc, self.evs_c_mc], self.bins_mc_new) # MC response P(Ej | Ci, MC) with Trash and Fake.
                
        # Measurement input
        self.evs_e  = evs_e
        self.bins_e = bins_e
        self.out_probability = out_probability
        if self.evs_e != None:
            self.x_e = MChistogramdd(self.evs_e, bins=self.bins_e)
        
        # IBU parameters
        self.ibu_itnum = ibu_itnum
                
    def lam(self, evs, bins):
        ''' Response matrix, normalized to be probability including Fakes and Trash 
            in the zeroth bins: j = [0, 1, ..., nE], i = [0, 1, ..., nC]
            Algorithm
            ---------
            lam[j1, j2, ..., i1, i2, ...] = P(Ej1, Ej2, ...| Ci1, Ci2, ...) 
                                          = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
            
            Parameters
            ----------
            evs_e = [arr_e1, arr_e2, ...] : list of array of causes.
            evs_c = [arr_c1, arr_c2, ...] : list of array of effects.
            bins  = [bins_e, bins_c] : list of array of bins including Trash and Fake.

            Returns
            -------
            lam = [p_ec, p_ec_err, bins]
        '''
        evs_e,  evs_c  = evs
        bins_e, bins_c = bins
        
        # Joint probability P(Ej1, Ej2, ..., Ci1, Ci2, ...) and its stat. unc.
        lam, err = MChistogramdd(evs_e+evs_c, bins=bins_e+bins_c, probability=True)[0:2]
        
        # Conditioned probability: 
        # P(Ej1, Ej2, ...| Ci1, Ci2, ...) = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
        H_c = MChistogramdd(evs_c, bins=bins[1], probability=True)[0]
        # Expand dimensions of P(Ci1, Ci2, ...) --> P(None, None, ..., Ci1, Ci2, ...) 
        # to evaluate the ratio.
        exp_shape = [None for j in range(len(bins_e))]+[slice(None) for i in range(len(bins_c))]
        lam /= H_c[tuple(exp_shape)] + 1e-50
        err /= H_c[tuple(exp_shape)] + 1e-50
        
        return [lam, err, bins]
    
    def eps(self):
        ''' Calculating the efficiency factor. The measurement doesn't know about Trash, 
            and measured histograms are not probabilities but self normalized densities. 
            Efficiency corrects for this. Here is a derivation of what do we calculate 
            as in higher dimension it is quite complicated.
            Trash and Fake are in j=0 and i=0.
            
            xC[i1,i2,...] 
            = sum_{j1,j2,...=0}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
            = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
              + th[i1,...,0,0,...] * xE[0,0,...] * dxE[0,0,...]
              + Permut sum_{jk=1}^nE      th[i1,..., 0,...,jk,...,0]           * xE[0,...,jk,...,0]          * dxE[0,...,jk,...,0]
              + Permut sum_{jk1,jk2=1}^nE th[i1,..., 0,...,jk1,...,jk2,...,0]  * xE[0,...,jk1,...,jk2,...,0] * dxE[0,...,jk1,...,jk2,...,0]
              + ...
              + Permur sum_{j1,...,NOT jk,...=1}^nE th[i1,...,j1,...,jk=0,...] * xE[j1,...,jk=0,...]         * dxE[j1,...,jk=0,...]
            
            Using Bayes theorem, invert th-->lam for terms containing Trash:
            = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
              + ( lam[0,..., i1,...] * dxE[0,...,jk,...,0] 
                + Permut sum_{jk=1}^nE                lam[0,...jk,...,0, i1,...]   * dxE[0,...,jk,...,0]
                + ...
                + Permut sum_{j1,...,NOT jk,...=1}^nE lam[j1,...,jk=0,..., i1,...] * dxE[j1,...,jk=0,...] 
                ) * xC[i1,i2,...]
            = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...] / (1 - eps[i1,i2,...])
            
            This function returns with 
            eps[i1,i2,...]
            = ( lam[0,..., i1,...] * dxE[j1,j2,...]
              + Permut sum_{jk=1}^nE                lam[0,...jk,...,0, i1,...]   * dxE[0,...,jk,...,0]
              + ...
              + Permut sum_{j1,...,NOT jk,...=1}^nE lam[j1,...,jk=0,..., i1,...] * dxE[j1,...,jk=0,...]  
              ).
                            
            We extend eps[i1,i2,..., j1,j2,...] making it easier to take the ratio.
        '''
        # Binwidth
        dbin_e = [np.diff(i) for i in self.p_ec_mc[-1][0]]
        dbin_c = [np.diff(i) for i in self.p_ec_mc[-1][1]]
        for i in range(len(dbin_e)):
            # Create a tuple for the binwidth in a shape of [j1,j2,...].
            if i==0: dbin_e_outer = dbin_e[i]
            else: dbin_e_outer = np.expand_dims(dbin_e_outer, axis=-1) * dbin_e[i]
    
        # The efficiency definition can be viewed as a sum of lam elements where 
        # at least one of the j index is 0 (Trash value).
        # All indices which are 0 at least once.
        mask = (np.indices(dbin_e_outer.shape)==0).any(axis=0)
        res = (np.expand_dims(dbin_e_outer[mask], axis=tuple(-np.arange(1,len(dbin_c)+1))) * self.p_ec_mc[0][mask]).sum(axis=0)
        # Extending the dimensions eps[i1,i2] --> eps[i1,i2,...,j1,j2,...].
        res = np.expand_dims(res, axis=tuple(-np.arange(1,len(dbin_e)+1)))
        
        return res
    
    def ibu(self, measure=None, prior=None):
        ''' Iterative Bayesian Unfolding in multi dimensions (D'Adostini 95').
            It includes additional notion of Trash and Fakes.
            Algorithm
            ---------
            x(Ci1, Ci2, ...) 
            = sum_{j1, j2, ...=0}^nE P(Ci1, Ci2, ... | Ej1, Ej2, ...) * x(Ej1, Ej2, ...), 

            where ,

            P(Ci1, Ci2, ... | Ej1, Ej2, ..) 
            = P(Ej1, Ej2, ... | Ci1, Ci2, ...) * P(Ci1, Ci2, ...) / P(Ej1, Ej2, ...).

            If x(Ej) doesn't include the Trash bin, the efficiency is considered.
            
            Parameters
            ----------
            meas  : x(Ej) = [hist, hist_err, bins] measured histogram. Contain trash bin or not.
            prior : P(Ci) = [hist, hist_err, bins] prior histogram.

            Returns
            -------
            pos : x(Ci) = [hist, bins] posterior distribution.

            Plotting
            --------
            plt.plot(bins[:-1]+np.diff(bins)/2, hist)
        '''
        
        # Prior: P(Ci1, Ci2, ...).
        # It is arbitrary and sources uncertainty. More the 
        # iteration, less the dependence on the prior.
        if prior != None: 
            # User defined prior.
            pri, pri_err, bins_c_mc = prior
        else:
            # Method 1. assuming the MC cause as a prior.
            pri, pri_err, bins_c_mc = self.p_c_mc
            ## TODO: Method 2. assuming flat cause as a prior.
            ##pri = np.ones(pri.shape)
        shape_c_mc = np.array(pri.shape)
        shape_e_mc = np.array(self.p_e_mc[0].shape)
            
        # Measured histogram: x(Ej1, Ej2, ...).
        # It is given and sources statistical uncertainty.
        if measure != None:
            meas, meas_err, bins_e = measure
        else:
            meas, meas_err, bins_e = self.x_e
        shape_e = np.array(meas.shape)
            
        # Bin widths are needed for summation in various shapes.            
        dbin_c_mc = [np.diff(i) for i in bins_c_mc]
        for i in range(len(dbin_c_mc)):
            if i==0: dbin_c_outer = dbin_c_mc[i]
            else: dbin_c_outer = np.expand_dims(dbin_c_outer, axis=-1) * dbin_c_mc[i]
        dbin_e = [np.diff(i) for i in bins_e]
        for i in range(len(dbin_e)):
            if i==0: dbin_e_outer = dbin_e[i]
            else: dbin_e_outer = np.expand_dims(dbin_e_outer, axis=-1) * dbin_e[i]
        
        # The posterior and response matrix are flattened. 
        pos = [pri.flatten()]
        lam = self.p_ec_mc[0].reshape(np.prod(shape_e_mc), np.prod(shape_c_mc))
        for it in range(self.ibu_itnum):
            # Iteration loop of the unfolding.
            # Inverted MC response with the iterated prior: 
            # th[i1, i2, ..., j1, j2, ...] = P(Ci1, Ci2, ...| Ej1, Ej2, ...) 
            #                              = P(Ej1, Ej2, ...| Ci1, Ci2, ...) * P(Ci1, Ci2, ...) / P(Ej1, Ej2, ...),
            # where P(Ej1, Ej2, ...) = sum_{i1,i2, ...=0}^nC P(Ej1, Ej2, ...| Ci1, Ci2, ...) * P(Ci1, Ci2, ...).
            # Here we flatten the arrays so: th[i12..., j12...].
            th  = lam * pos[-1][None,:]
            # When summing Ci, bin width is considered.
            th /= (th * dbin_c_outer.flatten()[None,:]).sum(axis=1)[:,None] + 1e-50
            th  = th.T
                   
            if th.shape[1] != meas.flatten().shape[0]: 
                # Efficiency is handled if there is no Trash bin in the measurement.
                # x(Ci) = sum_j P(Ci | Ej) * x(Ej) + P(Ci | T) * x(T)
                #       = sum_j P(Ci | Ej) * x(Ej) + P(T | Ci) * x(Ci)
                #       = sum_j P(Ci | Ej) * x(Ej) / (1 - P(T | Ci))
                # We include the efficiency factor in th[i1,i2,...,j1,j2,...].
                th  = th.reshape(np.append(shape_c_mc, shape_e_mc))
                eps = self.eps()
                th  = th / (1. - self.eps() + 1e-50)
                # Trimming the Trash bin of th_ij = th[i1=:, i2=:, ..., j1=1:, j2=1:, ...].
                new_shape = [slice(None) for i in range(th.ndim//2)] + [slice(1,None) for i in range(th.ndim//2)]
                th = th[tuple(new_shape)]
                # Flatten th[i1,i2,...,j1,j2,...] --> th[i12..., j12...].
                th = th.reshape(np.prod(shape_c_mc), np.prod(shape_e))
                
            # Posterior.
            # x(Ci1, Ci2, ...) = sum_{j1,j2, ...} P(Ci1, Ci2, ...| Ej1, Ej2, ...) * x(Ej1, Ej2, ...)
            # When summing Ej, bin width is considered.
            xC = (th * meas.flatten()[None,:] 
                     * dbin_e_outer.flatten()[None,:]).sum(axis=1)
            
            if self.out_probability: 
                # Normalize the posterior to probability if needed.
                # Note, that sum_i xC = 1 including the Fake!
                xC /= sum(xC * dbin_c_outer.flatten()) 
                
            pos.append(xC)
            ## TODO: Early stopping is when the posterior barely changes.
            #if abs(pos[-1]-pos[-2]).sum()/(pos[-1]-pos[-2]).std()<1e-5: break
        
        return [pos[-1].reshape(shape_c_mc), bins_c_mc]
    
    def ibu_err(self, prior_err_itnum=3, meas_err_itnum=3):
        ''' Estimating uncertainties for ibu.
            
            Algorithm
            ---------
            P(Ci | MC) each event assumed to have Poisson distribution.
            
            Parameters
            ----------
            prior_err_itnum : number of iteration for prior statistical uncertainty.
            meas_err_itnum  : number of iteration for measure statistical uncertainty.

            Returns
            -------
            pos : x(Ci) = [hist, hist_err, bins] posterior distribution.

            Plotting
            --------
            plt.errorbar(bins[:-1]+np.diff(bins)/2, pos_new_err[0], yerr=pos[1], xerr=np.diff(bins)/2)
        '''

        pos_err = [self.ibu()[0]]
        
            
        for i in range(prior_err_itnum):
            # Prior uncertainty.
            # Sampling several equally possible prior probability.
            # TODO: the sampling could be done on the histogram itself,
            # speeding up the algorithm.
            wp  = np.random.poisson(1, size=np.asarray(self.evs_c_mc).shape[1])
            pri = MChistogramdd(self.evs_c_mc, bins=self.bins_mc_new[1], probability=True, w=wp)
            for j in range(meas_err_itnum):
                # Measurememnt uncertainty.
                # Sampling several equally possible measurements.
                wm   = np.random.poisson(1, size=np.asarray(self.evs_e).shape[1])
                meas = MChistogramdd(self.evs_e, bins=self.bins_e, w=wm)
                pos  = self.ibu(measure=meas, prior=pri)
                pos_err.append(pos[0])
                                
        # Calculating the mean and the standard deviation.
        pos_mean = np.mean(np.asarray(pos_err), axis=0)
        pos_var  = np.std(np.asarray(pos_err), axis=0)
        
        return [pos_mean, pos_var, self.p_c_mc[-1]]