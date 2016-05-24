# JFR - This file was adapted from bhmm.transition_matrix_sampling_rev.py and also from stallone
__author__ = 'noe'

import numpy as np
import math
from copy import copy, deepcopy
from pyemma.msm.analysis.dense.decomposition import stationary_distribution_from_backward_iteration
from random import shuffle

__author__ = "Hao Wu, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["Hao Wu", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "Frank Noe"
__email__="frank.noe AT fu-berlin DOT de"


# some shortcuts
eps=np.spacing(0)
log=math.log
exp=math.exp

class TransitionMatrixBiasedSamplerRev(object):
    """
    Reversible transition matrix sampling using the algorithm from Trendelkamp, Noe, JCP, 2013 for variable stationary distribution.
    We have added an adjustment to acceptance probability which corresponds to sampling a normal distribution about some constraint, 
    while employing the unconstrained mle as a prior.
    """

    def __init__(self, _C, _prior, _X): 
        """
        Initializes the transition matrix sampler with the observed count matrix

        Parameters:
        -----------
        C : ndarray(n,n)
            count matrix containing observed counts. Do not add a prior, because this sampler intrinsically
            assumes a -1 prior!

        X : ndarray(n,n)
            expected count matrix, this may be the mle or just the simple symmetrized counts

        """
        # Variables depending on init input
        self.C = np.array(_C, dtype=np.float64)
        self.Xexp = deepcopy(_X)
        self.prior = deepcopy(_prior)
        self.n = self.C.shape[0]
        self.Crowsum = np.sum(self.C, dtype=float, axis=1, keepdims=True)
        self.Ccolsum = np.sum(self.C, dtype=float, axis=0, keepdims=True)
        self.sc = float(self.C.shape[0]*self.C.shape[1])
        self.chng_list = np.array( np.where( self.Xexp > 1e-12 ) ) 
        # 
	self.eps = 0.0
	self.lamb = None
	# To be updated during each step
        self.X = None
	self.T = None
        self.mu = None
        self.EW = None
        self.EW_old = None
        self.u = None
        self.backupRow = None
        self.EQ = None
        self.EQ_old = None
	# global parameter for this class to be passes in as input
	self.beta = None
        self.fixed_pi = False
        self.EQ_CG = None
        self.EQ_AA = None
        self.F_CG = None
        # keeping track of the MC moves
        self.nprop_ndE = 0
        self.nprop_pdE = 0
        self.nacc_pdE = 0
        # degrees of freedom.
        self.dof = np.zeros(self.n)
	# variables associated with the beta distribution
        self.alpha = np.zeros(self.n)
        self.beta = np.zeros(self.n)
        # arrays depending only on C, not T
        for i in range (0, self.n):
            self.dof[i] = self.n
            self.alpha[i] = self.Crowsum[i] + self.dof[i] - self.C[i,i] - 1
            self.beta[i] = self.C[i,i] + 1


    def _is_positive(self, x):
        """
        Helper function, tests if x is numerically positive

        :param x:
        :return:
        """
        return x>=eps and (not math.isinf(x)) and (not math.isnan(x))

    def _X_to_T(self):
        self.T =  self.X / np.sum(self.X, dtype=float, axis=1, keepdims=True)
    def _X_to_mu(self):
        self.mu = np.sum(self.X, dtype=float, axis=1, keepdims=True)

    def _T_to_X(self):
        mu = stationary_distribution_from_backward_iteration(self.T)
        self.X = np.dot(np.diag(mu), self.T)
        self.X /= np.sum(self.X, dtype=float)

    def _update_arrays(self):
        # stat dist
        self._X_to_mu()
        # EW for T
        self.EW_old = deepcopy(self.EW)
        self.EQ_old = deepcopy(self.EQ)


    def _drawMetzner(self, n):

        #Draw (i,j) uniformly from {0,..,n-1}x{0,...,n-1} subject to i<j
        k = int(np.random.random_integers(0, n-1, 1))
        l = int(np.random.random_integers(0, n-1, 1))

        if (self.fixed_pi is True):
            while ( ( np.abs(self.X[k,l]) <= 1e-12 ) or (k == l) ): #and (self.T[k,l] <= 1e-6) ): # JFR-nb: For fixed-pi, only sample non-diagonal elements
                k = int(np.random.random_integers(0, n-1, 1))
                l = int(np.random.random_integers(0, n-1, 1))
        else:
            while ( ( np.abs(self.X[k,l]) <= 1e-12 ) ): # and (self.T[k,l] <= 1e-6) ): # JFR-nb: Not sure about the constraints on T, but I don't want to sample elements that are basically zero.
                k = int(np.random.random_integers(0, n-1, 1))
                l = int(np.random.random_integers(0, n-1, 1))

        # Enforce i<j
        i = min(k,l)
        j = max(k,l)

        return i,j


    def _drawQuad(self, n):

        #Draw (i,j) uniformly from {0,..,n-1}x{0,...,n-1} subject to i<j
        k = int(np.random.random_integers(0, n-1, 1))
        l = int(np.random.random_integers(0, n-1, 1))
        # Exclude i=j
        while ( k==l ):
            k = int(np.random.random_integers(0, n-1, 1))
            l = int(np.random.random_integers(0, n-1, 1))
	
        # Enforce i<j
        i = min(k,l)
        j = max(k,l)
	
        return i,j


    def _drawRow(self, n):

        i = int(np.random.random_integers(0, n-1, 1))
        while ( (self.C[i,i] <= 0) or (self.dof[i] < 2) ):
            i = int(np.random.random_integers(0, n-1, 1))

        return i


    '''
       Conducts a single reversible edge shift MC step @returns true if the step
       has been accepted.
    '''
    def _step_Quad_MC(self, i, j, randU):

        if( (self.C[i,j]>0.0) and (self.C[j,i]>0.0) and (self.C[i,i]>0.0) and (self.C[j,j]>0.0) ):

            q = self.mu[j] / self.mu[i]
            dmin = max(-self.T[i,i], -q * self.T[j,j])
            #dmax = min(self.T[i,j], q * self.T[j, i])
	    dmax = self.T[i,j]

            if (dmin == dmax):
                return 0.00

            if (dmin > dmax):
                raise ValueError('Error during reversible edge shift in Transition Matrix Sampling:'
                        + 'Have reached an inconsistency between elements. dmin > dmax with:'
                        + 'dmin = max(-T[i,i], -q * T[j,j])'
                        + str(dmin) + ' = max(-' + str(self.T[i,i]) + ', -' + str(q) + ' * ' + str(self.T[j,j]) + ')'
                        + 'dmax = min(T[i,j], q * T[j,i])'
                        + str(dmax) + ' = min(' + str(self.T[i,j]) + ', ' + str(q) + ' * ' + str(self.T[j,i]) + ')'
                        + 'at i = ' + str(i) + '   j = ' + str(j))

            d1 = 0.0
            d2 = 0.0

            d1 = randU.uniform(dmin,dmax,1)
            d2 = d1 / q

            DEN = ( (self.T[i,j]**2) + (self.T[j,i]**2) )
	    NUM = ( (self.T[i,j] - d1)**2 )
	    NUM += ( (self.T[j,i] - d2)**2 )
            prop = np.sqrt( NUM / DEN )

            #pacc = prop 
	    #pacc *= (( (self.T[i,i] + d1) / self.T[i,i] )**self.C[i,i])
	    #tmp = ( (self.T[i,i] + d1) / self.T[i,i] )
	    #tmp = tmp**self.C[i,i]
	    #pacc *= tmp
	    #print self.C[i,i]
	    #print tmp
	    #pacc *= (( (self.T[i,j] - d1) / self.T[i,j] )**self.C[i,j]) 
	    #pacc *= (( (self.T[j,j] + d2) / self.T[j,j] )**self.C[j,j])  
	    #pacc *= (( (self.T[j,i] - d2) / self.T[j,i] )**self.C[j,i])  

            #tmp = np.log( (self.T[i,i] + d1) / self.T[i,i] )*(self.C[i,i]) 
            #tmp += np.log( (self.T[i,j] - d1) / self.T[i,j] )*(self.C[i,j])
            #tmp += np.log( (self.T[j,j] + d2) / self.T[j,j] )*(self.C[j,j])
            #tmp += np.log( (self.T[j,i] - d2) / self.T[j,i] )*(self.C[j,i])
	    #tmp = np.exp(tmp)
	    #pacc = prop*tmp
	    #pacc = prop

	    # For now let's just accept moves with neg change in energy
	    Rii = ( (self.T[i,i] + d1) / self.T[i,i] )
	    Rij = ( (self.T[i,j] - d1) / self.T[i,j] )
            Rjj = ( (self.T[j,j] + d2) / self.T[j,j] )
            Rji = ( (self.T[j,i] - d2) / self.T[j,i] )
	    dE = -np.log(Rii*Rij*Rjj*Rji)
	    #print dE
	    #if ( dE < 0 ):
	    #    pacc = 1.0
	    #else:
	    pacc = np.log(prop)
	    #pacc -= self.C[i,i]*dE
            pacc += (self.C[i,i])*np.log(Rii)
            pacc += (self.C[i,j])*np.log(Rij)
	    pacc += (self.C[j,j])*np.log(Rjj)
	    pacc += (self.C[j,i])*np.log(Rji)
	    if ( pacc > 0.0 ):
	        pacc = 1.0
	    else:
		#pacc = 0.00
	    #pacc = 1.0
	        pacc = np.exp(pacc)
	    #    pacc *= Rii**self.C[i,i]
            #    pacc *= Rij**self.C[i,j]
            #    pacc *= Rjj**self.C[j,j]
            #    pacc *= Rji**self.C[j,i]


            # Temporarily update T
            self.T[i,j] -= d1
            self.T[i,i] += (1.0 - np.sum(self.T, dtype=float, axis=1, keepdims=True)[i])
            self.T[j,i] -= d2
            self.T[j,j] += (1.0 - np.sum(self.T, dtype=float, axis=1, keepdims=True)[j])

            # check if there are problems and then revert to backup
            from pyemma.msm.estimation.dense.TransitionMatrixSamplingTools import isElementIn01
            if ( (not isElementIn01(self.T,i,i)) or (not isElementIn01(self.T,i,j)) or (not isElementIn01(self.T,j,i)) or (not isElementIn01(self.T,j,j)) ):
                pacc = 0.00

        else:
	    pacc = 0.0

        return pacc


    def _step_Quad_Gibbs(self, i, j, randU, randE):

        if( (i<j) and (self.C[i,j]+self.C[j,i]>=0.0) and (self.C[i,i]>=0.0) and (self.C[j,j]>=0.0) ):

            # Compute parameters
            a = self.C[i,j] + self.C[j,i]
            delt = self.T[i,i] + self.T[i,j]
            lamb = (self.mu[j]/self.mu[i]) * (self.T[j,j]+self.T[j,i])
            b = 0.0
            c = 0.0
	    d = 0.0

            #Ensure that d won't grow out of bounds
            if( (delt > 1e-15) and lamb > (1e-15) ):
                #Assign parameters according to ordering of delta and lambda
                if( delt <= lamb ):
                    b = self.C[i,i]
                    c = self.C[j,j]
                    d = lamb / delt
                else:
                    b = self.C[j,j]
                    c = self.C[i,i]
                    d = delt / lamb
         
                # Generate random variate
	        from pyemma.msm.estimation.dense import ScaledElementSampler
                x = ScaledElementSampler.sample( randU, randE, a, b, c, d)

                # Proposed quadruple
                Tnew_ij = x * min(delt, lamb)
                Tnew_ii = delt - Tnew_ij
                Tnew_ji = (self.mu[i] / self.mu[j]) * Tnew_ij
                Tnew_jj = (self.mu[i] / self.mu[j]) * lamb - Tnew_ji

                # Acceptance ratio according to Noe JCP08
                rprime = np.sqrt( Tnew_ij*Tnew_ij + Tnew_ji*Tnew_ji )
                r = np.sqrt( self.T[i,j]*self.T[i,j] + self.T[j,i]*self.T[j,i] )
	        if ( r < 1e-6 ):
	            pacc = 0.00
	        else:
	            pacc = min(1.0,rprime/r)

	            # temporarily update T
                    self.T[i,j] = Tnew_ij
                    self.T[i,i] = Tnew_ii
                    self.T[j,i] = Tnew_ji
                    self.T[j,j] = Tnew_jj

            else:
	        pacc = 0.0 # reject this step!
        else:
            pacc = 0.0 # reject this step!

        return pacc


    '''
       Samples from one row shift distribution via a beta distribution
    '''
    def _step_Row_MC(self, i, randU):

        # backup
        self.backupRow = self.T[i,:]

        maxTij = 0.0
        for j in range (0, self.n):
            if (j != i):
                if (self.T[i,j] > maxTij):
                    maxTij = self.T[i,j]

        a = randU.uniform(0, 1.0 / (1.0 - self.T[i,i]), 1)

        #prop = a**(self.dof[i] - 2.0)

        #pacc = prop * (((a * self.T[i,i] - a + 1.0) / self.T[i,i])**self.C[i,i]) * (a**(self.Crowsum[i] - self.C[i,i]))
	# take the log of the acc prob
	pacc = (self.dof[i] - 2.0 + self.Crowsum[i] - self.C[i,i])*np.log(a)
	pacc += self.C[i,i]*np.log( (1.0 - a*(1.0-self.T[i,i])) / self.T[i,i] )
	if (pacc > 0):
	    pacc = 1.0
	else:
	    pacc = np.exp(pacc)  
	    print pacc
        #print 'a'
	#print a

        # update matrix
        sum = 0
        for k in range (0, self.n):
            if (k != i):
                self.T[i,k] *= a
                sum += self.T[i, k]
        self.T[i,i] = 1.0 - sum

        # check if the move is valid
        from pyemma.msm.estimation.dense.TransitionMatrixSamplingTools import isRowIn01
        if (not isRowIn01(self.T, i)):
	    pacc = 0.0
	    self.T[i,:] = self.backupRow[:]

        return pacc


    '''
    This function samples from one row shift distribution via a beta distribution
    '''
    def _step_Row_Gibbs(self, i, randB):
	
        # x = rowDistribution[i].nextDouble() # JFR - each row has its own beta distribution?!
        x = randB.beta(self.alpha[i], self.beta[i], 1)
        a = x / (1.0 - self.T[i,i])

        # backup
        self.backupRow = self.T[i,:]

        # update matrix
        sum = 0.0
        for k in range (0,self.n):
            if (k != i):
                self.T[i,k] *= a
                # ensureValidElement(i, k);
                sum += self.T[i,k]
        self.T[i,i] =  1.0 - sum

        # check if there are problems and then revert to backup
	from pyemma.msm.estimation.dense.TransitionMatrixSamplingTools import isRowIn01
        if ( isRowIn01(self.T,i) ):
            self.u[i] += np.log(a)
            self.mu[i] = np.exp(-self.u[i])

            # rescale u if necessary
            if (np.abs(np.min(self.u)) > 1):
                self.u -= np.min(self.u)
                for k in range (0, self.n):
                    self.mu[k] = np.exp(-self.u[k])

	    return 1.0

        else:
            # restore Row
	    self.T[i,:] = self.backupRow

	    return 0.0

	
	return -1

    def _update_pacc_scalar(self, pacc, W, wavg, wavg_old):
        '''
	    This function adjusts the acceptance probability from the normal procedure (with no prior), to take the constraints into account
	'''
	#sig = 1.0 # JFR - This should eventually be an input parameter
        arg = ((W-wavg)**2) - ((W-wavg_old)**2)
	if ( arg >= 0 ):
	    npacc = pacc * np.exp( (-1.0 / (2.0*(self.sig**2))) * ( arg ) ) 
	else:
	    npacc = 1.0    

	npacc = min(1.0, npacc)

	return npacc

    def _update_pacc(self, pacc):
        '''
            This function adjusts the acceptance probability from the normal procedure (with no prior), to take the constraints into account
        '''
	#npacc = pacc
	npacc = 1.0
        self.dOBS = np.ones(self.OBS.shape[0])
	for i in range (0, self.OBS.shape[0]):
            self.dOBS[i] = self.OBS[i]-self.WT[i]
            arg = ((self.OBS[i]-self.WT[i])**2) - ((self.OBS[i]-self.WT_old[i])**2)
            if ( arg >= 0 ):
                npacc *= np.exp( (-1.0 / (2.0*(self.sig[i]**2))) * ( arg ) )
            else:
                npacc *= 1.0

        #npacc = min(1.0, npacc) # JFR-nb:  This is not necessary and can be used as some sort of test that the above expressions are correct.

	# weight the importance of the prior and the constraint
	npacc = (npacc**(1.0-self.lamb)) * (pacc**self.lamb)

	#npacc = min(1.0, npacc) # JFR-nb: Same as above, not necessary.

        return npacc

    def _logprob_T(self):
        assert np.all(self.T >= 0)
        return np.sum(np.multiply(self.C, np.ma.log(self.T)), dtype=float) # JFR-nb: avoid 0 elements in the log arg with np.ma.log 

    def _EQ(self):
        self.EQ = deepcopy( -1.0*(self._logprob_T()) )
        self.EQ -= self.EQ_CG
        self.EQ /= np.abs(self.EQ_AA - self.EQ_CG)

    def _EW(self,FW):
        self.EW = deepcopy( FW(self.T) )
        self.EW /= self.F_CG

    def _step_Metzner_MC(self, i, j, randU):
        '''
        metzner_mcmc_step(const double* Z, const double* N, double* K,
                      double* Q, const double* random, double* sc, int n_states,
                      int n_steps)
        '''
	a = 0.0
	b = 0.0
        if (self.fixed_pi is True):
            frac = 0.99 # just in case of some rounding error
            a = -frac*self.X[i,j]
            b = frac*min(self.X[i,i],self.X[j,j])
        else:
            sc = np.sum(self.X, dtype=float)
            kmin = 0.999 * sc # Values from Metzner paper, rescaled for arbitrary X normalization
            kpls = 1.001 * sc
            if (i == j):
                a = max(-self.X[i,j], kmin - sc)
                b = kpls - sc
            else: 
                a = max(-self.X[i,j], 0.5*(kmin - sc))
                b = 0.5 * (kpls - sc)
           
        # store the old log prob - already have these in self.EQ_old now
	#lpold = self._logprob_T()
        #EQold = deepcopy(self.EQ)
        # get and make the MC move
        self.eps = randU.uniform(a,b,1)
	self.X[i,j] += self.eps
	if ( i != j ):
	    self.X[j,i] += self.eps
        if (self.fixed_pi is True):
            self.X[i,i] += (self.mu[i] - np.sum(self.X, dtype=float, axis=1, keepdims=True)[i]) # -= self.eps
            if (abs(self.X[i,i]) < 1e-12):
                self.X[i,i] = 0.00
            self.X[j,j] += (self.mu[j] - np.sum(self.X, dtype=float, axis=1, keepdims=True)[j]) # -= self.eps
            if (abs(self.X[j,j]) < 1e-12):
                self.X[j,j] = 0.00

        # update T
	self._X_to_T()
        # get the new log prob
        #lp = self._logprob_T()
        self._EQ()
	#if ( (lp - lpold) > 0 ):
	#    pacc = 1.00
	#else:
	#    pacc = np.exp( lp - lpold )

	return (self.EQ - self.EQ_old) # pacc - JFR - just return the energy difference, dEQ

    def _update_Metzner_MC_nofail(self, n_step, FW):
        """
        MC sampler for reversible transiton matrix, following Metzner, Noe, Schutte, PRE (2010).
        Sampler is extended to fixed stationary distributions and to an extended MC energy function with constraints.
        This particular 'nofail' version of the sampler attempts to make a move until n_step successful moves, rejected moves are not counted!
        """

        pacc = 0.0

        # Initialize the random number generator
        seed = int(np.random.random_integers(0,1e7,1)) # generating a random seed for each process
        randU  = np.random.RandomState(seed)

        for iter in range(n_step): # Loop over MC sampling iterations, return a sample after n_step SUCCESSFUL steps
            failed_step = True
            while ( failed_step ):

		i, j = self._drawMetzner(self.n)
                saveij = deepcopy(self.X[i,j])
                saveii = deepcopy(self.X[i,i])
                savejj = deepcopy(self.X[j,j])
		dEQ = self._step_Metzner_MC(i, j, randU)

                if (FW is not None): # calculate the constraint energy from the given error function
                    self._EW(FW)
                    dEW = self.EW - self.EW_old 
                    dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW
                else:
                    dEtot = deepcopy(dEQ)

                if (dEtot <= 0.0): # accept the move
                    self.nprop_ndE += 1
                    failed_step = False
                    self._update_arrays()
                else:
                    self.nprop_pdE += 1
                    pacc = np.exp( -self.beta*dEtot )
                    if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                        failed_step = True
		        self.X[i,j] = deepcopy(saveij)
                        self.X[j,i] = deepcopy(saveij)
                        self.X[i,i] = deepcopy(saveii)
                        self.X[j,j] = deepcopy(savejj)
		        self._X_to_T()
                    else:
                        failed_step = False
                        self.nacc_pdE += 1 # JFR - keeping track of the accepted moves
                        self._update_arrays()

    def _update_Metzner_MC_fixedstep(self, n_step, FW):
        """
        MC sampler for reversible transiton matrix, following Metzner, Noe, Schutte, PRE (2010).
        Sampler is extended to fixed stationary distributions and to an extended MC energy function with constraints.
        This particular 'fixedstep' version of the sampler makes a fixed number of trial moves during a single sweep (whether or not they are accepted).
        Additionally, a list of changeable elements is used to enforce that a single move is attempted on each of these elements during each sweep.
        """

        pacc = 0.0

        # Set the number of moves based on the changeable list of elements
        n_step = self.chng_list.shape[1] # nb - the n_step input parameter is being ignored!

        # Initialize the random number generator
        seed = int(np.random.random_integers(0,1e7,1)) # generating a random seed for each process
        randU  = np.random.RandomState(seed)

        # Shuffle the changeable list of elements
        shuff_chng_list = deepcopy(self.chng_list)
        shuffle(shuff_chng_list)

        fp = open('./EW_data_test_lamb-'+str(self.lamb)+'.dat', 'w+')

        #for ind in range(n_step): # Loop over the changeable elements
        for ind in range(500):

            i = shuff_chng_list[0][ind]
            j = shuff_chng_list[1][ind]
            saveij = deepcopy(self.X[i,j])
            saveii = deepcopy(self.X[i,i])
            savejj = deepcopy(self.X[j,j])
            dEQ = self._step_Metzner_MC(i, j, randU)

            if (FW is not None): # calculate the constraint energy from the given error function
                if ( (ind%10) == 0 ):
                    self._EW(FW) # I would like to remove this so that it is only calculated once per sweep, then I will need to input the old value
                    fp.write(str(self.EW))
                    fp.write('\n')
                    dEW = self.EW - self.EW_old
                else:
                    dEW = 0.0
                dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW
            else:
                dEtot = deepcopy(dEQ)

            if (dEtot <= 0.0): # accept the move
                self.nprop_ndE += 1
                self._update_arrays()
            else:
                self.nprop_pdE += 1
                pacc = np.exp( -self.beta*dEtot )
                if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                    self.X[i,j] = deepcopy(saveij)
                    self.X[j,i] = deepcopy(saveij)
                    self.X[i,i] = deepcopy(saveii)
                    self.X[j,j] = deepcopy(savejj)
                    self._X_to_T()
                else:
                    self.nacc_pdE += 1 # JFR - keeping track of the accepted moves
                    self._update_arrays()


    def _update_Metzner_MC_fixedstep_corrmove(self, n_step, FW):
        """
        MC sampler for reversible transiton matrix, following Metzner, Noe, Schutte, PRE (2010).
        Sampler is extended to fixed stationary distributions and to an extended MC energy function with constraints.
        This particular 'fixedstep' version of the sampler makes a fixed number of trial moves during a single sweep (whether or not they are accepted).
        Additionally, a list of changeable elements is used to enforce that a single move is attempted on each of these elements during each sweep.
        """

        pacc = 0.0

        # Set the number of moves based on the changeable list of elements
        #n_step = self.chng_list.shape[1] # nb - the n_step input parameter is being ignored!
        n_step = 10000 # DEBUG

        # Initialize the random number generator
        seed = int(np.random.random_integers(0,1e7,1)) # generating a random seed for each process
        randU  = np.random.RandomState(seed)

        # Shuffle the changeable list of elements
        shuff_chng_list = deepcopy(self.chng_list)
        shuffle(shuff_chng_list)

        fp = open('./EW_data_test_lamb-'+str(self.lamb)+'.dat', 'w+')

        step_ctr = 0
        while ( step_ctr < n_step ):

            # first choose the step length at random
            step_len = int(np.random.random_integers(1,min(self.X.shape[0],n_step-step_ctr),1))
            # save the old data
            Xold = deepcopy(self.X)
            EQold = deepcopy(self.EQ)

            for ind in range(step_ctr, step_ctr+step_len-1): # Loop over the changeable elements

                i = shuff_chng_list[0][ind]
                j = shuff_chng_list[1][ind]
                saveij = deepcopy(self.X[i,j])
                saveii = deepcopy(self.X[i,i])
                savejj = deepcopy(self.X[j,j])
                dEQ = self._step_Metzner_MC(i, j, randU)

                # evaluate the energy as if EW did not change
                dEW = 0.0
                dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

                if (dEtot <= 0.0): # accept the move
                    self._update_arrays()
                else:
                    pacc = np.exp( -self.beta*dEtot )
                    if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                        self.X[i,j] = deepcopy(saveij)
                        self.X[j,i] = deepcopy(saveij)
                        self.X[i,i] = deepcopy(saveii)
                        self.X[j,j] = deepcopy(savejj)
                        self._X_to_T()
                    else:
                        self._update_arrays()    

            # On the final sub-move, evaluate EW
            i = shuff_chng_list[0][step_ctr+step_len-1]
            j = shuff_chng_list[1][step_ctr+step_len-1]
            dEQ = self._step_Metzner_MC(i, j, randU)
            dEQ = self.EQ - EQold

            self._EW(FW)
            dEW = self.EW - self.EW_old
            dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

            if (dEtot <= 0.0): # accept the move
                self.nprop_ndE += 1
                self._update_arrays()
            else:
                self.nprop_pdE += 1
                pacc = np.exp( -self.beta*dEtot )
                if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                    self.X = deepcopy(Xold)
                    self._X_to_T()
                    self._X_to_mu()
                    self.EQ_old = deepcopy(EQold)
                else:
                    self.nacc_pdE += 1 # JFR - keeping track of the accepted moves
                    self._update_arrays()

            fp.write('len = '+str(step_len))
            fp.write('\n')
            fp.write('EW = '+str(self.EW))
            fp.write('\n')
            fp.write('nprop_ndE = '+str(self.nprop_ndE))
            fp.write('\n')
            fp.write('nprop_pdE = '+str(self.nprop_pdE))
            fp.write('\n')
            fp.write('nacc_pdE = '+str(self.nacc_pdE))
            fp.write('\n')
            fp.write('\n')
            step_ctr += step_len


    def _update_MC(self, n_step, eval_const, W):
        """
        MC sampler for reversible transiton matrix
        Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)
        """
        if (eval_const is not None):
            wavg_old = eval_const(self.T)
        pacc = 0.00

        # Initialize the random number generators
        seed = int(np.random.random_integers(0,1e7,1)) # JFR - generating a random seed for each process
        randU  = np.random.RandomState(seed)
        randU2 = np.random.RandomState(seed)

        for iter in range(n_step): # Loop over MC sampling iterations, return a sample after n_step SUCCESSFUL steps
            failed_step = True
            while ( failed_step ):
                # weights for step types, JFR - note that I changed the check to zeros instead of -1,-2 to account for no -1 prior, change this back now for unbiased sampling with a -1 prior
                dofQuad = 0
                dofRow = 0
                for i in range (0,self.n):
                    if ( (self.C[i,i] > -1 ) and (self.Crowsum[i] > - self.n) ):
                        dofRow += 1
                    for j in range (i+1,self.n):
                        if (self.C[i,j]+self.C[j,i] > -2):
                            dofQuad += 1

                p_step_row = float(dofRow) / (float(dofQuad) + float(dofRow))

                # make a copy of T
                T = self.T[:]
                quadstep = False
                self.nprop += 1
                pacc = 0.0
                if ( np.random.uniform(0,1,1) < p_step_row ):
		#if ( 0 < p_step_row ):
                    i, j = self._drawQuad(self.n)
		    #print i, j
                    quadstep = True
                    pacc = self._step_Quad_MC(i, j, randU)
                    #print pacc
                    #print 'quadstep'
                else:
                    i = self._drawRow(self.n)
		    #print i
                    pacc = self._step_Row_MC(i,randU2)
                    #print pacc
                    #print 'rowstep'

                if ( np.abs(pacc) < 1e-12 ): # This means the step was rejected
		    self.T = T[:] # restore the matrix
                    continue

                # evaluate the constraints as a function of T
                if (eval_const is not None):
                    wavg = eval_const(self.T)
                    #print wavg

                    # update the acceptance probability to take the constraints into account
                    pacc = self._update_pacc(pacc, wavg, wavg_old)
                    #print pacc

                if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                    failed_step = True
                    if ( quadstep == True ):
                        self.T[i,j] = T[i,j]
                        self.T[i,i] = T[i,i]
                        self.T[j,i] = T[j,i]
                        self.T[j,j] = T[j,j]
                    else:
                        self.T[i,:] = T[i,:]
                else:
                    failed_step = False
                    self.nacc += 1 # JFR - keeping track of the accepted moves
                    self._update_arrays(False) # JFR - update arrays based on the new T


    def _update_Gibbs(self, n_step, eval_const, W):
        """
        Gibbs sampler for reversible transiton matrix
        Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)
        """
	if (eval_const is not None):
            wavg_old = eval_const(self.T)
	pacc = 0.00

        # Initialize the random number generators
        seed = int(np.random.random_integers(0,1e7,1)) # JFR - generating a random seed for each process
        randU = np.random.RandomState(seed)
        randE = np.random.RandomState(seed)
        randB = np.random.RandomState(seed)


        for iter in range(n_step): # Loop over Gibbs sampling iterations, return a sample after n_step SUCCESSFUL steps
	    failed_step = True
	    while ( failed_step ):
                # weights for step types, JFR - note that I changed the check to zeros instead of -1,-2 to account for no -1 prior, change this back now for unbiased sampling with a -1 prior
                dofQuad = 0 
	        dofRow = 0
                for i in range (0,self.n):
                    if ( (self.C[i,i] > -1 ) and (self.Crowsum[i] > - self.n) ):
                        dofRow += 1
                    for j in range (i+1,self.n):
                        if (self.C[i,j]+self.C[j,i] > -2):
                            dofQuad += 1

                p_step_row = float(dofRow) / (float(dofQuad) + float(dofRow))
	    
                # make a copy of T
	        T = self.T[:]
	        quadstep = False
	        self.nprop += 1
                pacc = 0.0
	        if ( np.random.uniform(0,1,1) < p_step_row ):
                    i, j = self._drawQuad(self.n)
                    quadstep = True
                    pacc = self._step_Quad_Gibbs(i, j, randU, randE)
                    #print pacc
		    #print 'quadstep'
	        else:
	            i = self._drawRow(self.n)
		    pacc = self._step_Row_Gibbs(i, randB)
		    #print pacc
	            #print 'rowstep'

	    	
	        if ( np.abs(pacc) < 1e-12 ): # This means the row step was rejected and the T matrix has already been restored
	            continue
	    
	        # evaluate the constraints as a function of T
		if (eval_const is not None):
	            wavg = eval_const(self.T)
   	            #print wavg
       
                    # update the acceptance probability to take the constraints into account
	            pacc = self._update_pacc(pacc, wavg, wavg_old)
	            #print pacc
	    
                if( randU.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
		    failed_step = True
                    if ( quadstep == True ): 
                        self.T[i,j] = T[i,j]
                        self.T[i,i] = T[i,i]
                        self.T[j,i] = T[j,i]
                        self.T[j,j] = T[j,j]
                    else:
		        self.T[i,:] = T[i,:]
	        else:
		    failed_step = False
                    self.nacc += 1 # JFR - keeping track of the accepted moves
                    self._update_arrays(False) # JFR - update arrays based on the new T
	


    def sample(self, n_step, T_init = None, X_init = None, EQ_CG = None, EQ_AA = None, F_const_err = None, F_CG = None, beta = None, lamb = None, fixed_pi = False):
        """
        Runs n_step successful! Metzner (or fixed-pi Metzner-type) sampling steps and returns a new transition matrix.

        Parameters:
        -----------
        n_step : int
            number of Metzner-type sampling steps.
        T_init : ndarray (n,n)
            initial transition matrix. If not given, will start from C+C.T, row-normalized
        X_init: ndarray (n,n)
            initial count matrix.  If given, the initial T will be inferred directly.
        EQ_CG: float
            the MSM energy of the best CG model (i.e., the mle)
        EQ_AA: float
            the MSM energy of the AA model (mle)
        F_const_err: func 
            input -> T, output -> vector of estimated observables
        F_CG: float
            the measured error of the CG model (mle) wrt the constraints
        beta: float
            the sampling temperature
        lamb: float 
            the interpolation parameter between MSM and const. energy functions
        fixed_pi: bool
            True => keep the stationary distribution fixed when sampling.

        Returns:
        --------
        self.T: ndarray (n,n)
            The transition matrix after n_step sampling steps
        self.X: ndarray (n,n)
            The count matrix after n_step sampling steps
        self.mu: ndarray (n)
            The stationary distribution after n_step sampling steps
        self.EQ: float
            The MSM energy of the output matrix
        self.EW: float
            The const energy (error) of the output matrix

        Raises:
        -------
        ValueError
            if T_init is not a reversible transition matrix

        """
        # input
        if (beta is not None):
            self.beta = deepcopy(beta)
        else: 
            self.beta = 1.0
        if (lamb is not None):
            self.lamb = deepcopy(lamb)
        else:
            self.lamb = 1.0
        if (fixed_pi is not None):
            self.fixed_pi = deepcopy(fixed_pi)
        else:
            self.fixed_pi = False
        if (EQ_CG is not None):
            self.EQ_CG = deepcopy(EQ_CG)
        else:
            self.EQ_CG = 0.0
        if (EQ_AA is not None):
            self.EQ_AA = deepcopy(EQ_AA)
        else:
            self.EQ_AA = 1.0
        if (F_CG is not None):
            self.F_CG = deepcopy(F_CG)
        else:
            self.F_CG = 1.0
        # T_init given?
        if (X_init is not None):
            self.X = deepcopy(X_init)
            self._X_to_T()
        elif (T_init is not None):
            self.T = deepcopy(T_init)
	    self._T_to_X()
	else:
	    self.X = 0.5 * (self.C + self.C.T)
	    self._X_to_T()
            #raise ValueError('not reading X_init properly')
        # reversible?
        if not np.allclose(self.X, self.X.T):
            raise ValueError('Initial transition matrix is not reversible.')

        # init the arrays 
        self._X_to_mu()
        if (F_const_err is not None):
            self._EW(F_const_err)
            self.EW_old = deepcopy(self.EW)
        self._EQ()
        self.EQ_old = deepcopy(self.EQ)

        # adjust the changeable element list for fixed-pi
        if (self.fixed_pi is True):
            for ind in range(self.chng_list.shape[1]):
                if ( self.chng_list[0][ind] == self.chng_list[1][ind] ):
                    self.chng_list = np.delete(self.chng_list,(ind),axis=1)
          
	#self._update_Gibbs(n_step, eval_const, OBS)
        #self._update_MC(n_step, eval_const, OBS)
	#self._update_Metzner_MC_nofail(n_step, F_const_err)
        #self._update_Metzner_MC_fixedstep(n_step, F_const_err)
        self._update_Metzner_MC_fixedstep_corrmove(n_step, F_const_err)
	
        #return self.T, self.X, self.mu, self.EQ, self.EW # JFR - don't need to return anything, the variables exist within the sampler data structure!


    #TODO: Should be used for efficiency purposes. Currently we just call sample.
    def sample_func(self, eval_fun, n_sample, T_init = None):
        """
        Samples the function of T given.

        eval_fun : python-function
            a function that uses a transition matrix as input
        n_step : int
            number of Gibbs sampling steps. Every step samples from the conditional distribution of that element.
        T_init : ndarray (n,n)
            initial transition matrix. If not given, will start from C+C.T, row-normalized

        Returns:
        --------
        The function value after n_step sampling steps
        """
        T = self.sample(n_sample, T_init = T_init)
        return eval_fun(T)



def main():
    """
    This is a test function

    :return:
    """

    # plot histogram
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from timeit import default_timer as timer

    #C = np.array([[2,1,0],
    #              [1,5,1],
    #              [1,2,10]])
    C = np.array([[787, 54, 27],
                  [60, 2442, 34],
                  [22, 39, 6534]], dtype = np.int32)


    T = pyemma.msm.transition_matrix(C, reversible=True)
    #import pyemma.msm.analysis.timescales as ts
    #def OBS_ts (T):
#	import pyemma.msm.analysis.timescales as ts
#        ts_est = ts(T)

#        return ts_est[1]
#    OBS = OBS_ts(T)
#    nstep = 10
#    import pyemma.msm.estimation.dense.transition_matrix_biased_sampling_rev.TransitionMatrixBiasedSamplerRev.sample as sample

#    Topt = sample(nstep, T_init = T, eval_const = OBS_ts, OBS = OBS)

    print T
#    print Topt

#    t1 = timer()
#    nsample = 300000
#    nstep   = 1
#    x = np.zeros(nsample)
#    y = np.zeros(nsample)
#    for i in range(nsample):
#        P = sampler.sample(nstep)
        #ts.update(C, sumC, n, X, nstep)
        #P = X/X.sum(axis=1)[:,None]
#        x[i] = P[0,1]
#        y[i] = P[1,0]
#    t2 = timer()
#    print (t2-t1)


#    plt.hist2d(x, y, bins=100, range=((0,1),(0,1)), cmap=cm.jet)
#    plt.colorbar()
#    plt.xlim(0,1)
#    plt.ylim(0,1)
#    plt.savefig('sample_c.png')


#if __name__ == "__main__":
#    main()
