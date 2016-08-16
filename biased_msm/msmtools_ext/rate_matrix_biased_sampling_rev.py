# JFR - This file was adapted from bhmm.transition_matrix_sampling_rev.py and also from stallone
__author__ = 'noe'

import numpy as np
import math
from copy import copy, deepcopy
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from random import shuffle
import scipy

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

class RateMatrixBiasedSamplerRev(object):
    """
    Reversible rate matrix sampling using Metzner-type sampling (2010), with the rate matrix construction from McGibbon and Pande (2015).
    """

    def __init__(self, _C, _prior, _S, _MC_type, _lT, _tau): 
        """
        Initializes the rate matrix sampler with the observed count matrix

        Parameters:
        -----------
        C : ndarray(n,n)
            count matrix containing observed counts.

        S : ndarray(n,n)
            this is sort of the analog of the expected count matrix, when working in the rate matrix representation (see McGibbon,2015) 

        """
        # Variables depending on init input
        self.C = np.array(_C, dtype=np.float64)
        self.Xexp = deepcopy(_S)
        self.prior = deepcopy(_prior)
        self.MC_types = [ 'fixed_step', 'corr_move' ]
        self.MC_type = deepcopy(_MC_type)
        self.lT = _lT
        self.tau = _tau
        self.n = self.C.shape[0]
        self.Crowsum = np.sum(self.C, dtype=float, axis=1, keepdims=True)
        self.Ccolsum = np.sum(self.C, dtype=float, axis=0, keepdims=True)
        self.sc = float(self.C.shape[0]*self.C.shape[1])
        #self.chng_list = np.array( np.where( self.Xexp > 1e-12 ) ) 
        self.chng_list = np.array( np.where( self.Xexp > -1e5 ) ) # try changing all elements
        self.chng_list = np.delete(self.chng_list, np.where( self.chng_list[0][:] <= self.chng_list[1][:] ) ,axis=1) # the diagonal elements are always constrained in this case, also only need to consider the upper triangle
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

    def _K_to_T(self):
	return scipy.linalg.expm(self.tau*self.T, q=None)

    def _S_to_K(self):
        self.T = -1.0*np.dot( np.dot(np.diag(np.sqrt(self.mu)**(-1)),self.X), np.diag(np.sqrt(self.mu)) )
        for i in range(self.T.shape[0]):
            self.T[i,i] = -( np.sum(self.T[i]) - self.T[i,i] )

    def _K_to_mu(self):
        self.mu = stationary_distribution(self._K_to_T())

    def _K_to_S(self):
        self.X = np.dot( np.dot(np.diag(np.sqrt(self.mu)),self.T), np.diag(np.sqrt(self.mu)**(-1)) )
        self.X /= np.sum(self.X, dtype=float)

    def _update_arrays(self, flag_acc):
        if ( flag_acc ):
            # Energies
            self.EW_old = deepcopy(self.EW)
            self.EQ_old = deepcopy(self.EQ)
        else:
            # Energies
            self.EW = deepcopy(self.EW_old)
            self.EQ = deepcopy(self.EQ_old)


    def _logprob_T(self):
        T = self._K_to_T()
        assert np.all(T >= 0)
        return np.sum(np.multiply(self.C, np.ma.log(T)), dtype=float) # avoid 0 elements in the log arg with np.ma.log 

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
        # Always make a "fixed_pi" move
        #if (self.fixed_pi is True):
        frac = 0.1 # only allow changes by this fraction of the current value
        sc_frac = 0.01
        sc = np.sum(self.X, dtype=float) # also put a constraint on the total sum
        Sdel = np.abs((self.mu[i] / self.mu[j]) + (self.mu[j] / self.mu[i]) - 2)
        Sdel = max(Sdel,sc_frac)
        # nb - off diagonal elements of S are neg and diagonal are pos
        a0 = max( frac*self.X[i,j], -frac*min(self.X[i,i],self.X[j,j]), -sc_frac/Sdel )
        b0 = sc_frac/Sdel
        # now, add restrictions for the diagonal elements
        a1 = -(self.mu[i] / self.mu[j]) * self.X[i,i]
        a2 = -(self.mu[j] / self.mu[i]) * self.X[j,j]
        # now, get the bounds
        a = max(a0,a1,a2)
        b = b0
        if ( np.abs(self.X[i,j]) < 1e-12 ): # if the element is zero, only allow positive moves
            a = 0.

        # check the bounds
        if ( a > 0 or b < 0 ):
            print 'a = '+str(a)
            print 'b = '+str(b)
            print 'i = '+str(i)
            print 'j = '+str(j)
            print 'eps = '+str(eps)
            np.save('S_debug',self.X)
            np.save('mu_debug',self.mu)
            raise ValueError('Something is wrong with the bounds in _step_Metzner_MC()!')
        # get and make the MC move
        self.eps = randU.uniform(a,b,1)
	self.X[i,j] -= self.eps
	self.X[j,i] -= self.eps
        #self.X[i,i] += (self.mu[i]/self.mu[j])*self.eps
        self.X[i,i] = -np.sum(self.X, axis=1)[i] + self.X[i,i]
        #self.X[j,j] += (self.mu[j]/self.mu[i])*self.eps
        self.X[j,j] = -np.sum(self.X, axis=1)[j] + self.X[j,j]

        # update T
	self._S_to_K()
        # get the new log prob
        self._EQ()

	return (self.EQ - self.EQ_old) # return the energy difference, dEQ

    def _step_redist_pi(self, i, j, randU):
        '''
          change the stationary distribution
        '''
        frac = 1e-3 # only allow changes by this fraction of the current value
        a0 = -frac*self.mu[i]
        b0 = frac*self.mu[j] 
        # now, get the bounds
        a = a0
        b = b0

        # check the bounds
        if ( a > 0 or b < 0 ):
            raise ValueError('Something is wrong with the bounds in _step_redist_pi()!')
        # get and make the MC move
        self.eps = randU.uniform(a,b,1)
        self.mu[i] += self.eps
        self.mu[j] -= self.eps

        # update K
        self._S_to_K()

        # get the new log prob
        self._EQ()

        return (self.EQ - self.EQ_old) # return the energy difference, dEQ


    def _update_Metzner_MC_fixedstep(self, n_step, FW):
        """
        MC sampler for reversible transiton matrix, following Metzner, Noe, Schutte, PRE (2010).
        Sampler is extended to fixed stationary distributions and to an extended MC energy function with constraints.
        This particular 'fixedstep' version of the sampler makes a fixed number of trial moves during a single sweep (whether or not they are accepted).
        Additionally, a list of changeable elements is used to enforce that a single move is attempted on each of these elements during each sweep.
        """

        pacc = 0.0

        # Set the number of moves based on the changeable list of elements
        n_MC = self.chng_list.shape[1] # nb - the n_step input parameter is being ignored!

        # Initialize the random number generator
        seed = int(np.random.random_integers(0,1e7,1)) # generating a random seed for each process
        randU  = np.random.RandomState(seed)

        # Shuffle the changeable list of elements
        shuff_chng_list = deepcopy(self.chng_list)
        shuff_chng_list = shuff_chng_list.T
        np.random.shuffle(shuff_chng_list)
        shuff_chng_list = shuff_chng_list.T

        for ind in range(n_MC): # Loop over the changeable elements

            # get the indices
            i = shuff_chng_list[0][ind]
            j = shuff_chng_list[1][ind]
            # save the old state
            saveij = deepcopy(self.X[i,j])
            saveii = deepcopy(self.X[i,i])
            savejj = deepcopy(self.X[j,j])
            # make a fixed_pi MC move
            dEQ = self._step_Metzner_MC(i, j, randU)

            # calculate the constraint energy from the given error function
            self._EW(FW) # This is expensive!
            dEW = self.EW - self.EW_old
            dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

            if (dEtot <= 0.0): # accept the move
                self.nprop_ndE += 1
                self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW
            else:
                self.nprop_pdE += 1
                pacc = np.exp( -self.beta*dEtot )
                if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                    # restore the old state
                    self.X[i,j] = deepcopy(saveij)
                    self.X[j,i] = deepcopy(saveij)
                    self.X[i,i] = deepcopy(saveii)
                    self.X[j,j] = deepcopy(savejj)
                    self._S_to_K()
                    self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                else:
                    self.nacc_pdE += 1 # keeping track of the accepted moves
                    self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW

            if ( self.fixed_pi is False ): # also try to redist the stat dist
                # save the old state
                save_ii = deepcopy(self.mu[i])
                save_jj = deepcopy(self.mu[j])
                # make the MC move
                dEQ = self._step_redist_pi(i, j, randU)

                # calculate the constraint energy from the given error function
                self._EW(FW) # This is expensive!
                dEW = self.EW - self.EW_old
                dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

                if (dEtot <= 0.0): # accept the move
                    self.nprop_ndE += 1
                    self._update_arrays( True ) # EQ_old -> EQ, EW_old -> EW
                else:
                    self.nprop_pdE += 1
                    pacc = np.exp( -self.beta*dEtot )
                    if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                        # restore the old state
                        self.mu[i] = deepcopy(saveii)
                        self.mu[j] = deepcopy(savejj)
                        self._S_to_K()
                        self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                    else:
                        self.nacc_pdE += 1 # keeping track of the accepted moves
                        self._update_arrays( True ) # EQ_old -> EQ, EW_old -> EW


    def _update_Metzner_MC_fixedstep_corrmove(self, n_step, FW):
        """
        MC sampler for reversible transiton matrix, following Metzner, Noe, Schutte, PRE (2010).
        Sampler is extended to fixed stationary distributions and to an extended MC energy function with constraints.
        This particular 'fixedstep' version of the sampler makes a fixed number of trial moves during a single sweep (whether or not they are accepted).
        Additionally, a list of changeable elements is used to enforce that a single move is attempted on each of these elements during each sweep.
        """

        pacc = 0.0

        # Set the number of moves based on the changeable list of elements
        n_MC = self.chng_list.shape[1]

        # Initialize the random number generator
        seed = int(np.random.random_integers(0,1e7,1)) # generating a random seed for each process
        randU  = np.random.RandomState(seed)

        # Shuffle the changeable list of elements
        shuff_chng_list = deepcopy(self.chng_list)
        shuffle(shuff_chng_list)

        step_ctr = 0
        while ( step_ctr < n_MC ):

            # set the corr len from input 
            step_max = n_step
            step_len = min(step_max,n_MC-step_ctr) # for the last step, in case n_MC is not perfectly divisible by n_step
            # save the old data
            Sold = deepcopy(self.X)
            EQold = deepcopy(self.EQ)

            for ind in range(step_ctr, step_ctr+step_len): # Loop over the changeable elements

                # get the indices
                i = shuff_chng_list[0][ind]
                j = shuff_chng_list[1][ind]
                # save the old state
                saveij = deepcopy(self.X[i,j])
                saveii = deepcopy(self.X[i,i])
                savejj = deepcopy(self.X[j,j])
                # make a fixed_pi MC move
                dEQ = self._step_Metzner_MC(i, j, randU)

                # evaluate the energy as if EW did not change
                dEW = 0.0
                dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

                if (dEtot <= 0.0): # accept the move
                    self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW
                else:
                    pacc = np.exp( -self.beta*dEtot )
                    if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                        # restore the old state
                        self.X[i,j] = deepcopy(saveij)
                        self.X[j,i] = deepcopy(saveij)
                        self.X[i,i] = deepcopy(saveii)
                        self.X[j,j] = deepcopy(savejj)
                        self._S_to_K()
                        self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                    else:
                        self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW    

                if ( self.fixed_pi is False ): # also try to redist the stat dist
                    # save the old state
                    save_ii = deepcopy(self.mu[i])
                    save_jj = deepcopy(self.mu[j])
                    # make the MC move
                    dEQ = self._step_redist_pi(i, j, randU)

                    # evaluate the energy as if EW did not change
                    dEW = 0.0
                    dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

                    if (dEtot <= 0.0): # accept the move
                        self._update_arrays( True ) # EQ_old -> EQ, EW_old -> EW
                    else:
                        pacc = np.exp( -self.beta*dEtot )
                        if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                            # restore the old state
                            self.mu[i] = deepcopy(saveii)
                            self.mu[j] = deepcopy(savejj)
                            self._S_to_K()
                            self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                        else:
                            self._update_arrays( True ) # EQ_old -> EQ, EW_old -> EW
            
            # Afer the final sub-move, evaluate a MC step according to EW alone
            dEQ = 0.0
            self._EW(FW)
            dEW = self.EW - self.EW_old
            dEtot = self.lamb*dEQ + (1.0-self.lamb)*dEW

            if (dEtot <= 0.0): # accept the move
                self.nprop_ndE += 1
                # arrays should be current, just update EW
                self.EW_old = deepcopy(self.EW)
            else:
                self.nprop_pdE += 1
                pacc = np.exp( -self.beta*dEtot )
                if( np.random.uniform(0,1,1) > pacc ): # do not accept and revert back to original matrix
                    self.X = deepcopy(Sold)
                    self._S_to_K()
                    # update some arrays
                    if ( not self.fixed_pi ):
                        self._K_to_mu()
                    self.EQ_old = deepcopy(EQold)
                    self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                else:
                    self.nacc_pdE += 1 # keeping track of the accepted moves
                    # arrays should be current, just update EW
                    self.EW_old = deepcopy(self.EW)
            
            step_ctr += step_len




    def sample(self, n_step, T_init = None, X_init = None, mu_init = None, EQ_CG = None, EQ_AA = None, F_fun = None, F_CG = None, beta = None, lamb = None, fixed_pi = False):
        """
        Runs n_step successful! Metzner (or fixed-pi Metzner-type) sampling steps and returns a new transition matrix.

        Parameters:
        -----------
        n_step : int
            number of Metzner-type sampling steps.
        K_init : ndarray (n,n)
            initial rate matrix.
        S_init: ndarray (n,n)
            initial symm rate matrix.  If given along with mu, the initial K will be inferred directly.
        mu_init: ndarray (n)
            initial stationary distribution. 
        EQ_CG: float
            the MSM energy of the best CG model (i.e., the mle)
        EQ_AA: float
            the MSM energy of the AA model (mle)
        F_fun: func 
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
            The rate matrix after n_step sampling steps
        self.X: ndarray (n,n)
            The symm rate matrix after n_step sampling steps
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
        if ( (X_init is not None) and (mu_init is not None) ):
            self.X = deepcopy(X_init)
            self.mu = deepcopy(mu_init)
            self._S_to_K()
        elif (T_init is not None):
            self.T = deepcopy(T_init)
            self._K_to_mu()
	    self._K_to_S()
	else:
            raise ValueError('No valid starting matrix! Check your input')
        # symmetric?
        if not np.allclose(self.X, self.X.T):
            raise ValueError('Initial symmetric rate matrix is not symmetric.')

        # init the arrays 
        if (F_fun is not None):
            self._EW(F_fun)
            self.EW_old = deepcopy(self.EW)
        else: raise ValueError('F_fun is None.  Need to pass in a constraint, even with unbiased sampling!')
        self._EQ()
        self.EQ_old = deepcopy(self.EQ)

        # adjust the changeable element list for fixed-pi; constraints are the same for free and fixed pi
        #if (self.fixed_pi is True):
        #    self.chng_list = np.delete(self.chng_list, np.where( self.chng_list[0][:] <= self.chng_list[1][:] ) ,axis=1)
         
        # do the MC! 
        if ( self.MC_type == self.MC_types[0] ): # if MC_type == 'fixed_step'
            self._update_Metzner_MC_fixedstep(n_step, F_fun)
        elif ( self.MC_type == self.MC_types[1] ): # if MC_type == 'corr_move' 
            self._update_Metzner_MC_fixedstep_corrmove(n_step, F_fun)
        else:
            raise ValueError('MC_type not supported!  Check initialization of the sampler.')


def main():
    """
    This is a test function

    :return:
    """

#if __name__ == "__main__":
#    main()
