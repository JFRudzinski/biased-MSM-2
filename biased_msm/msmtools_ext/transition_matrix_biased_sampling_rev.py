# JFR - This file was adapted from bhmm.transition_matrix_sampling_rev.py and also from stallone
__author__ = 'noe'

import numpy as np
import math
from copy import copy, deepcopy
from msmtools.analysis.dense.stationary_vector import stationary_distribution_from_backward_iteration
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

    def __init__(self, _C, _prior, _X, _MC_type, _lT): 
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
        self.MC_types = [ 'fixed_step', 'corr_move' ]
        self.MC_type = deepcopy(_MC_type)
        self.lT = _lT
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


    def _X_to_T(self):
        self.T =  self.X / np.sum(self.X, dtype=float, axis=1, keepdims=True)
    def _X_to_mu(self):
        self.mu = np.sum(self.X, dtype=float, axis=1, keepdims=True)

    def _T_to_X(self):
        mu = stationary_distribution_from_backward_iteration(self.T)
        self.X = np.dot(np.diag(mu), self.T)
        self.X /= np.sum(self.X, dtype=float)

    def _update_arrays(self, flag_acc):
        if ( flag_acc ):
            # stat dist
            self._X_to_mu()
            # Energies
            self.EW_old = deepcopy(self.EW)
            self.EQ_old = deepcopy(self.EQ)
        else:
            # Energies
            self.EW = deepcopy(self.EW_old)
            self.EQ = deepcopy(self.EQ_old)


    def _logprob_T(self):
        assert np.all(self.T >= 0)
        return np.sum(np.multiply(self.C, np.ma.log(self.T)), dtype=float) # avoid 0 elements in the log arg with np.ma.log 

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
        if (self.fixed_pi is True):
            frac = 0.5 # only allow changes by half of the current value
            a0 = -frac*self.X[i,j]
            b0 = frac*min(self.X[i,i],self.X[j,j])
            # Now, add a limit for individual T values: Tij <= lT
            # first, Tij,Tji <= lT
            b1 = (self.lT * self.mu[i]) - self.X[i,j]
            b2 = (self.lT * self.mu[j]) - self.X[j,i]
            # now, Tii,Tjj <= lT
            a1 = self.X[i,i] - self.lT*self.mu[i]
            a2 = self.X[j,j] - self.lT*self.mu[j]
            # nb - Til(l!=j),Tjl(l!=i) <= lT auto satisfied.
            # now, get the bounds
            a = max(a0,a1,a2)
            b = min(b0,b1,b2)
        else:
            sc = np.sum(self.X, dtype=float)
            kmin = 0.999 * sc # Similar to the Metzner paper, rescaled for arbitrary X normalization
            kpls = 1.001 * sc
            if (i == j):
                a0 = max(-0.5*self.X[i,j], kmin - sc)
                b0 = min( 0.5*self.X[i,j], kpls - sc)
                # Now, add a limit for individual T values: Tij <= lT
                # first, Tij,Tji <= lT
                if (np.abs(self.lT - 1.0) >= 1e-6):
                    b1 = (-self.X[i,j] + (self.lT*np.sum(self.X,axis=1)[i]) ) / (1.0-self.lT)
                else:
                    b1 = b0
                # now, Til(l!=j),Tjl(l!=i) <= lT
                a1 = (self.X[i,:] - (self.lT*np.sum(self.X,axis=1)[i]) ) / (self.lT)
                a1 = np.delete(a1,[j])
                a1 = np.max(a1)
                # now, get the bounds
                a = max(a0,a1)
                b = min(b0,b1)
            else: 
                # the normal Metzner constraints plus slight stricter adjustments
                a0 = max(-0.5*self.X[i,j], 0.5*(kmin - sc)) # JFR - adjusted a0, b0 so that Xij can change by at most half its current value
                b0 = min( 0.5*self.X[i,j], 0.5*(kpls - sc)) 
                # Now, add a limit for individual T values: Tij <= lT
                # first, Tij,Tji <= lT
                if (np.abs(self.lT - 1.0) >= 1e-6):
                    b1 = (-self.X[i,j] + (self.lT*np.sum(self.X,axis=1)[i]) ) / (1.0-self.lT)
                    b2 = (-self.X[j,i] + (self.lT*np.sum(self.X,axis=1)[j]) ) / (1.0-self.lT)
                else:
                    b1 = b0
                    b2 = b0
                # now, Til(l!=j),Tjl(l!=i) <= lT
                a1 = (self.X[i,:] - (self.lT*np.sum(self.X,axis=1)[i]) ) / (self.lT)
                a1 = np.delete(a1,[j])
                a1 = np.max(a1)
                a2 = (self.X[j,:] - (self.lT*np.sum(self.X,axis=1)[j]) ) / (self.lT)
                a2 = np.delete(a2,[i])
                a2 = np.max(a2)
                # now, get the bounds
                a = max(a0,a1,a2)
                b = min(b0,b1,b2)
           
        # check the bounds
        if ( a > 0 or b < 0 ):
            raise ValueError('Something is wrong with the bounds in _step_Metzner_MC()!')
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
        shuffle(shuff_chng_list)

        for ind in range(n_MC): # Loop over the changeable elements

            # get the indices
            i = shuff_chng_list[0][ind]
            j = shuff_chng_list[1][ind]
            # save the old state
            saveij = deepcopy(self.X[i,j])
            saveii = deepcopy(self.X[i,i])
            savejj = deepcopy(self.X[j,j])
            # make the MC move
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
                    self._X_to_T()
                    self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                else:
                    self.nacc_pdE += 1 # keeping track of the accepted moves
                    self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW


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
            Xold = deepcopy(self.X)
            EQold = deepcopy(self.EQ)

            for ind in range(step_ctr, step_ctr+step_len): # Loop over the changeable elements

                # get the indices
                i = shuff_chng_list[0][ind]
                j = shuff_chng_list[1][ind]
                # save the old state
                saveij = deepcopy(self.X[i,j])
                saveii = deepcopy(self.X[i,i])
                savejj = deepcopy(self.X[j,j])
                # make the MC move
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
                        self._X_to_T()
                        self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                    else:
                        self._update_arrays( True ) # mu -> mu(T), EQ_old -> EQ, EW_old -> EW    
            
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
                    self.X = deepcopy(Xold)
                    self._X_to_T()
                    # update some arrays
                    self._X_to_mu()
                    self.EQ_old = deepcopy(EQold)
                    self._update_arrays( False ) # EQ -> EQ_old, EW -> EW_old
                else:
                    self.nacc_pdE += 1 # keeping track of the accepted moves
                    # arrays should be current, just update EW
                    self.EW_old = deepcopy(self.EW)
            
            step_ctr += step_len




    def sample(self, n_step, T_init = None, X_init = None, EQ_CG = None, EQ_AA = None, F_fun = None, F_CG = None, beta = None, lamb = None, fixed_pi = False):
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
        if (F_fun is not None):
            self._EW(F_fun)
            self.EW_old = deepcopy(self.EW)
        else: raise ValueError('F_fun is None.  Need to pass in a constraint, even with unbiased sampling!')
        self._EQ()
        self.EQ_old = deepcopy(self.EQ)

        # adjust the changeable element list for fixed-pi
        if (self.fixed_pi is True):
            self.chng_list = np.delete(self.chng_list, np.where( self.chng_list[0][:] == self.chng_list[1][:] ) ,axis=1)
         
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
