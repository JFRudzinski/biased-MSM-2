# JFR - adapted from stallone

import numpy as np

'''
 * @author noe
'''

#class TransitionMatrixSamplingTools:

#    '''
#    Checks whether the given element is still within [0,1] or else puts it back to that
#    value.
#    '''

def ensureValidElement(T, i, j):

    if (T[i,j] < 0):
        T[i,j] = 0.0

    if (T[i,j] > 1.0):
        T[i,j] = 1.0

def isElementIn01(T, i, j):

    if (T[i,j] < 0):
        return False

    if (T[i, j] > 1):
        return False

    return True

def isRowIn01(T, i):

    for j in range (0, T.shape[1]):

        if (T[i,j] < 0):
            return False

        if (T[i,j] > 1):
            return False

    return True

'''
 * @param mu invariant density
 * @return
'''
def computeDetailedBalanceError(T, mu):

    err = 0.0
    for i in range (0, T.shape[0]):
        for j in range (0, T.shape[1]):
            err += np.abs(mu[i]*T[i,j] - mu[j]*T[j,i])

    return err

'''
Makes sure that the row still sums up to 1.
'''
def ensureValidRow(T, i):

    rowsum = np.sum(T, axis=1)
    T /= rowsum

    return T


