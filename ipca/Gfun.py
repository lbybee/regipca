from numpy.linalg import multi_dot
import numpy as np


def Gfun(Gamma, Factors, regconst, beta, y, tind, T, TN, L, K):
    """
    Evaluates Target

    Gamma : numpy array
        L x K
    Factors : numpy array
        K x T
    beta : numpy array
        (T * N) x K
        panel of estimated betas (X Gamma)
    y : numpy array
        (T * N) x 1
        panel of returns
    tind : numpy array
        N + 1 vector indicating the start/stop indices of each sub-panel
    T : scalar
        number of time periods
    TN : scalar
        number of firm-months
    L : scalar
        number of characteristics
    K : scalar
        number of factors
    """

    # build regconst arr if not provided
    if np.isscalar(regconst):
        regconst_arr = np.array([regconst] * L)
    else:
        regconst_arr = regconst

    # init params/temporary variables
    ypred = np.empty(TN, order="F")

    for t in range(T):

        # start/stop indices for firms in current time period
        ix_mn = tind[t]
        ix_mx = tind[t+1]
        ypred[ix_mn:ix_mx] = np.squeeze(beta[ix_mn:ix_mx,:].dot(Factors[:,t]))

    #end for
    SOSe = np.square(np.linalg.norm(ypred - y)) / 2.

    if regconst_arr[0] > 0:

        PenaltyTerm = 0
        for l in range(L):
            add = regconst_arr[l] * np.sqrt(Gamma[l,:].T.dot(Gamma[l,:]))
            PenaltyTerm += add

        G = (SOSe + TN * PenaltyTerm + np.square(np.linalg.norm(Factors)))

    else:
        G = SOSe

    return G
