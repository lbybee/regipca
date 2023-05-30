from sklearn.linear_model import Ridge
from scipy.linalg.lapack import dsysv
from proxcd_lasso import lasso
from datetime import datetime
from proxcd import glasso
from .Gfun import Gfun
from kron import kront
import scipy.linalg as sla
import numpy as np


def ALS(Gamma, Factors, X, y, XX, Xy, tind, T, N, TN, L, Kobs, K,
        max_iter=1000, iter_tol=1e-5, regconst=0., regconst_weight=None,
        norm=True, reg_method="glasso", silent=False, **kwargs):
    """fits ALS using group lasso

    Gamma : numpy array
        L x K
    Factors : numpy array
        K x T
    X : numpy array
        TN x L
        panel of characteristics
    y : numpy array
        TN x 1
        panel of returns
    XX : numpy array
        L x L
        precomputed X X inner product
    Xy : numpy array
        L x 1
        precomputed X y inner product
    tind : numpy array
        N + 1 vector indicating the start/stop indices of each date panel
    T : scalar
        number of time periods
    N : scalar
        number of firms
    TN : scalar
        number of firm-months
    L : scalar
        number of characteristics
    Kobs : scalar
        number of factors including observed factors (plus alpha)
    K : scalar
        number of factors
    max_iter : scalar
        maximum number of iterations for ALS
    iter_tol : scalar
        target tolerance below which we stop ALS
    regconst : scalar
        this corresponds to the regularizing constant
    regconst_weight : None or numpy array
        vector of weights to apply to columns of gamma
    norm : bool
        whether to normalize the resulting estimates
    method : str
        label for method used for Gamma estimation
    silent : bool
        indicator for whether to silence state reporting
    """

    # TODO replace OLS step with faster grad
    # TODO replace l2reg approach with full elastic net

    # set reg method
    if reg_method == "glasso":
        lfn = glasso
    elif reg_method == "lasso":
        lfn = lasso
    else:
        raise ValueError("Unknown reg_method: %s" % reg_method)


    # assemble regconst array
    if regconst_weight is not None:
        regconst_arr = regconst_weight * regconst
    else:
        regconst_arr = np.array([regconst] * L)

    # init params/temporary variables
    Xs = int((L * (L - 1)) / 2. + L)
    fs = int((Kobs * (Kobs - 1)) / 2. + Kobs)
    XFXF = np.zeros((L * Kobs, L * Kobs), order="F")
    w = np.zeros(L * Kobs, order="F")
    factordot = np.zeros((Kobs, Kobs), order="F")
    ypred = np.zeros(TN, order="F")
    Gamma_n = np.zeros(Gamma.shape, order="F")
    Factors_n = np.zeros(Factors.shape, order="F")
    if Kobs > K:
        Factors_n[K:,:] = Factors[K:,:]
    LK = L * Kobs

    # init starting value for beta and target
    beta = np.dot(X, Gamma)
    target_old = Gfun(Gamma, Factors, regconst_arr, beta, y, tind,
                      T, TN, L, Kobs)

    # init trace list
    trace = [target_old]

    for i in range(max_iter):

        # init runtime
        t0 = datetime.now()

        # init temp arrays
        XFXFvec = np.zeros((Xs * fs))
        XFy = np.zeros(L * Kobs)

        # build Xint/factors
        for t in range(T):


            # start/stop indices for firms in current time period
            ix_mn = tind[t]
            ix_mx = tind[t+1]
            Nt = ix_mx - ix_mn

            # build new factors
            if Kobs > K:
                resid = (y[ix_mn:ix_mx] -
                         beta[ix_mn:ix_mx,K:].dot(Factors_n[K:,t]))
            else:
                resid = y[ix_mn:ix_mx]

            if regconst > 0:
                bb = beta[ix_mn:ix_mx,:K].T.dot(beta[ix_mn:ix_mx,:K])
                bb += 2. * np.eye(K)
                br = beta[ix_mn:ix_mx,:K].T.dot(resid)
                Factors_n[:K,t] = np.linalg.inv(bb).dot(br)
            else:
                Factors_n[:K,t] = sla.lstsq(beta[ix_mn:ix_mx,:K], resid)[0]

            # XFy
            Xyt = Xy[:,t]
            XFy += (Xyt[:,None] * Factors_n[None,:,t]).reshape(L * Kobs)

        target_n = Gfun(Gamma, Factors_n, regconst_arr, beta, y, tind,
                        T, TN, L, Kobs)
        ftarget = target_n - target_old

        # construct XFXF
        kront(XX, Factors_n, XFXFvec, XFXF, factordot, T, L, Kobs)

        # prep w (proximal gradient descent coefs)
        w[:] = Gamma.T.reshape(Kobs * L)

        # get Gfun before fitting glasso
        target_p = Gfun(Gamma, Factors_n, regconst_arr, beta, y, tind,
                        T, TN, L, Kobs)

        # fit base group lasso
        if regconst > 0:

            # calculate current gradient
            grad = (XFXF.dot(w) - XFy) / TN

            # calculate step size
            stp_size = [np.linalg.eigvalsh(XFXF[l*Kobs:(l+1)*Kobs,
                                                l*Kobs:(l+1)*Kobs])[-1] / TN
                        for l in range(L)]
            stp_size = np.array(stp_size) * (1 + 10e-6)
            w = lfn(w, XFXF / TN, XFy / TN, grad, stp_size, regconst_arr,
                    L, Kobs, **kwargs)

        # fit unregularized
        else:
            lult, piv, w, _ = dsysv(XFXF, XFy, lower=1)

        # reshape w into Gamma
        Gamma_n[:,:] = np.array(w).reshape((L, Kobs))

        # build target for current iteration + update beta
        beta = np.dot(X, Gamma_n)
        target = Gfun(Gamma_n, Factors_n, regconst_arr, beta, y, tind,
                      T, TN, L, Kobs)
        gtarget = target - target_p

        # update trace
        trace.append(target)

        # set new arrays as old arrays
        Gamma = Gamma_n
        Factors = Factors_n

        # target diff
        diff = np.abs(target - target_old) / target_old

        # TODO remove nz
        if not silent:
            print("iteration:", i, "target:", target, "diff:", diff,
                  "nz:", np.sum(Gamma != 0), "regconst:", regconst,
                  "fnorm", np.linalg.norm(Factors),
                  "gnorm", np.linalg.norm(Gamma),
                  "gdiff", gtarget, "fdiff", ftarget,
                  "runtime:", datetime.now() - t0)

        # exit loop if target below threshold
        if diff <= iter_tol:
            break
        target_old = target

    if norm or regconst == 0:

        # apply normalization
        R1 = sla.cholesky(Gamma[:,:K].T.dot(Gamma[:,:K]))
        R2 = np.linalg.eig(R1.dot(Factors[:K,:].dot(Factors[:K,:].T).dot(R1.T)))[1]
        Gamma[:,:K] = sla.lstsq(Gamma[:,:K].T, R1.T)[0].dot(R2)
        Factors[:K,:] = sla.lstsq(R2, R1.dot(Factors[:K,:]))[0]

        # Enforce sign convention for Gamma_Beta and F_New
        sg = np.sign(np.mean(Factors[:K,:], axis=1)).reshape((-1, 1))
        sg[sg == 0] = 1
        Gamma[:,:K] = np.multiply(Gamma[:,:K], sg.T)
        Factors[:K,:] = np.multiply(Factors[:K,:], sg)

    return Gamma, Factors, np.array(trace)
