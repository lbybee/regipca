#from labbot.decorators import profiledec
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from scipy.linalg.lapack import dsysv
from datetime import datetime
from partinner import innert
from .ALS import ALS
import scipy.linalg as sla
import pandas as pd
import numpy as np
import scipy as sp
import os


class InstrumentedPCA(BaseEstimator):
    """
    This class implements the IPCA algorithm.

    It includes both the baseline implementation
    from Kelly, Pruitt, Su (2017) and the regularized implementation
    from Bybee, Kelly, Su (2021).

    Parameters
    ----------

    n_factors : int, default=1
        The total number of factors to estimate. Note, the number of
        estimated factors is automatically reduced by the number of
        pre-specified factors. For example, if n_factors = 2 and one
        pre-specified factor is passed, then InstrumentedPCA will estimate
        one factor estimated in addition to the pre-specified factor.

    intercept : boolean, default=False
        Determines whether the model is estimated with or without an intercept

    max_iter : int, default=10000
        Maximum number of alternating least squares updates before the
        estimation is stopped

    iter_tol : float, default=10e-6
        Tolerance threshold for stopping the alternating least squares
        procedure

    regconst : scalar
        Regularizing constant for Gamma estimation.  If this is set to
        zero then the estimation defaults to non-regularized.

    groupl : scalar
        whether the regularized estimation is done with group lasso
        or ridge regression
    """

    def __init__(self, n_factors=1, intercept=False, max_iter=10000,
                 iter_tol=10e-6, regconst=0., groupl=1):

        # paranoid parameter checking to make it easier for users to know when
        # they have gone awry and to make it safe to assume some variables can
        # only have certain settings
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if not isinstance(intercept, bool) and not isinstance(intercept, int):
            raise NotImplementedError('intercept must be  boolean or int')
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            raise ValueError('Iteration tolerance must be smaller than 1.')
        if regconst < 0.:
            raise ValueError("regconst must be greater than or equal to 0")

        self.cached_PSF = False

        # Save parameters to the object
        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)


    def fit(self, X=None, y=None, indices=None, PSF=None, Gamma=None,
            Factors=None, date_ind=1, **kwargs):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------
        X :  numpy array or pandas DataFrame or None
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.

            If given as a DataFrame, we assume that it contains a MutliIndex
            mapping to each entity-time pair

        y : numpy array or pandas Series or None
            dependent variable where indices correspond to those in X

            If given as a Series, we assume that it contains a MutliIndex
            mapping to each entity-time pair

        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:

            - Column 1: entity id (i)
            - Column 2: time index (t)

            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.

        PSF : numpy array, optional
            Set of pre-specified factors as matrix of dimension (M, T)

        Gamma : numpy array, optional
            If provided, starting values for Gamma (see Notes)

        Factors : numpy array
            If provided, starting values for Factors (see Notes)

        date_ind : scalar
            which level in the indices correspond to the date
            defaults to the second (index 1)

        Returns
        -------
        self

        Notes
        -----
        - Updates InstrumentedPCA instances to include param estimates:

        Gamma : numpy array
            Array with dimensions (L, n_factors) containing the
            mapping between characteristics and factors loadings. If there
            are M many pre-specified factors in the model then the
            matrix returned is of dimension (L, (n_factors+M)).
            If an intercept is included in the model, its loadings are
            returned in the last column of Gamma.

        Factors : numpy array
            Array with dimensions (n_factors, T) containing the estimated
            factors. If pre-specified factors were passed the returned
            array is of dimension ((n_factors - M), T),
            corresponding to the n_factors - M many factors estimated on
            top of the pre-specified ones.

        - The current fit method assumes that the chars are sorted by date
        """

        # TODO handle observed factors

        # prep inputs if new data is provided
        if X is not None:
            self = self.cache_input(X, y, indices, date_ind=date_ind)
        self = self.cache_params(Gamma, Factors, PSF)

        # load inputs from self
        X, y = self.X, self.y
        TN, L = self.X.shape
        T = self.metad["T"]
        N = self.metad["N"]
        Kobs = self.n_factors
        K = self.n_factors_est
        Gamma, Factors = self.Gamma, self.Factors
        XX, Xy = self.XX, self.Xy
        tind = self.tind

        # fit ALS
        Gamma, Factors, trace = ALS(Gamma, Factors,
                                    X, y, XX, Xy, tind,
                                    T, N, TN, L, Kobs, K,
                                    max_iter=self.max_iter,
                                    iter_tol=self.iter_tol,
                                    regconst=self.regconst,
                                    **kwargs)

        self.Gamma, self.Factors = Gamma, Factors
        self.trace = trace

        return self


    def fit_path(self, X=None, y=None, indices=None, PSF=None,
                 label_ind=False, date_ind=1, is_pred_type=None,
                 oos_pred_type=None,  oos_X=None, oos_y=None, oos_indices=None,
                 regconst_stps=101, regconst_min=None, regconst_max=None,
                 regconst_path=None, update_est=True, fail_stp=False,
                 stp_dec=True, out_dir=None, drop_unreg=False,
                 r2=True, **kwargs):
        """fits entire path of regularized IPCA models

        Parameters
        ----------
        regconst_stps : int
            number of grid steps for regularizing constant
        regconst_path : iterable or None
            if provided, use this vector as the path of reg consts
        update_est : bool
            whether to use the previously estimated values
            at each grid step
        fail_stp : bool
            whether to raise or supress linalg errors when
            thresh(L) <= K.
        stp_dec : bool
            whether the series of regularizing constants should
            be increasing or decreasing
        is_pred_type : list or None
            labels for score method used for in-sample reg selection
        oos_pred_type : list or None
            labels for score method used for out-of-sample reg selection
        oos_X :  numpy array or pandas DataFrame or None
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.

            If given as a DataFrame, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        oos_y : numpy array or pandas Series or None
            dependent variable where indices correspond to those in X

            If given as a Series, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        oos_indices : numpy array, optional
            array containing the panel indices for test sample.
            Should consist of two columns:

            - Column 1: entity id (i)
            - Column 2: time index (t)

            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        out_dir : str or None
            if provided, write the intermediate paths to an output dir
        drop_unreg : bool
            indicator for whether to drop unregularized case from path
        r2 : bool
            indicator for whether to yield R2 or SSE for path

        Returns
        -------
        self

        Including:
            r2 : numpy array
                array containing the predictive R2s for each reg cnst
            pred : numpy array
                matrix where rows correspond to reg cnst and columns
                correspond to observations
            Gamma : numpy array
                tensor of Gammas where rows correspond to reg cnst
            Factors : numpy array
                tensor of Factors where rows correspond to reg cnst
        """

        if oos_pred_type is None:
            oos_pred_type = []
        if is_pred_type is None:
            is_pred_type = []

        # prep cached values
        if X is not None:
            self = self.cache_input(X, y, indices, date_ind=date_ind)
        self = self.cache_params(None, None, PSF)

        # build regconst path
        if regconst_path is None:
            self = self.build_regconst_path(regconst_stps=regconst_stps,
                                            regconst_min=regconst_min,
                                            regconst_max=regconst_max,
                                            stp_dec=stp_dec,
                                            drop_unreg=drop_unreg)
        else:
            self.regconst_path = regconst_path

        # fit IPCA over regconst path
        Gamma_path = []
        Factors_path = []
        trace_path = []
        r2_path = []
        Gamma = self.Gamma
        Factors = self.Factors

        for regconst in self.regconst_path:

            self.regconst = regconst
            try:
                self.fit(Gamma=Gamma, Factors=Factors, PSF=PSF, **kwargs)
            except np.linalg.LinAlgError as e:
                if fail_stp:
                    raise e
                else:
                    print(e)
                    break

            if update_est:
                Gamma = self.Gamma
                Factors = self.Factors

            Gamma_path.append(self.Gamma)
            Factors_path.append(self.Factors)
            trace_path.append(self.trace)
            nz_count = np.sum(self.Gamma != 0)
            res_dict = {}

            for pt in is_pred_type:
                res_dict[pt] = self.score(pred_type=pt,
                                          r2=r2)
            for pt in oos_pred_type:
                res_dict[pt] = self.score(X=oos_X, y=oos_y,
                                          indices=oos_indices,
                                          pred_type=pt,
                                          r2=r2)
            res_dict["nz"] = nz_count
            r2_path.append(res_dict)

        self.Gamma_path = np.array(Gamma_path)
        self.Factors_path = np.array(Factors_path)
        self.trace_path = np.array(trace_path)
        self.r2_path = pd.DataFrame(r2_path)
        self.r2_path.index = self.regconst_path[:self.r2_path.shape[0]]
        self.r2_path.index.name = "regconst"

        return self


    def fit_path_pred(self, X=None, y=None, indices=None, PSF=None,
                      label_ind=False, date_ind=1, is_pred_type=None,
                      oos_pred_type=None,  oos_X=None, oos_y=None,
                      oos_indices=None, regconst_stps=101, regconst_min=None,
                      regconst_max=None, regconst_path=None, update_est=True,
                      fail_stp=False, stp_dec=True, out_dir=None,
                      drop_unreg=False, **kwargs):
        """fits entire path of regularized IPCA models yielding pred values

        Parameters
        ----------
        regconst_stps : int
            number of grid steps for regularizing constant
        regconst_path : iterable or None
            if provided, use this vector as the path of reg consts
        update_est : bool
            whether to use the previously estimated values
            at each grid step
        fail_stp : bool
            whether to raise or supress linalg errors when
            thresh(L) <= K.
        stp_dec : bool
            whether the series of regularizing constants should
            be increasing or decreasing
        is_pred_type : list or None
            labels for score method used for in-sample reg selection
        oos_pred_type : list or None
            labels for score method used for out-of-sample reg selection
        oos_X :  numpy array or pandas DataFrame or None
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.

            If given as a DataFrame, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        oos_y : numpy array or pandas Series or None
            dependent variable where indices correspond to those in X

            If given as a Series, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        oos_indices : numpy array, optional
            array containing the panel indices for test sample.
            Should consist of two columns:

            - Column 1: entity id (i)
            - Column 2: time index (t)

            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        out_dir : str or None
            if provided, write the intermediate paths to an output dir
        drop_unreg : bool
            indicator for whether to drop unregularized case from path

        Returns
        -------
        self

        Including:
            r2 : numpy array
                array containing the predictive R2s for each reg cnst
            pred : numpy array
                matrix where rows correspond to reg cnst and columns
                correspond to observations
            Gamma : numpy array
                tensor of Gammas where rows correspond to reg cnst
            Factors : numpy array
                tensor of Factors where rows correspond to reg cnst
        """

        if oos_pred_type is None:
            oos_pred_type = []
        if is_pred_type is None:
            is_pred_type = []

        # prep cached values
        if X is not None:
            self = self.cache_input(X, y, indices, date_ind=date_ind)
        self = self.cache_params(None, None, PSF)

        # build regconst path
        if regconst_path is None:
            self = self.build_regconst_path(regconst_stps=regconst_stps,
                                            regconst_min=regconst_min,
                                            regconst_max=regconst_max,
                                            stp_dec=stp_dec,
                                            drop_unreg=drop_unreg)
        else:
            self.regconst_path = regconst_path

        # fit IPCA over regconst path
        Gamma_path = []
        Factors_path = []
        trace_path = []
        r2_path = []
        pred_path = []
        nz_path = []
        Gamma = self.Gamma
        Factors = self.Factors

        for regconst in self.regconst_path:

            self.regconst = regconst
            try:
                self.fit(Gamma=Gamma, Factors=Factors, PSF=PSF, **kwargs)
            except np.linalg.LinAlgError as e:
                if fail_stp:
                    raise e
                else:
                    print(e)
                    break

            if update_est:
                Gamma = self.Gamma
                Factors = self.Factors

            Gamma_path.append(self.Gamma)
            Factors_path.append(self.Factors)
            trace_path.append(self.trace)
            nz_count = np.sum(self.Gamma != 0)
            res_dict = {}

            for pt in is_pred_type:
                res_dict[pt] = self.predict(pred_type=pt,
                                            label_ind=True)
            for pt in oos_pred_type:
                res_dict[pt] = self.predict(X=oos_X, y=oos_y,
                                            indices=oos_indices,
                                            pred_type=pt,
                                            label_ind=True)
            pred_path.append(res_dict)
            nz_path.append(nz_count)

        self.Gamma_path = np.array(Gamma_path)
        self.Factors_path = np.array(Factors_path)
        self.trace_path = np.array(trace_path)

        pred_dict = {}
        for pt in is_pred_type:
            pred_dict[pt] = []
        for pt in oos_pred_type:
            pred_dict[pt] = []
        for p_d in pred_path:
            for pt in is_pred_type:
                pred_dict[pt].append(p_d[pt])
            for pt in oos_pred_type:
                pred_dict[pt].append(p_d[pt])
        yhat_null = self.predict(pred_type="null", label_ind=True)
        yhat_null_oos = self.predict(X=oos_X, y=oos_y,
                                     indices=oos_indices,
                                     pred_type="null_oos",
                                     label_ind=True)
        for pt in is_pred_type:
            df = pd.concat([yhat_null] + pred_dict[pt], axis=1)
            df.columns = (["null"] +
                          list(self.regconst_path[:len(pred_path)]))
            pred_dict[pt] = df
        for pt in oos_pred_type:
            df = pd.concat([yhat_null_oos] + pred_dict[pt], axis=1)
            df.columns = (["null"] +
                          list(self.regconst_path[:len(pred_path)]))
            pred_dict[pt] = df
        self.pred_dict = pred_dict
        nz_path = pd.Series(nz_path,
                            index=self.regconst_path[:len(pred_path)])
        nz_path.name = "nz"
        nz_path.index.name = "regconst"
        self.nz_path = nz_path

        return self


    def cache_input(self, X, y, indices=None, date_ind=1):
        """prepares the inputs for fitting IPCA

        Returns
        -------
        self

        Populated with necessary inputs for further fitting
        """

        # handle input
        X, y, indices, tind, metad = _prep_panel(X, y, indices, date_ind)
        N, L, T = metad["N"], metad["L"], metad["T"]

        # handle pre-caching of key inputs
        XX = np.empty(L * L * T, order="F")
        Xy = np.zeros(L * T, order="F")
        innert(X, y, tind, XX, Xy, T, L)
        Xy = Xy.reshape(T, L).T

        # store data
        self.X, self.y = X, y
        self.XX, self.Xy = XX, Xy
        self.indices, self.tind = indices, tind
        self.metad = metad

        return self


    def cache_params(self, Gamma, Factors, PSF):
        """cache the Factors/Gamma values"""

        T = self.metad["T"]

        if not self.cached_PSF:
            # Handle pre-specified factors
            if PSF is not None:
                if np.size(PSF, axis=1) != T:
                    raise ValueError("""Number of PSF observations must match
                                     number of unique dates""")

                if np.size(PSF, axis=0) == self.n_factors:
                    print("""Note: The number of factors (n_factors) to be
                          estimated matches the number of
                          pre-specified factors. No additional factors
                          will be estimated. To estimate additional
                          factors increase n_factors.""")
                self.n_factors_est = self.n_factors - PSF.shape[0]

            else:
                self.n_factors_est = self.n_factors

            # Treating intercept as if was a prespecified factor
            if self.intercept:
                self.n_factors = self.n_factors + 1
                if PSF is not None:
                    PSF = np.concatenate((PSF, np.ones((1, T)) / T), axis=0)
                else:
                    PSF = np.ones((1, T)) / T
                self.n_factors_est = self.n_factors - PSF.shape[0]

            self.PSF = PSF
            self.cached_PSF = True

        elif not hasattr(self, "n_factors_est"):
            self.n_factors_est = self.n_factors

        # build starting values
        # TODO what happens when T < K?
        if Gamma is None or Factors is None:
            Gamma, s, v = np.linalg.svd(self.Xy)
            Gamma = Gamma[:, :self.n_factors]
            if self.n_factors_est > 0:
                s = s[:self.n_factors_est]
                v = v[:self.n_factors_est, :]
                Factors = np.diag(s).dot(v)
                if PSF is not None:
                    Factors = np.vstack((Factors, PSF))
            else:
                Factors = PSF

        self.Factors, self.Gamma = Factors, Gamma

        return self


    def build_regconst_path(self, regconst_stps=101, regconst_min=None,
                            regconst_max=None, stp_dec=True, drop_unreg=False):
        """build optimal path of reg constants"""

        # prep min regconst
        if regconst_min is None:
            regconst_min = 0

        # get optimal regconst values given data
        if regconst_max is None:
            XXyyt = np.zeros(self.metad["L"])
            for t in range(self.metad["T"]):
                ix_mn = self.tind[t]
                ix_mx = self.tind[t+1]
                XXyyt += np.square(np.dot(self.X[ix_mn:ix_mx,:].T,
                                          self.y[ix_mn:ix_mx]))
#            XXyyt = 2 * np.sqrt(XXyyt) / (self.X.shape[0])
            XXyyt = np.sqrt(XXyyt) / (self.X.shape[0])
            regconst_max = np.max(XXyyt)

        ind = np.logspace(0, 1, regconst_stps)
        ind -= 1
        ind /= ind.max()
        ind *= regconst_max
        ind += regconst_min

        if drop_unreg:
            ind = ind[1:]

        if stp_dec:
            self.regconst_path = np.flip(ind)
        else:
            self.regconst_path = ind

        return self


    def get_factors(self, label_ind=False):
        """returns a tuple containing Gamma and Factors

        Parameters
        ----------
        label_ind : bool
            if provided we return the factors as pandas DataFrames with
            index info applied

        Returns
        -------
        tuple
            containing Gamma and Factors
        """

        Gamma, Factors = self.Gamma.copy(), self.Factors.copy()
        if label_ind:
            Gamma = pd.DataFrame(Gamma, index=self.metad["chars"])
            Factors = pd.DataFrame(Factors, columns=self.metad["dates"])

        return Gamma, Factors


    def predict(self, X=None, y=None, indices=None,
                pred_type="panel_total", label_ind=False, date_ind=1):
        """wrapper around different data type predict methods

        Parameters
        ----------
        X :  numpy array or pandas DataFrame, optional
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.

            If given as a DataFrame, we assume that it contains a mutliindex
            mapping to each entity-time pair

            If None we use the values associated with the current model

        y : numpy array or pandas Series, optional
            dependent variable where indices correspond to those in X

            If given as a Series, we assume that it contains a mutliindex
            mapping to each entity-time pair

        indices : pandas MultiIndex, optional
            array containinng the panel indices

            If None we use the values associated with the current model

        pred_type : str
            label for type used for prediction, one of the following:

        label_ind : bool
            whether to apply the indices to fitted values and return
            pandas Series

        date_ind : scalar
            which level in the indices correspond to the date
            defaults to the second (index 1)

        Returns
        -------
        numpy array or pandas DataFrame/Series
            The exact value returned depends on two things:

            1. The pred_type
                a. If panel or oos pred_type, this will be a series of
                values for the panel ys

                b. If portfolio pred_type, this will a matrix
                of predicted char formed portfolio Qs

            2. label_ind
                If label_ind is True, we return pandas variants of the
                predicted values.  If not, we return the underlying
                numpy arrays.
        """

        if pred_type == "panel_total":
            ypred = self.predict_panel_total(label_ind)
        elif pred_type == "panel_predictive":
            ypred = self.predict_panel_predictive(label_ind)
        elif pred_type == "panel_total_oos":
            ypred = self.predict_panel_total_oos(X, y, indices,
                                                 date_ind, label_ind)
        elif pred_type == "panel_predictive_oos":
            ypred = self.predict_panel_predictive_oos(X, y, indices,
                                                      date_ind, label_ind)
        elif pred_type == "panel_predictive_oos_unreg":
            ypred = self.predict_panel_predictive_oos_unreg(X, y, indices,
                                                            date_ind,
                                                            label_ind)
        elif pred_type == "panel_predictive_prevoos":
            ypred = self.predict_panel_predictive_prevoos(X, y, indices,
                                                          date_ind, label_ind)
        elif pred_type == "null":
            ypred = self.predict_null(label_ind)
        elif pred_type == "null_oos":
            ypred = self.predict_null_oos(X, y, indices, date_ind, label_ind)
        else:
            raise ValueError("Unsupported pred_type: %s" % pred_type)

        return ypred


    def predict_panel_total(self, label_ind):
        """
        Predicts fitted values for fitted regressor + panel data

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        X, metad = self.X, self.metad
        indices, tind = self.indices, self.tind
        N, L, T = metad["N"], metad["L"], metad["T"]

        if T != self.Factors.shape[1]:
            raise ValueError("If mean_factor isn't used date shape must\
                              align with Factors shape")
        else:
            ypred = np.full((X.shape[0]), np.nan)
            for t in range(T):
                ix_mn = tind[t]
                ix_mx = tind[t+1]
                ypred[ix_mn:ix_mx] = np.squeeze(X[ix_mn:ix_mx, :]\
                                                .dot(self.Gamma)\
                                                .dot(self.Factors[:, t]))

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_panel_predictive(self, label_ind):

        X, metad = self.X, self.metad
        indices, tind = self.indices, self.tind
        N, L, T = metad["N"], metad["L"], metad["T"]

        # generated fitted values using mean factors
        mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))
        ypred = np.squeeze(X.dot(self.Gamma).dot(mean_Factors))

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_panel_total_oos(self, X, y, indices, date_ind, label_ind):
        """
        Predicts time t+1:t+tau observation using out-of-sample design

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        inp = _prep_panel(X, y, indices, date_ind)
        X, y, indices, tind, metad = inp
        N, L, T = metad["N"], metad["L"], metad["T"]

        ypred_l = []
        for t in range(T):
            ix_mn = tind[t]
            ix_mx = tind[t+1]
            beta = X[ix_mn:ix_mx,:].dot(self.Gamma)
            try:
                Factor_OOS = sla.lstsq(beta, y[ix_mn:ix_mx])[0]
                ypred = beta.dot(Factor_OOS)
                ypred_l.append(ypred)
            except np.linalg.LinAlgError as e:
                print(e)
                ypred_l.append(np.array([np.nan] * (ix_mx - ix_mn)))
        ypred = np.hstack(ypred_l)

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_panel_predictive_oos(self, X, y, indices, date_ind, label_ind):
        """Predicts fitted values using factor means out-of-sample

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        inp = _prep_panel(X, y, indices, date_ind)
        X, y, indices, tind, metad = inp
        N, L, T = metad["N"], metad["L"], metad["T"]

        # generate OOS factors
        mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))
        ypred = np.squeeze(X.dot(self.Gamma).dot(mean_Factors))

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_panel_predictive_prevoos(self, X, y, indices, date_ind,
                                         label_ind):
        """Predicts fitted values using the most recent factor estimate

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        # TODO update factor construction
        inp = _prep_panel(X, y, indices, date_ind)
        X, y, indices, tind, metad = inp
        N, L, T = metad["N"], metad["L"], metad["T"]

        # generate OOS factors

        # starting values
        ypred_l = []
        ix_mn = self.tind[-2]
        ix_mx = self.tind[-1]
        beta = self.X[ix_mn:ix_mx,:].dot(self.Gamma)
        Numer = beta.T.dot(y[ix_mn:ix_mx])
        Denom = beta.T.dot(X[ix_mn:ix_mx]).dot(self.Gamma)
        try:
            ix_mn = tind[0]
            ix_mx = tind[1]
            Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))
            ypred = X[ix_mn:ix_mx,:].dot(self.Gamma).dot(Factor_OOS)[:,0]
            ypred_l.append(ypred)
        except np.linalg.LinAlgError as e:
            ypred_l.append(np.array([np.nan] * (ix_mx - ix_mn)))

        for t in range(T - 1):
            ix_mn = tind[t]
            ix_mx = tind[t+1]
            beta = X[ix_mn:ix_mx,:].dot(self.Gamma)
            Numer = beta.T.dot(y[ix_mn:ix_mx])
            Denom = beta.T.dot(X[ix_mn:ix_mx]).dot(self.Gamma)
            try:
                ix_mn = tind[t+1]
                ix_mx = tind[t+2]
                Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))
                ypred = X[ix_mn:ix_mx,:].dot(self.Gamma).dot(Factor_OOS)[:,0]
                ypred_l.append(ypred)
            except np.linalg.LinAlgError as e:
                ypred_l.append(np.array([np.nan] * (ix_mx - ix_mn)))
        ypred = np.hstack(ypred_l)

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_null(self, label_ind):
        """
        Yields original values in comparable form to pred

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        X, ypred, metad = self.X, self.y, self.metad
        indices, tind = self.indices, self.tind
        N, L, T = metad["N"], metad["L"], metad["T"]

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def predict_null_oos(self, X, y, indices, date_ind, label_ind):
        """
        Yields original values in comparable form to pred

        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        inp = _prep_panel(X, y, indices, date_ind)
        X, ypred, indices, tind, metad = inp
        N, L, T = metad["N"], metad["L"], metad["T"]

        if label_ind:
            ypred = pd.DataFrame(ypred, columns=["yhat"], index=indices)

        return ypred


    def score(self, y=None, r2=True, pred_type="panel_total",
              **kwargs):
        """generate R^2

        Parameters
        ----------
        test_y : pandas Series, numpy array or None
            y values used for comparison
        r2 : bool
            indicator for whether to yield R2 or SSE

        Returns
        -------
        r2 : scalar
            summary of model performance
        """

        if y is None:
            y = self.y


        if r2 and pred_type != "null" and pred_type != "null_oos":
            yhat = self.predict(y=y, pred_type=pred_type, **kwargs)
            score = 1 - np.nansum((yhat-y)**2)/np.nansum(y**2)
        elif r2 and (pred_type == "null" or pred_type == "null_oos"):
            raise ValueError("Can't calculate score for null pred_type")
        elif not r2 and pred_type != "null" and pred_type != "null_oos":
            yhat = self.predict(y=y, pred_type=pred_type, **kwargs)
            score = np.nansum((yhat-y)**2)
        elif not r2 and (pred_type == "null" or pred_type == "null_oos"):
            score = np.nansum(y**2)

        return score


    def BS_Walpha(self, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Bootstrap inference on the hypothesis Gamma_alpha = 0

        Parameters
        ----------

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        backend : optional
            Value is either 'loky' or 'multiprocessing'

        n_jobs  : integer
            Number of workers to be used. If -1, all available workers are
            used.

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        # TODO replace alpha with regconst
        # TODO replace Q/W
        # TODO update joblib imports

        if self.alpha > 0.:
            raise ValueError("Bootstrap currently not supported for\
                              regularized estimation.")

        if not self.intercept:
            raise ValueError('Need to fit model with intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "Q"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.metad["N"], self.metad["L"], self.metad["T"]

        # Compute Walpha
        Walpha = self.Gamma[:, -1].T.dot(self.Gamma[:, -1])

        # Compute residuals
        d = np.full((L, T), np.nan)

        for t in range(T):
            d[:, t] = self.Q[:, t]-self.W[:, :, t].dot(self.Gamma)\
                .dot(self.Factors[:, t])

        print("Starting Bootstrap...")
        Walpha_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Walpha_sub)(self, n, d) for n in range(ndraws))
        print("Done!")

        # print(Walpha_b, Walpha)
        pval = np.sum(Walpha_b > Walpha)/ndraws
        return pval


    def BS_Wbeta(self, l, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Test of instrument significance.
        Bootstrap inference on the hypothesis  l-th column of Gamma_beta = 0.

        Parameters
        ----------

        l   : integer
            Position of the characteristics for which the bootstrap is to be
            carried out. For example, if there are 10 characteristics, l is in
            the range 0 to 9 (left-/right-inclusive).

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        n_jobs  : integer
            Number of workers to be used for multiprocessing.
            If -1, all available Workers are used.

        backend : optional

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if self.alpha > 0.:
            raise ValueError("Bootstrap currently not supported for\
                              regularized estimation.")

        if self.PSFcase:
            raise ValueError('Need to fit model without intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "Q"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.metad["N"], self.metad["L"], self.metad["T"]

        # Compute Wbeta_l if l-th characteristics is set to zero
        Wbeta_l = self.Gamma[l, :].dot(self.Gamma[l, :].T)
        Wbeta_l = np.trace(Wbeta_l)
        # Compute residuals
        d = np.full((L, T), np.nan)
        for t in range(T):
            d[:, t] = self.Q[:, t]-self.W[:, :, t].dot(self.Gamma)\
                .dot(self.Factors[:, t])

        print("Starting Bootstrap...")
        Wbeta_l_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Wbeta_sub)(self, n, d, l) for n in range(ndraws))
        print("Done!")

        pval = np.sum(Wbeta_l_b > Wbeta_l)/ndraws
        # print(Wbeta_l_b, Wbeta_l)

        return pval


def _BS_Walpha_sub(model, n, d):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:,np.random.randint(0,high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b, W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True,
                                             data_type="portfolio")
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass


    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)
    #Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma)
    Gamma_beta_l[l, :] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:,np.random.randint(0,high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors[:, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b, W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True,
                                             data_type="portfolio")

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b



def _prep_panel(X, y=None, indices=None, date_ind=1):
    """handle mapping from different inputs type to consistent internal data

    Parameters
    ----------
    X :  numpy array or pandas DataFrame
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.

        If given as a DataFrame, we assume that it contains a mutliindex
        mapping to each entity-time pair

    y : numpy array or pandas Series, optional
        dependent variable where indices correspond to those in X

        If given as a Series, we assume that it contains a mutliindex
        mapping to each entity-time pair

    indices : pandas MultiIndex, optional
        array containinng the panel indices

        If None we use the values associated with the current model

    date_ind : scalar
        which level in the indices correspond to the date
        defaults to the second (index 1)

    Returns
    -------
    X :  numpy array
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.

    y : numpy array
        dependent variable where indices correspond to those in X

    indices : pandas MultiIndex
        array containinng the panel indices

    tind : numpy array
        array containing beginning and end of contiguous time period
        blocks for panel

    metad : dict
        contains metadata on inputs:

        dates : array-like
            unique dates in panel
        ids : array-like
            unique ids in panel
        chars : array-like
            labels for X chars/columns
        T : scalar
            number of time periods
        N : scalar
            number of ids
        L : scalar
            total number of characteristics
    """

    # Check panel input
    if X is None:
        raise ValueError('Must pass panel input data.')

    # if data-frames passed, break out indices from data
    if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
        indices = X.index
        chars = X.columns
        X = X.values
    elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        indices = y.index
        y = y.values
        chars = np.arange(X.shape[1])
    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        Xind = X.index
        chars = X.columns
        yind = y.index
        X = X.values
        y = y.values
        indices = Xind
    else:
        chars = np.arange(X.shape[1])

    if indices is None:
        raise ValueError("entity-time indices must be provided either\
                          separately or as a MultiIndex with X/y")

    # extract numpy array and labels from multiindex
    dates, ix_mn = np.unique(indices.get_level_values(date_ind).tolist(),
                             return_index=True)
    tind = np.array(list(ix_mn) + [X.shape[0]])
    ids = indices.levels[1-date_ind].values

    # init data dimensions
    T = np.size(dates, axis=0)
    N = np.size(ids, axis=0)
    L = np.size(chars, axis=0)

    # prep metadata
    metad = {}
    metad["dates"] = dates
    metad["ids"] = ids
    metad["chars"] = chars
    metad["TN"] = X.shape[0]
    metad["T"] = T
    metad["N"] = N
    metad["L"] = L

    return X, y, indices, tind, metad
