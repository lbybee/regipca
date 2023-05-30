from numpy cimport float64_t, ndarray, complex128_t
from numpy import log as nplog
from numpy import square, cov
from numpy import (identity, dot, kron, pi, sum, zeros_like, ones,
                   asfortranarray)
from numpy.linalg import pinv, norm, svd
cimport cython
cimport numpy as cnp
from libc.math cimport sin, cos, acos, exp, sqrt
from cython cimport double

cnp.import_array()

# included in Cython numpy headers
from numpy cimport PyArray_ZEROS
#from ._cython_blas cimport (_axpy, _dot, _asum, _ger, _gemv, _nrm2, 
#                           _copy, _scal)
#from scipy.linalg.cython_blas cimport dgemm, dgemv, zgemm, zgemv
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport sasum, dasum
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport scopy, dcopy
from scipy.linalg.cython_blas cimport sscal, dscal
from scipy.linalg.cython_blas cimport srotg, drotg
from scipy.linalg.cython_blas cimport srot, drot
from scipy.linalg.cython_blas cimport sgemv, dgemv
from scipy.linalg.cython_blas cimport sger, dger
from scipy.linalg.cython_blas cimport sgemm, dgemm
from scipy.linalg.cython_blas cimport ssyr, dsyr


ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
cdef int FORTRAN = 1

#cdef extern from "math.h":
#    double log(double x)

cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


cdef void _syr(char uplo, int n, double alpha, double *x, int incx,
               double *A, int lda) nogil:
    """A := alpha*x*x.T + A"""

    dsyr(&uplo, &n, &alpha, x, &incx, A, &lda)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kront(double[::1] XX, double[::1,:] factors,
          double[::1] kronvec, double[::1,:] kronmat,
          double[::1,:] factordot, int T, int L, int K):
    """compute the vectorized kronecker product for all T"""

    cdef:
        int t
        int i
        int j
        int k
        int l
        int iK
        int jK
        int loc
        int XXloc = 0
        int Xnloc = 0

    # populate vector for speed (reduce memory access)
    for t in range(T):

        _syr("l", K, 1., &factors[0,t], 1, &factordot[0,0], K)

        loc = 0
        Xnloc = 0
        for i in range(L):
            Xnloc += i
            for j in range(i, L):
                for k in range(K):
                    for l in range(k + 1):
                        kronvec[loc] += XX[XXloc+Xnloc] * factordot[k,l]
                        loc += 1
                Xnloc += 1

        _syr("l", K, -1., &factors[0,t], 1, &factordot[0,0], K)

        XXloc += L * L

    # convert to matrix
    loc = 0
    for i in range(0, L * K, K):
        # handle X diag
        for k in range(K):
            for l in range(k):
                kronmat[i+k,i+l] = kronvec[loc]
                kronmat[i+l,i+k] = kronvec[loc]
                loc += 1
            # handle k diag
            kronmat[i+k,i+k] = kronvec[loc]
            loc += 1
        for j in range(i + K, L * K, K):
            for k in range(K):
                for l in range(k):
                    kronmat[i+k,j+l] = kronvec[loc]
                    kronmat[i+l,j+k] = kronvec[loc]
                    kronmat[j+k,i+l] = kronvec[loc]
                    kronmat[j+l,i+k] = kronvec[loc]
                    loc += 1
                # handle k diag
                kronmat[i+k,j+k] = kronvec[loc]
                kronmat[j+k,i+k] = kronvec[loc]
                loc += 1
