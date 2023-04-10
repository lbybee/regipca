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
from scipy.linalg.cython_blas cimport ssyrk, dsyrk


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


cdef void _axpy(int n, double alpha, double *x, int incx,
                double *y, int incy) nogil:
    """y := alpha * x + y"""

    daxpy(&n, &alpha, x, &incx, y, &incy)


cdef void _syrk(char uplo, char trans, int n, int k, double alpha,
                double *A, int lda, double beta, double *C,
                int ldc) nogil:
    """C := alpha*A.T*A + beta*C"""

    dsyrk(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc)


cdef void _gemv(char ta, int m, int n, double alpha,
                double *A, int lda, double *x, int incx,
                double beta, double *y, int incy) nogil:
    """y := alpha * op(A).x + beta * y"""

    dgemv(&ta, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def innert(double [::1,:] X, double[::1] y, long long[::1] tind,
           double[::1] XX, double[::1] Xy, int T, int L):
    """compute the inner product for each sub X matrix"""

    cdef:
        int ind
        int indF
        int offset
        int t = 0
        int Xloc = 0
        int XXloc = 0
        int Xyloc = 0
        double [::1,:] Xt

    for t in range(T):

        ind = tind[t]
        indF = tind[t+1]
        offset = indF - ind

        # TODO can we do better here?
        Xt = X[ind:indF,:].copy_fortran()

        _syrk("l", "t", L, offset, 1., &Xt[0,0], offset, 0.,
              &XX[XXloc], L)
        _gemv("T", offset, L, 1., &Xt[0,0], offset, &y[ind], 1,
              1., &Xy[Xyloc], 1)

        Xloc += offset * L
        XXloc += L * L
        Xyloc += L
