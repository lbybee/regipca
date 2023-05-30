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
from scipy.linalg.cython_blas cimport ssymv, dsymv


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


cdef void _axpy(int n, double regconst, double *x, int incx,
                double *y, int incy) nogil:
    """y := regconst * x + y"""

    daxpy(&n, &regconst, x, &incx, y, &incy)


cdef void _symv(char uplo, int n, double regconst,
                double *A, int lda, double *x, int incx,
                double beta, double *y, int incy) nogil:
    """y := regconst * op(A).x + beta * y"""

    dsymv(&uplo, &n, &regconst, A, &lda, x, &incx, &beta, y, &incy)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double lsgrad_iter(double[::1] w, double[::1,:] XFXF, double[::1] XFy,
                        double[::1] stp_size, double[::1] grad,
                        int L, int K, int LK) nogil:

    cdef:
        int l = 0
        int k = 0
        double w_lk
        double grad_lk
        double target_num = 0
        double target_den = 0

    # generate current "fitted values" (X \kron F)^T (X \kron F) w
    _symv("l", LK, 1., &XFXF[0, 0], LK, &w[0], 1, 0, &grad[0], 1)

#    # generate current gradient
    _axpy(LK, -1, &XFy[0], 1, &grad[0], 1)

    # iterate over chars/groups
    for l in range(L):

        # iterate over factors
        for k in range(K):

            w_lk = w[l*K+k]
            w[l*K+k] = (w[l*K+k] - grad[l*K+k] * stp_size[l])

            # update target
            target_num += (w[l*K+k] - w_lk) ** 2
            target_den += w_lk ** 2

    return target_num / target_den


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lsgrad(double[::1] w, double[::1,:] XFXF, double[::1] XFy,
           double[::1] stp_size, int L, int K,
           int gliter=100, double gltol=1e-5, double glstpdep=1.):
    """Cython least-square grad-descent implementation"""

    cdef:
        int i = 0
        int l = 0
        int LK = K * L

    cdef cnp.npy_intp pshape[2]
    pshape[0] = <cnp.npy_intp> K
    cdef cnp.npy_intp gshape[2]
    gshape[0] = <cnp.npy_intp> LK

    cdef:
        double [::1] grad = PyArray_ZEROS(1, gshape, cnp.NPY_DOUBLE, FORTRAN)

#    regconst = regconst * sqrt(K)

    # iterations
    with nogil:
        for i in range(gliter):

            # update est
            target = lsgrad_iter(w, XFXF, XFy, stp_size, grad, L, K, LK)

#            # backtracking line search
#            stp_size = stp_size * glstpdep

            # return based on target
            if target < gltol:
                break

    if gliter == (i + 1):
        print("Didn't converge")

    return w
