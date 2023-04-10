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
from scipy.linalg.cython_blas cimport sgemv, dgemv
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


cdef void _dgemv(char trans, int m, int n, double regconst, double *A, int lda,
                 double *x, int incx, double beta, double *y, int incy) nogil:
    """y := regconst*A*x + beta*y"""

    dgemv(&trans, &m, &n, &regconst, A, &lda, x, &incx, &beta, y, &incy)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double glasso_iter(double[::1] w, double[::1,:] XFXF, double[::1] XFy,
                        double[::1] stp_size, double[::1] grad,
                        double[::1] grad_update, double[::1] prox,
                        double[::1] regconst, int L, int K, int LK) nogil:

    cdef:
        int l = 0
        int k = 0
        double w_lk
        double gprox
        double score_num = 0
        double score_den = 0

    # iterate over chars/groups
    for l in range(L):

        gprox = 0

        # iterate over factors
        for k in range(K):

            # add the NEGATIVE gradient - (XFXFw - XFy)
            prox[k] = (w[l*K+k] * stp_size[l] - grad[l*K+k])
            gprox += prox[k] ** 2

        # generate slice fitted values
        _dgemv("n", LK, K, 1., &XFXF[0, l*K], LK, &w[l*K], 1,
               0, &grad_update[0], 1)

        # update grad (subtract off current estimates)
        _axpy(LK, -1, &grad_update[0], 1, &grad[0], 1)

        # build group penalty
        gprox = (1. - (regconst[l] / sqrt(gprox)))
        gprox = max(0, gprox)

        for k in range(K):

            w_lk = w[l*K+k]
            w[l*K+k] = prox[k] * gprox / stp_size[l]

            # update score
            score_num += (w[l*K+k] - w_lk) ** 2
            score_den += w_lk ** 2

        # generate slice fitted values
        _dgemv("n", LK, K, 1., &XFXF[0, l*K], LK, &w[l*K], 1,
               0, &grad_update[0], 1)

        # update grad (add back updated estimates)
        _axpy(LK, 1, &grad_update[0], 1, &grad[0], 1)

    return score_num / score_den


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def glasso(double[::1] w, double[::1,:] XFXF, double[::1] XFy,
           double[::1] grad, double[::1] stp_size, double[::1] regconst,
           int L, int K, int gliter=100, double gltol=1e-5,
           double glstpdep=1.):
    """Cython glasso implementation"""

    cdef:
        int i = 0
        int l = 0
        int LK = K * L

    cdef cnp.npy_intp pshape[2]
    pshape[0] = <cnp.npy_intp> K
    cdef cnp.npy_intp gshape[2]
    gshape[0] = <cnp.npy_intp> LK

    cdef:
        double [::1] prox = PyArray_ZEROS(1, pshape, cnp.NPY_DOUBLE, FORTRAN)
        double [::1] grad_update = PyArray_ZEROS(1, gshape, cnp.NPY_DOUBLE,
                                                  FORTRAN)

    # iterations
    with nogil:
        for i in range(gliter):

            # update est
            score = glasso_iter(w, XFXF, XFy, stp_size, grad, grad_update,
                                prox, regconst, L, K, LK)

#            # backtracking line search
#            stp_size = stp_size * glstpdep

            # return based on score
            if score < gltol:
                break

    if gliter == (i + 1):
        print("Didn't converge")

    return w
