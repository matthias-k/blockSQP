# distutils: language = c++
### dddistutils: sources = Rectangle.cpp

from __future__ import print_function

from libc.stdio cimport FILE
cimport cpython.ref as cpy_ref

from blockSQP_matrix cimport Matrix, SymMatrix
from blockSQP_options cimport SQPoptions
from blockSQP_problemspec cimport IProblemspec, Problemspec
from blockSQP_stats cimport SQPstats
from blockSQP_iterate cimport SQPiterate
from blockSQP_method cimport SQPmethod

import numpy as np
cimport numpy as np

np.import_array()

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t


cdef class PyMatrix:
    cdef Matrix *thisptr      # hold a C++ instance which we're wrapping
    cdef object py_data
    def __cinit__(self, int m = 1, int n = 1, data=None, int ldim = -1):
        if type(self) is not PyMatrix:
            return
        cdef DTYPEd_t [::1] data_view
        if data is not None:
            self.py_data = data
            data_view = data
            self.thisptr = new Matrix(m, n, &data_view[0] if n*m>0 else NULL, ldim)
            self.thisptr.tflag = 1 #thisptr does not own the data
        else:
            self.thisptr = new Matrix(m, n, ldim)

    def __dealloc__(self):
        if type(self) is not PyMatrix:
            return
        del self.thisptr

    @property
    def numpy_data(self):
        cdef Matrix* matrix = self.thisptr
        cdef int m = matrix.M()
        cdef int n = matrix.N()

        # TODO: m*n==0
        cdef DTYPEd_t[::1] data_view = <DTYPEd_t[:m*n]>matrix.array
        return np.asarray(data_view)

    @property
    def as_array(self):
        data_array = self.numpy_data
        cdef int m = self.thisptr.M()
        cdef int n = self.thisptr.N()

        return data_array.reshape((m, n), order='F')

    @classmethod
    def from_numpy_2d(cls, data):
        """
        Construct pymatrix from 2d numpy array.
        If data is not fortran contigous or of wrong
        type, it will be copied.
        """
        assert data.ndim == 2
        cdef int m, n
        m, n= data.shape
        flat_data = data.astype(np.float).ravel('F')

        return cls(m, n, data = flat_data)


    def Dimension(self, int m, int n=1, int ldim = -1):
        self.thisptr.Dimension(m, n, ldim)
        return self

    def Initialize(self, double value):
        self.thisptr.Initialize(value)
        return self

    def Print(self):
        self.thisptr.Print()
        return self

    def copy(self):
        return from_blockSQP_matrix(self.thisptr)


cdef from_blockSQP_matrix(Matrix* matrix):
    cdef int m = matrix.M()
    cdef int n = matrix.N()

    # TODO: m*n==0
    cdef DTYPEd_t[::1] data_view = <DTYPEd_t[:m*n]>matrix.array
    return PyMatrix(m, n, data_view)

# TODO: There should be an easier way to do this
cdef from_const_blockSQP_matrix(const Matrix* matrix):
    cdef int m = matrix.M()
    cdef int n = matrix.N()

    # TODO: m*n==0
    cdef DTYPEd_t[::1] data_view = <DTYPEd_t[:m*n]>matrix.array
    return PyMatrix(m, n, data_view)


cdef class PySymMatrix(PyMatrix):
    """PySymMatrix interfaces blockSQP's SymMatrix. SymMatrix
    describes a symmetric matrix by storing the n*(n+1)/2 elements
    of the upper triangle. They are stored in an array layed out
    in c order (i.e. row wise).

    As numpy does not support storing symmetric matrices with
    reduced memory, when converting from or to 2d numpy arrays,
    the data has to be copied and extended or filtered."""
    def __cinit__(self, int m = 1, data=None, int n = 1, int ldim = -1):
        cdef DTYPEd_t [::1] data_view
        if data is not None:
            self.py_data = data
            data_view = data
            self.thisptr = <Matrix*> new SymMatrix(m, n, &data_view[0] if n*m>0 else NULL, ldim)
            self.thisptr.tflag = 1 #thisptr does not own the data
        else:
            self.thisptr = <Matrix*>new SymMatrix(m, n, ldim)

    def __dealloc__(self):
        del self.thisptr

    def Dimension(self, int m, int n=1, int ldim = -1):
        (<SymMatrix*>(self.thisptr)).Dimension(m, n, ldim)
        return self

    def Initialize(self, double value):
        (<SymMatrix*>(self.thisptr)).Initialize(value)
        return self

    @property
    def numpy_data(self):
        cdef Matrix* matrix = self.thisptr
        cdef int m = matrix.M()
        cdef int n = matrix.N()

        cdef int length = m*(n+1)/2

        # TODO: m*n==0
        cdef DTYPEd_t[::1] data_view = <DTYPEd_t[:length]>matrix.array
        return np.asarray(data_view)

    @property
    def as_array(self):
        """
        As numpy arrays do not support the storage
        format of SymMatrix, this property is
        always a copy of the actual data. Therefore you
        should not modify it, modifications will not affect
        the SymMatrix. If you have to modify the actual data,
        use the property numpy_data.
        """
        data_array = self.numpy_data
        cdef int m = self.thisptr.M()
        cdef int n = self.thisptr.N()

        output = np.empty((m, n))

        output[np.triu_indices(n)] = data_array
        output.T[np.triu_indices(n)] = data_array

        return output

    @classmethod
    def from_numpy_2d(cls, data):
        """
        Construct PySymMatrix from 2d numpy array.
        As numpy does not support storing symmetric
        matrices explicitly, the data has to be copied
        and reordered. Only the upper triangle of data
        is used.
        """
        assert data.ndim == 2
        cdef int m, n
        m, n= data.shape
        assert m == n

        flat_data = data[np.triu_indices(n)].copy().astype(np.float)

        return cls(m, data = flat_data)


cdef from_blockSQP_symmatrix(SymMatrix* matrix):
    cdef int m = matrix.M()
    cdef int length = m*(m+1)/2

    # TODO: m*n==0
    cdef DTYPEd_t[::1] data_view = <DTYPEd_t[:length]>matrix.array
    return PySymMatrix(m=m, data=data_view)


cdef class PySQPoptions:
    cdef SQPoptions *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new SQPoptions()

    def __dealloc__(self):
        del self.thisptr

    def optionsConsistency(self):
        self.thisptr.optionsConsistency()

    property printLevel:
        def __get__(self): return self.thisptr.printLevel
        def __set__(self, printLevel): self.thisptr.printLevel = printLevel

    property printColor:
        def __get__(self): return self.thisptr.printColor
        def __set__(self, printColor): self.thisptr.printColor = printColor

    property debugLevel:
        def __get__(self): return self.thisptr.debugLevel
        def __set__(self, debugLevel): self.thisptr.debugLevel = debugLevel

    property eps:
        def __get__(self): return self.thisptr.eps
        def __set__(self, eps): self.thisptr.eps = eps

    property inf:
        def __get__(self): return self.thisptr.inf
        def __set__(self, inf): self.thisptr.inf = inf

    property opttol:
        def __get__(self): return self.thisptr.opttol
        def __set__(self, opttol): self.thisptr.opttol = opttol

    property nlinfeastol:
        def __get__(self): return self.thisptr.nlinfeastol
        def __set__(self, nlinfeastol): self.thisptr.nlinfeastol = nlinfeastol

    property sparseQP:
        def __get__(self): return self.thisptr.sparseQP
        def __set__(self, sparseQP): self.thisptr.sparseQP = sparseQP

    property globalization:
        def __get__(self): return self.thisptr.globalization
        def __set__(self, globalization): self.thisptr.globalization = globalization

    property restoreFeas:
        def __get__(self): return self.thisptr.restoreFeas
        def __set__(self, restoreFeas): self.thisptr.restoreFeas = restoreFeas

    property maxLineSearch:
        def __get__(self): return self.thisptr.maxLineSearch
        def __set__(self, maxLineSearch): self.thisptr.maxLineSearch = maxLineSearch

    property maxConsecReducedSteps:
        def __get__(self): return self.thisptr.maxConsecReducedSteps
        def __set__(self, maxConsecReducedSteps): self.thisptr.maxConsecReducedSteps = maxConsecReducedSteps

    property maxConsecSkippedUpdates:
        def __get__(self): return self.thisptr.maxConsecSkippedUpdates
        def __set__(self, maxConsecSkippedUpdates): self.thisptr.maxConsecSkippedUpdates = maxConsecSkippedUpdates

    property maxItQP:
        def __get__(self): return self.thisptr.maxItQP
        def __set__(self, maxItQP): self.thisptr.maxItQP = maxItQP

    property blockHess:
        def __get__(self): return self.thisptr.blockHess
        def __set__(self, blockHess): self.thisptr.blockHess = blockHess

    property hessScaling:
        def __get__(self): return self.thisptr.hessScaling
        def __set__(self, hessScaling): self.thisptr.hessScaling = hessScaling

    property fallbackScaling:
        def __get__(self): return self.thisptr.fallbackScaling
        def __set__(self, fallbackScaling): self.thisptr.fallbackScaling = fallbackScaling

    property maxTimeQP:
        def __get__(self): return self.thisptr.maxTimeQP
        def __set__(self, maxTimeQP): self.thisptr.maxTimeQP = maxTimeQP

    property iniHessDiag:
        def __get__(self): return self.thisptr.iniHessDiag
        def __set__(self, iniHessDiag): self.thisptr.iniHessDiag = iniHessDiag

    property colEps:
        def __get__(self): return self.thisptr.colEps
        def __set__(self, colEps): self.thisptr.colEps = colEps

    property colTau1:
        def __get__(self): return self.thisptr.colTau1
        def __set__(self, colTau1): self.thisptr.colTau1 = colTau1

    property colTau2:
        def __get__(self): return self.thisptr.colTau2
        def __set__(self, colTau2): self.thisptr.colTau2 = colTau2

    property hessDamp:
        def __get__(self): return self.thisptr.hessDamp
        def __set__(self, hessDamp): self.thisptr.hessDamp = hessDamp

    property hessDampFac:
        def __get__(self): return self.thisptr.hessDampFac
        def __set__(self, hessDampFac): self.thisptr.hessDampFac = hessDampFac

    property hessUpdate:
        def __get__(self): return self.thisptr.hessUpdate
        def __set__(self, hessUpdate): self.thisptr.hessUpdate = hessUpdate

    property fallbackUpdate:
        def __get__(self): return self.thisptr.fallbackUpdate
        def __set__(self, fallbackUpdate): self.thisptr.fallbackUpdate = fallbackUpdate

    property hessLimMem:
        def __get__(self): return self.thisptr.hessLimMem
        def __set__(self, hessLimMem): self.thisptr.hessLimMem = hessLimMem

    property hessMemsize:
        def __get__(self): return self.thisptr.hessMemsize
        def __set__(self, hessMemsize): self.thisptr.hessMemsize = hessMemsize

    property whichSecondDerv:
        def __get__(self): return self.thisptr.whichSecondDerv
        def __set__(self, whichSecondDerv): self.thisptr.whichSecondDerv = whichSecondDerv

    property skipFirstGlobalization:
        def __get__(self): return self.thisptr.skipFirstGlobalization
        def __set__(self, skipFirstGlobalization): self.thisptr.skipFirstGlobalization = skipFirstGlobalization

    property convStrategy:
        def __get__(self): return self.thisptr.convStrategy
        def __set__(self, convStrategy): self.thisptr.convStrategy = convStrategy

    property maxConvQP:
        def __get__(self): return self.thisptr.maxConvQP
        def __set__(self, maxConvQP): self.thisptr.maxConvQP = maxConvQP

    property maxSOCiter:
        def __get__(self): return self.thisptr.maxSOCiter
        def __set__(self, maxSOCiter): self.thisptr.maxSOCiter = maxSOCiter

    property gammaTheta:
        def __get__(self): return self.thisptr.gammaTheta
        def __set__(self, gammaTheta): self.thisptr.gammaTheta = gammaTheta

    property gammaF:
        def __get__(self): return self.thisptr.gammaF
        def __set__(self, gammaF): self.thisptr.gammaF = gammaF

    property kappaSOC:
        def __get__(self): return self.thisptr.kappaSOC
        def __set__(self, kappaSOC): self.thisptr.kappaSOC = kappaSOC

    property kappaF:
        def __get__(self): return self.thisptr.kappaF
        def __set__(self, kappaF): self.thisptr.kappaF = kappaF

    property thetaMax:
        def __get__(self): return self.thisptr.thetaMax
        def __set__(self, thetaMax): self.thisptr.thetaMax = thetaMax

    property thetaMin:
        def __get__(self): return self.thisptr.thetaMin
        def __set__(self, thetaMin): self.thisptr.thetaMin = thetaMin

    property delta:
        def __get__(self): return self.thisptr.delta
        def __set__(self, delta): self.thisptr.delta = delta

    property sTheta:
        def __get__(self): return self.thisptr.sTheta
        def __set__(self, sTheta): self.thisptr.sTheta = sTheta

    property sF:
        def __get__(self): return self.thisptr.sF
        def __set__(self, sF): self.thisptr.sF = sF

    property kappaMinus:
        def __get__(self): return self.thisptr.kappaMinus
        def __set__(self, kappaMinus): self.thisptr.kappaMinus = kappaMinus

    property kappaPlus:
        def __get__(self): return self.thisptr.kappaPlus
        def __set__(self, kappaPlus): self.thisptr.kappaPlus = kappaPlus

    property kappaPlusMax:
        def __get__(self): return self.thisptr.kappaPlusMax
        def __set__(self, kappaPlusMax): self.thisptr.kappaPlusMax = kappaPlusMax

    property deltaH0:
        def __get__(self): return self.thisptr.deltaH0
        def __set__(self, deltaH0): self.thisptr.deltaH0 = deltaH0

    property eta:
        def __get__(self): return self.thisptr.eta
        def __set__(self, eta): self.thisptr.eta = eta


cdef public api int cy_call_func(object self, char* method, int *error):
    try:
        func = getattr(self, method);
    except AttributeError:
        error[0] = 1
    else:
        error[0] = 0
        return func()



cdef class PyProblemspec:
    cdef IProblemspec *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new IProblemspec(<cpy_ref.PyObject*>self)

    def __dealloc__(self):
        del self.thisptr

    property nVar:
        def __get__(self): return self.thisptr.nVar
        def __set__(self, int nVar): self.thisptr.nVar = nVar

    property nCon:
        def __get__(self): return self.thisptr.nCon
        def __set__(self, int nCon): self.thisptr.nCon = nCon

    property nnCon:
        def __get__(self): return self.thisptr.nnCon
        def __set__(self, int nnCon): self.thisptr.nnCon = nnCon

    property objLo:
        def __get__(self): return self.thisptr.objLo
        def __set__(self, double objLo): self.thisptr.objLo = objLo

    property objUp:
        def __get__(self): return self.thisptr.objUp
        def __set__(self, double objUp): self.thisptr.objUp = objUp

#    property bl:
#        def __get__(self): return self.thisptr.bl
#        def __set__(self, bl): self.thisptr.bl = bl

    property nBlocks:
        def __get__(self): return self.thisptr.nBlocks
        def __set__(self, int nBlocks): self.thisptr.nBlocks = nBlocks

    property blockIdx:
        def __get__(self):
            cdef int n = self.thisptr.nBlocks
            cdef int [::1] blockIdx_view = <int[:n+1]*>self.thisptr.blockIdx
            return blockIdx_view
        def __set__(self, blockIdx):
            # TODO: right now we need a np.int32 np-array
            # should this method also work with int64 and python lists?
            # Or should this be handled in a more highlevel
            # wrapper class in python?
            if not len(blockIdx) > 0:
                raise ValueError('blockIdx has to be at least of length 1')
            if not blockIdx[0] == 0:
                raise ValueError('it is required that blockIdx[0]==0')
            if not blockIdx[-1] == self.nVar:
                raise ValueError('it is required that blockIdx[-1] == nVar!')
            cdef int [::1] blockIdx_view = blockIdx
            # Make sure we have a reference in python
            self.blockIdx_view = blockIdx_view
            self.thisptr.nBlocks = len(blockIdx_view)-1
            self.thisptr.blockIdx = &blockIdx_view[0]

    def initialize_dense(self, PyMatrix xi, PyMatrix lambda_, PyMatrix constrJac):
        """
        Set initial values for xi (and possibly lambda) and parts
        of the Jacobian that correspond to linear constraints (dense version).
        """
        pass

    def initialize_sparse(self, PyMatrix xi, PyMatrix lambda_):
        """
        Set initial values for xi (and possibly lambda) and parts
        of the Jacobian that correspond to linear constraints (sparse version).
        """
        raise NotImplementedError()

    def evaluate_dense(self,
                       PyMatrix xi,
                       PyMatrix lambda_,
                       PyMatrix constr,
                       PyMatrix gradObj,
                       PyMatrix constrJac,
                       PySymMatrix hess,
                       int dmode):
        """
        Evaluate objective, constraints, and derivatives (dense version).
        Has to return objective and update contr, gradObj, contrJac
        and (if dmode>1) hess.
        """
        raise NotImplementedError()




cdef public api int cy_call_initialize_dense(object self,
                                             Matrix &xi,
                                             Matrix &lambda_,
                                             Matrix &constrJac,
                                             ):

    cdef PyMatrix py_xi = from_blockSQP_matrix(&xi)
    cdef PyMatrix py_lambda = from_blockSQP_matrix(&lambda_)
    cdef PyMatrix py_constrJac = from_blockSQP_matrix(&constrJac)

    func = getattr(self, "initialize_dense")
    func(py_xi, py_lambda, py_constrJac)


cdef public api int cy_call_initialize_sparse(object self,
                                             Matrix &xi,
                                             Matrix &lambda_,
                                             double *&jacNz,
                                             int *&jacIndRow,
                                             int *&jacIndCol,
                                             ):

    cdef PyMatrix py_xi = from_blockSQP_matrix(&xi)
    cdef PyMatrix py_lambda = from_blockSQP_matrix(&lambda_)

    func = getattr(self, "initialize_sparse")
    jacNz_, jacIndRow_, jacIndCol_ = func(py_xi, py_lambda)

    jacNz_ = np.ascontiguousarray(jacNz_, DTYPEd)
    jacIndRow_ = np.ascontiguousarray(jacIndRow_, np.int32)
    jacIndCol_ = np.ascontiguousarray(jacIndCol_, np.int32)

    # make sure that memory is not freed before object terminates
    # TODO: actually the memory is freed by blockSQP, thus
    # it should not even be freed when PyProblemspec is garbadge collected
    self._jacNz = jacNz_
    self._jacIndRow = jacIndRow_
    self._jacIndCol = jacIndCol_

    cdef DTYPEd_t[::1] jacNz_view = jacNz_
    cdef int[::1] jacIndRow_view = jacIndRow_
    cdef int[::1] jacIndCol_view = jacIndCol_

    # This ugly syntax is a workaround for
    # https://groups.google.com/forum/#!topic/cython-users/j58Sp3QMrD4
    (&jacNz)[0] = &jacNz_view[0]
    (&jacIndRow)[0] = &jacIndRow_view[0]
    (&jacIndCol)[0] = &jacIndCol_view[0]


cdef public api int cy_call_evaluate_dense(object self,
                                           const Matrix &xi,
                                           const Matrix &lambda_,
                                           double *objval,
                                           Matrix &constr,
                                           Matrix &gradObj,
                                           Matrix &constrJac,
                                           SymMatrix *&hess,
                                           int dmode,
                                           int *info
                                           ):

    cdef PyMatrix py_xi = from_const_blockSQP_matrix(&xi)
    cdef PyMatrix py_lambda = from_const_blockSQP_matrix(&lambda_)
    cdef PyMatrix py_constr = from_blockSQP_matrix(&constr)
    cdef PyMatrix py_gradObj = from_blockSQP_matrix(&gradObj)
    cdef PyMatrix py_constrJac = from_blockSQP_matrix(&constrJac)
    cdef double objVal_

    cdef PySymMatrix py_hess = None

    if dmode == 2:
        py_hess = from_blockSQP_symmatrix(&(hess[self.nBlocks-1]))
    elif dmode == 3:
        raise NotImplementedError()


    func = getattr(self, "evaluate_dense")
    info[0] = 0
    try:
        objVal_ = func(py_xi, py_lambda, py_constr,
                       py_gradObj, py_constrJac, py_hess,
                       dmode)
        objval[0] = objVal_
    except Exception as e:
        # self.__exception__ = sys.exc_info()
        print("Exception in evaluate_dense")
        print(e)
        info[0] = 1


cdef class PySQPStats:
    cdef SQPstats *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, char* filename):
        self.thisptr = new SQPstats(filename)

    def __dealloc__(self):
        del self.thisptr


cdef class PySQPMethod:
    cdef SQPmethod *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, PyProblemspec problemspec, PySQPoptions options, PySQPStats stats):

        cdef Problemspec* _problemspec = problemspec.thisptr
        cdef SQPoptions* _options =  options.thisptr
        cdef SQPstats* _stats = stats.thisptr
        self.thisptr = new SQPmethod(_problemspec,
                                     _options,
                                     _stats)

    def __dealloc__(self):
        del self.thisptr

    def init(self):
        """Initialization, has to be called before run"""
        self.thisptr.init()

    def run(self, int maxIt, int warmStart = 0):
        """Main Loop of SQP method"""
        return self.thisptr.run(maxIt, warmStart)

    def finish(self):
        """all after the last call of run, to close output files etc. """
        self.thisptr.finish()

    def printInfo(self, int printLevel):
        """Print information about the SQP method"""
        self.thisptr.printInfo(printLevel)

    property vars:
        def __get__(self):
            py_sqp_iterate = PySQPiterate()
            py_sqp_iterate.thisptr = self.thisptr.vars
            return py_sqp_iterate




cdef class PySQPiterate:
    """
    This class is only used to interface existing SQPiterates.
    Therefore it does not free it's SQPiterate instance on
    __dealloc__.
    """
    cdef SQPiterate *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    property xi:
        def __get__(self):
            return from_blockSQP_matrix(&(self.thisptr.xi))

    property lambda_:
        def __get__(self):
            return from_blockSQP_matrix(&(self.thisptr.lambda_))

    property nBlocks:
        def __get__(self):
            return self.thisptr.nBlocks

    property hess:
        def __get__(self):
            blocks = []
            cdef int i
            cdef SymMatrix* this_hess
            for i in range(self.thisptr.nBlocks):
                this_hess = &(self.thisptr.hess[i])
                blocks.append(from_blockSQP_symmatrix(this_hess))
            return blocks

