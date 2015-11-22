# distutils: language = c++
### dddistutils: sources = Rectangle.cpp

from __future__ import print_function

from libc.stdio cimport FILE
cimport cpython.ref as cpy_ref

from blockSQP_matrix cimport Matrix, SymMatrix
from blockSQP_options cimport SQPoptions
from blockSQP_problemspec cimport IProblemspec, Problemspec
from blockSQP_stats cimport SQPstats
#from blockSQP_iterate cimport SQPiterate
from blockSQP_method cimport SQPmethod

import numpy as np
cimport numpy as np

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t


cdef class PyMatrix:
    cdef Matrix *thisptr      # hold a C++ instance which we're wrapping
    cdef object py_data
    def __cinit__(self, int m = 1, int n = 1, data=None, int ldim = -1):
        cdef DTYPEd_t [::1] data_view
        #cdef np.ndarray[DTYPEd_t, ndim=1] np_data_flat
        if data is not None:
            self.py_data = data
            # Matrix uses column major
            #np_data_flat = np.array(data, dtype=DTYPEd).flatten('F')
            data_view = data
            self.thisptr = new Matrix(m, n, &data_view[0] if n*m>0 else NULL, ldim)
            self.thisptr.tflag = 1 #thisptr does not own the data
        else:
            self.thisptr = new Matrix(m, n, ldim)

    def __dealloc__(self):
        del self.thisptr

    @classmethod
    def from_numpy_2d(cls, data):
        """
        Construct pymatrix from 2d numpy array.
        If data is not fortran contigous, it
        will be copied.
        """
        assert data.ndim == 2
        cdef int m, n
        m, n= data.shape
        flat_data = data.ravel('F')

        return cls(m, n, data = flat_data)

    def Dimension(self, int m, int n=1, int ldim = -1):
        self.thisptr.Dimension(m, n, ldim)
        return self

    def Initialize(self, double value):
        self.thisptr.Initialize(value)
        return self

    def Print(self):
        # does not work: segfault
        #self.thisptr.Print()
        #return self
        cdef int i, j
        for i in range(self.thisptr.M()):
            for j in range(self.thisptr.N()):
                print(self.thisptr.get(i, j), end=" ")
            print()
        return self

cdef class PySymMatrix:
    cdef SymMatrix *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int m = 1, data=None, int n = 1, int ldim = -1):
        cdef np.ndarray[DTYPEd_t, ndim=1] np_data_flat
        if data is not None:
            # Matrix uses column major
            np_data_flat = np.array(data, dtype=DTYPEd).flatten('F')
            self.thisptr = new SymMatrix(m, n, <double*>np_data_flat.data, ldim)
        else:
            self.thisptr = new SymMatrix(m, n, ldim)

    def __dealloc__(self):
        del self.thisptr

    def Dimension(self, int m, int n=1, int ldim = -1):
        self.thisptr.Dimension(m, n, ldim)
        return self

    def Initialize(self, double value):
        self.thisptr.Initialize(value)
        return self

    def Print(self):
        # does not work: segfault
        #self.thisptr.Print()
        #return self
        cdef int i, j
        for i in range((<Matrix*>(self.thisptr)).M()):
            for j in range((<Matrix*>(self.thisptr)).N()):
                print(self.thisptr.get(i, j), end=" ")
            print()
        return self

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

cdef public api int cy_call_evaluate_func_and_grad(object self,
                                                   char* method,
                                                   int *error,
                                                   const Matrix &xi,
                                                   const Matrix &lambda_,
                                                   double *objval,
                                                   Matrix &constr,
                                                   Matrix &gradObj,
                                                   Matrix &constrJac,
                                                   ):
    cdef double [::1] xi_view
    cdef double [::1, :] lambda_view

    cdef double objval_py
    cdef double [::1] constr_view
    cdef double [::1, :] gradObj_view
    cdef double [::1, :] constrJac_view

    try:
        func = getattr(self, method);
    except AttributeError:
        error[0] = 1
    else:
        error[0] = 0
        objval_py, constr_view, gradObj_view, constrJac_view = func(xi_view, lambda_view)


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



#cdef class PySQPiterate:
#    cdef SQPiterate *thisptr      # hold a C++ instance which we're wrapping
#    def __cinit__(self, problemspec, sqpoptions, ):
#        self.thisptr = new SQPiterate()
#
#    def __dealloc__(self):
#        del self.thisptr
