# distutils: language = c++
### dddistutils: sources = Rectangle.cpp

from __future__ import print_function

from libc.stdio cimport FILE

from blockSQP_matrix cimport Matrix, SymMatrix

import numpy as np
cimport numpy as np

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t


cdef class PyMatrix:
    cdef Matrix *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int m = 1, int n = 1, data=None, int ldim = -1):
        cdef np.ndarray[DTYPEd_t, ndim=1] np_data_flat
        if data is not None:
            # Matrix uses column major
            np_data_flat = np.array(data, dtype=DTYPEd).flatten('F')
            self.thisptr = new Matrix(m, n, <double*>np_data_flat.data, ldim)
        else:
            self.thisptr = new Matrix(m, n, ldim)

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


#cdef extern from "blocksqp_problemspec.hpp" namespace "blockSQP":
#    cdef cppclass Problemspec:
#        int         nVar;               ///< number of variables
#        int         nCon;               ///< number of constraints
#        int         nnCon;              ///< number of nonlinear constraints
#
#        double      objLo;              ///< lower bound for objective
#        double      objUp;              ///< upper bound for objective
#        Matrix      bl;                 ///< lower bounds of variables and constraints
#        Matrix      bu;                 ///< upper bounds of variables and constraints
#
#        int         nBlocks;            ///< number of separable blocks of Lagrangian
#        int*        blockIdx;           ///< [blockwise] index in the variable vector where a block starts
