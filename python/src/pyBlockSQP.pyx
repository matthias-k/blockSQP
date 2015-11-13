# distutils: language = c++
### dddistutils: sources = Rectangle.cpp

from libc.stdio cimport FILE

cdef extern from "blocksqp_matrix.hpp" namespace "blockSQP":
    cdef cppclass Matrix:
        int m                       #                                 ///< internal number of rows
        int n                       #                                 ///< internal number of columns
        int ldim                    #                                 ///< internal leading dimension not necesserily equal to m or n
        double *array               #                                 ///< array of how the matrix is stored in the memory
        int tflag                   #

        Matrix( int, int, int) except +     #                    ///< constructor with standard arguments
        Matrix( int, int, double*, int = -1 ) except +    #

        Matrix &Dimension( int, int, int)
        Matrix &Initialize( double (*)( int, int ) )
        Matrix &Initialize( double )

        const Matrix &Print( FILE* = stdout, #  ///< file for output
                               int = 13,     #  ///< number of digits
                               int = 1       #  ///< Flag for format
                             ) const


cdef class PyMatrix:
    cdef Matrix *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int m = 1, int n = 1, int ldim = -1):
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
        self.thisptr.Print()
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
