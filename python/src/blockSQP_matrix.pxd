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
        Matrix( int, int, double*, int) except +    #

        int M() const #                                          ///< number of rows
        int N() const #                                          ///< number of columns
        int LDIM() const #                                       ///< leading dimensions
        double *ARRAY() const#                                   ///< returns pointer to data array
        int TFLAG() const #

        double& get "operator()"( int i, int j )
        #double& get "operator()"( int i, int j ) const
        double& get "operator()"( int i )
        #double& get "operator()"( int i ) const

        Matrix &Dimension( int, int, int)
        Matrix &Initialize( double (*)( int, int ) )
        Matrix &Initialize( double )

        const Matrix &Print( FILE* = stdout, #  ///< file for output
                               int = 13,     #  ///< number of digits
                               int = 1       #  ///< Flag for format
                             ) const

    cdef cppclass SymMatrix:
        SymMatrix(int)
        SymMatrix(int, double*)
        SymMatrix(int, int, int)
        SymMatrix(int, int, double*, int)
        SymMatrix( const Matrix& A )
        SymMatrix( const SymMatrix& A )

        double& get "operator()"( int i, int j )
        #double& "operator()"( int i, int j ) const;
        double& get "operator()"( int i )
        #double& "operator()"( int i ) const;

        SymMatrix &Dimension(int)
        SymMatrix &Dimension(int, int, int )
        SymMatrix &Initialize( double (*)( int, int ) )
        SymMatrix &Initialize( double )

        SymMatrix& Submatrix( const Matrix&, int, int, int, int)
        SymMatrix& Arraymatrix( int, double*)
        SymMatrix& Arraymatrix( int, int, double*, int)
