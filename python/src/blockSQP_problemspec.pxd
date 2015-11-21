# distutils: language = c++

cimport cpython.ref as cpy_ref

from blockSQP_matrix cimport Matrix, SymMatrix


cdef extern from "blocksqp_problemspec.hpp" namespace "blockSQP":
    cdef cppclass Problemspec:
        int         nVar               # number of variables
        int         nCon               # number of constraints
        int         nnCon              # number of nonlinear constraints

        double      objLo              # lower bound for objective
        double      objUp              # upper bound for objective
        Matrix      bl                 # lower bounds of variables and constraints
        Matrix      bu                 # upper bounds of variables and constraints

        int         nBlocks            # number of separable blocks of Lagrangian
        int*        blockIdx           # [blockwise] index in the variable vector where a block starts

        Problemspec() except +

        # Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (dense version).
        void initialize( Matrix &xi,            #///< optimization variables
                         Matrix &lambda_,        #///< Lagrange multipliers
                         Matrix &constrJac      #///< constraint Jacobian (dense)
                         )

        # Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (sparse version).
        void initialize( Matrix &xi,            # optimization variables
                         Matrix &lambda_,        # Lagrange multipliers
                         double *&jacNz,        # nonzero elements of constraint Jacobian
                         int *&jacIndRow,       # row indices of nonzero elements
                         int *&jacIndCol        # starting indices of columns
                         )

        # Evaluate objective, constraints, and derivatives (dense version).
        void evaluate( const Matrix &xi,        # optimization variables
                       const Matrix &lambda_,    # Lagrange multipliers
                       double *objval,          # objective function value
                       Matrix &constr,          # constraint function values
                       Matrix &gradObj,         # gradient of objective
                       Matrix &constrJac,       # constraint Jacobian (dense)
                       SymMatrix *&hess,        # Hessian of the Lagrangian (blockwise)
                       int dmode,               # derivative mode
                       int *info                # error flag
                       )

        # Evaluate objective, constraints, and derivatives (sparse version).
        void evaluate( const Matrix &xi,        # optimization variables
                       const Matrix &lambda_,    # Lagrange multipliers
                       double *objval,          # objective function value
                       Matrix &constr,          # constraint function values
                       Matrix &gradObj,         # gradient of objective
                       double *&jacNz,          # nonzero elements of constraint Jacobian
                       int *&jacIndRow,         # row indices of nonzero elements
                       int *&jacIndCol,         # starting indices of columns
                       SymMatrix *&hess,        # Hessian of the Lagrangian (blockwise)
                       int dmode,               # derivative mode
                       int *info                # error flag
                       )

        # Short cut if no derivatives are needed
        void evaluate( const Matrix &xi,        # optimization variables
                       double *objval,          # objective function value
                       Matrix &constr,          # constraint function values
                       int *info                # error flag
                       )



#cdef extern from "blocksqp_problemspec.hpp" namespace "blockSQP":
cdef extern from "IProblemspec.h" namespace "blockSQP":
    cdef cppclass IProblemspec(Problemspec):
        IProblemspec(cpy_ref.PyObject *obj) except +
