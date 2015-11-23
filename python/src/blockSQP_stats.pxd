# distutils: language = c++

from libcpp cimport bool
from libc.stdio cimport FILE

from blockSQP_problemspec cimport Problemspec
from blockSQP_matrix cimport Matrix
from blockSQP_defs cimport PATHSTR
from blockSQP_iterate import SQPiterate


cdef extern from "blocksqp_stats.hpp" namespace "blockSQP":
    cdef cppclass SQPstats:
        int itCount                 # iteration number
        int qpIterations            # number of qp iterations in the current major iteration
        int qpIterations2           # number of qp iterations for solving convexified QPs
        int qpItTotal               # total number of qp iterations
        int qpResolve               # how often has QP to be convexified and resolved?
        int nFunCalls               # number of function calls
        int nDerCalls               # number of derivative calls
        int nRestHeurCalls          # number calls to feasibility restoration heuristic
        int nRestPhaseCalls         # number calls to feasibility restoration phase
        int rejectedSR1             # count how often the SR1 update is rejected
        int hessSkipped             # number of block updates skipped in the current iteration
        int hessDamped              # number of block updates damped in the current iteration
        int nTotalUpdates
        int nTotalSkippedUpdates
        double averageSizingFactor  # average value (over all blocks) of COL sizing factor
        PATHSTR outpath             # path where log files are stored

        FILE *progressFile          # save stats for each SQP step
        FILE *updateFile            # print update sequence (SR1/BFGS) to file
        FILE *primalVarsFile        # primal variables for every SQP iteration
        FILE *dualVarsFile          # dual variables for every SQP iteration
        FILE *jacFile               # Jacobian of one iteration
        FILE *hessFile              # Hessian of one iteration

        # Constructor
        SQPstats( PATHSTR myOutpath )
        # Open output files
        void initStats( SQPoptions *param )
        # Print Debug information in logfiles
        void printDebug( SQPiterate *vars, SQPoptions *param )
        # Print current iterate of primal variables to file
        void printPrimalVars( const Matrix &xi )
        # Print current iterate of dual variables to file
        void printDualVars( const Matrix &lambda )
        # Print all QP data to files to be read in MATLAB
        void dumpQPMatlab( Problemspec *prob, SQPiterate *vars, int sparseQP )
        void dumpQPCpp( Problemspec *prob, SQPiterate *vars, qpOASES::SQProblem *qp, int sparseQP )
        void printVectorCpp( FILE *outfile, double *vec, int len, char* varname )
        void printVectorCpp( FILE *outfile, int *vec, int len, char* varname )
        void printCppNull( FILE *outfile, char* varname )
        # Print current (full) Jacobian to Matlab file
        void printJacobian( const Matrix &constrJacFull )
        void printJacobian( int nCon, int nVar, double *jacNz, int *jacIndRow, int *jacIndCol )
        # Print current (full) Hessian to Matlab file
        void printHessian( int nBlocks, SymMatrix *&hess )
        void printHessian( int nVar, double *hesNz, int *hesIndRow, int *hesIndCol )
        # Print a sparse Matrix in (column compressed) to a MATLAB readable file
        void printSparseMatlab( FILE *file, int nRow, int nVar, double *nz, int *indRow, int *indCol )
        # Print one line of output to stdout about the current iteration
        void printProgress( Problemspec *prob, SQPiterate *vars, SQPoptions *param, bool hasConverged )
        # Must be called before returning from run()
        void finish( SQPoptions *param )

