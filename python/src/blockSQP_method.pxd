# distutils: language = c++

from blockSQP_matrix cimport Matrix
from blockSQP_problemspec cimport Problemspec
from blockSQP_options cimport SQPoptions
from blockSQP_stats cimport SQPstats

cdef extern from "blocksqp_method.hpp" namespace "blockSQP":
    cdef cppclass SQPmethod:
        SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ) except +

        # Initialization, has to be called before run
        void init()
        # Main Loop of SQP method
        int run( int maxIt, int warmStart = 0 )
        # Call after the last call of run, to close output files etc.
        void finish()
        # Print information about the SQP method
        void printInfo( int printLevel )
