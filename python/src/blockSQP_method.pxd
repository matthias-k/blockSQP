# distutils: language = c++

from blockSQP_matrix cimport Matrix
from blockSQP_problemspec cimport Problemspec
from blockSQP_options cimport SQPoptions
from blockSQP_stats cimport SQPstats

cdef extern from "blocksqp_method.hpp" namespace "blockSQP":
    cdef cppclass SQPmethod:
        SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ) except +
