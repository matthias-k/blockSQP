# distutils: language = c++
### dddistutils: sources = Rectangle.cpp

from blocksqp_matrix cimport Matrix
from blocksqp_problemspec import Problemspec

cdef extern from "blocksqp_method.hpp" namespace "blockSQP":
    cdef cppclass SQPmethod:
        SQPmethod( Problemspec *problem, SQPoptions *parameters, SQPstats *statistics ) except +
