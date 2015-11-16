# distutils: language = c++

from libcpp cimport bool
from libc.stdio cimport FILE

from blockSQP_problemspec cimport Problemspec
from blockSQP_matrix cimport Matrix


cdef extern from "blocksqp_defs.hpp" namespace "blockSQP":
    ctypedef char PATHSTR[4096]

