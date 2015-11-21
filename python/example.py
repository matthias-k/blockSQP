import pyBlockSQP
import numpy as np


class MyProblemspec(pyBlockSQP.PyProblemspec):
    def __init__(self, nVar, nCon, blockIdx, bl, x):
        super(MyProblemspec, self).__init__()
        self.nVar = nVar
        self.nCon = nCon
        #self.nnCon = 0
        self.objLo = -np.inf
        self.objUp = np.inf
        self.blockIdx = np.array(blockIdx, np.int32)

p = MyProblemspec(2, 1, [0,1,2], None, None)
o = pyBlockSQP.PySQPoptions()
s = pyBlockSQP.PySQPStats('./')
m = pyBlockSQP.PySQPMethod(p, o, s)
