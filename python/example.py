import pyBlockSQP
import numpy as np


class MyProblemspec(pyBlockSQP.PyProblemspec):
    def __init__(self, nVar, nCon, blockIdx, bl, x):
        super(MyProblemspec, self).__init__()
        self.nVar = nVar
        self.nCon = nCon
        self.objLo = -np.inf
        self.objUp = np.inf
        self.blockIdx = np.array(blockIdx, np.int32)
        self.x0 = x

    def initialize_dense(self, xi, lambda_, constrJac):
        print("Init!")
        xi.numpy_data[:] = self.x0

x0 = np.array([10, 10.0])

p = MyProblemspec(2, 1, [0, 1, 2], None, x0)
opts = pyBlockSQP.PySQPoptions()
opts.opttol = 1.0e-12
opts.nlinfeastol = 1.0e-12
opts.globalization = 0
opts.hessUpdate = 0
opts.hessScaling = 0
opts.fallbackScaling = 0
opts.hessLimMem = 0
opts.maxConsecSkippedUpdates = 200
opts.blockHess = 0
opts.whichSecondDerv = 0

#opts.sparseQP = 2
opts.sparseQP = 0

opts.printLevel = 2


s = pyBlockSQP.PySQPStats('./')
m = pyBlockSQP.PySQPMethod(p, opts, s)

m.init()
