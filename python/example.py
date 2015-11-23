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

    def initialize_sparse(self, xi, lambda_):
        print("init sparse!")
        xi.numpy_data[:] = self.x0

        constrDummy = pyBlockSQP.PyMatrix().Dimension(self.nCon)\
                          .Initialize(0.0)
        gradObjDummy = pyBlockSQP.PyMatrix().Dimension(self.nVar)\
                          .Initialize(0.0)
        constrJac = pyBlockSQP.PyMatrix().Dimension(self.nCon, self.nVar)\
                        .Initialize(np.inf)

        # TODO: evaluate sparse

        return [0.0], [0], [0]

    def evaluate_dense(self, xi, lambda_, constr,
                       gradObj, constrJac, hess,
                       dmode):

        objval = 0.0

        xi_ = xi.as_array.ravel()

        if dmode >= 0:
            objval = xi_[0] * xi_[0] - 0.5*xi_[1]*xi_[1]
            constr.as_array[0] = xi_[0] - xi_[1]

        if dmode > 0:
            gradObj.as_array[0] = 2*xi_[0]
            gradObj.as_array[1] = -xi_[0]

            constrJac.as_array[0, 0] = 1.0
            constrJac.as_array[0, 1] = -1.0

        return objval

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
#opts.sparseQP = 1
opts.sparseQP = 0

opts.printLevel = 2


s = pyBlockSQP.PySQPStats('./')
m = pyBlockSQP.PySQPMethod(p, opts, s)

m.init()

ret = m.run(100)

m.finish()

if ret == 1:
    print("***Maximum number of iterations reached.***")

print("Primal solution:")
m.vars.xi.Print()

print("Dual solution")
m.vars.lambda_.Print()

print("Hessian approximation at the solution:")
for h in m.vars.hess:
    h.Print()
