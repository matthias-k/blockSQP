import pyBlockSQP
import numpy as np


class MyProblemspec(pyBlockSQP.PyProblemspec):
    def __init__(self, nVar, nCon, blockIdx, bl, bu, x):
        super(MyProblemspec, self).__init__()
        self.nVar = nVar
        self.nCon = nCon
        self.blockIdx = np.array(blockIdx, np.int32)

        print(nVar + nCon)
        self.bl.Dimension(nVar + nCon).Initialize(-np.inf)
        self.bu.Dimension(nVar + nCon).Initialize(np.inf)

        print(self.bl.numpy_data)

        self.bl.numpy_data[:] = bl
        self.bu.numpy_data[:] = bu

        self.objLo = -np.inf
        self.objUp = np.inf
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

        self.evaluate_dense(xi, lambda_, constrDummy, gradObjDummy, constrJac,
                            None, 1)

        return self.convertJacobian(constrJac, firstCall=True)

    def convertJacobian(self, constrJac, jacNz = None, jacIndRow = None, jacIndCol = None, firstCall=False):
        constrJac_ = constrJac.as_array
        if firstCall:
            nnz = 0
            for j in range(self.nVar):
                for i in range(self.nCon):
                    if np.abs(constrJac_[i, j]) < np.inf:
                        nnz += 1

            jacNz = np.empty(nnz)
            jacIndRow = np.empty(nnz, np.int32)
            jacIndCol = np.empty(self.nVar + 1, np.int32)

        else:
            nnz = jacIndCol[self.nVar]

        count = 0
        for j in range(self.nVar):
            jacIndCol[j] = count
            for i in range(self.nCon):
                if np.abs(constrJac_[i, j]) < np.inf:
                    jacNz[count] = constrJac_[i, j]
                    jacIndRow[count] = i
                    count += 1

        jacIndCol[self.nVar] = count
        if count != nnz:
            print("Error in convertJacobian: {} elements processed, should be {} elements!\n".format(count, nnz))

        return jacNz, jacIndRow, jacIndCol


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

    def evaluate_sparse(self, xi, lambda_, constr,
                        gradObj, jacNz, jacIndRow, jacIndCol,
                        hess, dmode):

        constrJac = pyBlockSQP.PyMatrix().Dimension(self.nCon, self.nVar).Initialize(np.inf)

        objval = self.evaluate_dense(xi, lambda_, constr,
                            gradObj, constrJac, hess, dmode)

        if dmode != 0:
            self.convertJacobian(constrJac, jacNz, jacIndRow, jacIndCol)

        return objval


nVar = 2
nCon = 1
x0 = np.array([10, 10.0])
bl = np.ones(nVar + nCon) * (-np.inf)
bu = np.ones(nVar + nCon) * np.inf

bl[nVar] = 0.0
bu[nVar] = 0.0

p = MyProblemspec(2, 1, [0, 1, 2], bl, bu, x0)
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
opts.sparseQP = 1
#opts.sparseQP = 0

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
