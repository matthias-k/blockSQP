#ifndef IPROBLEMSPEC_H_
#define IPROBLEMSPEC_H_

#include "blocksqp_problemspec.hpp"
#include "blocksqp_matrix.hpp"
#include "pyBlockSQP_api.h"

namespace blockSQP {

class IProblemspec: public Problemspec {
public:
    PyObject *m_obj;

	IProblemspec(PyObject *obj);
	virtual ~IProblemspec();
    /// Evaluate objective, constraints, and derivatives (dense version).
    virtual void evaluate( const Matrix &xi,        ///< optimization variables
					   const Matrix &lambda,    ///< Lagrange multipliers
					   double *objval,          ///< objective function value
					   Matrix &constr,          ///< constraint function values
					   Matrix &gradObj,         ///< gradient of objective
					   Matrix &constrJac,       ///< constraint Jacobian (dense)
					   SymMatrix *&hess,        ///< Hessian of the Lagrangian (blockwise)
					   int dmode,               ///< derivative mode
					   int *info                ///< error flag
					   );

};
	
}

#endif /* IPROBLEMSPEC_H_ */
