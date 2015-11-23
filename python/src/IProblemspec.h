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

    /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (dense version).
    virtual void initialize( Matrix &xi,            ///< optimization variables
                             Matrix &lambda,        ///< Lagrange multipliers
                             Matrix &constrJac      ///< constraint Jacobian (dense)
                             );

    /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (sparse version).
    //virtual void initialize( Matrix &xi,            ///< optimization variables
    //                         Matrix &lambda,        ///< Lagrange multipliers
    //                         double *&jacNz,        ///< nonzero elements of constraint Jacobian
    //                         int *&jacIndRow,       ///< row indices of nonzero elements
    //                         int *&jacIndCol        ///< starting indices of columns
    //                         );

};
	
}

#endif /* IPROBLEMSPEC_H_ */
