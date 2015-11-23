#include "IProblemspec.h"

namespace blockSQP {

IProblemspec::IProblemspec(PyObject *obj): m_obj(obj) {
    // Provided by "pyBlockSQP_api.h"
	if (import_pyBlockSQP()) {
	} else {
        Py_XINCREF(this->m_obj);
	}
}

IProblemspec::~IProblemspec() {
    Py_XDECREF(this->m_obj);
}


/// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (dense version).
void IProblemspec::initialize( Matrix &xi,            ///< optimization variables
                               Matrix &lambda,        ///< Lagrange multipliers
                               Matrix &constrJac      ///< constraint Jacobian (dense)
                              ){
    if (this->m_obj) {
        cy_call_initialize_dense(this->m_obj, xi, lambda, constrJac);
    }
}

/// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (sparse version).
void IProblemspec::initialize( Matrix &xi,            ///< optimization variables
                               Matrix &lambda,        ///< Lagrange multipliers
                               double *&jacNz,        ///< nonzero elements of constraint Jacobian
                               int *&jacIndRow,       ///< row indices of nonzero elements
                               int *&jacIndCol        ///< starting indices of columns
                               ){
    if (this->m_obj) {
        cy_call_initialize_sparse(this->m_obj, xi, lambda, jacNz, jacIndRow, jacIndCol);
    }
}



/// Evaluate objective, constraints, and derivatives (dense version).
void IProblemspec::evaluate( const Matrix &xi,        ///< optimization variables
					   const Matrix &lambda,    ///< Lagrange multipliers
					   double *objval,          ///< objective function value
					   Matrix &constr,          ///< constraint function values
					   Matrix &gradObj,         ///< gradient of objective
					   Matrix &constrJac,       ///< constraint Jacobian (dense)
					   SymMatrix *&hess,        ///< Hessian of the Lagrangian (blockwise)
					   int dmode,               ///< derivative mode
					   int *info                ///< error flag
					   ){
    if (this->m_obj) {
	    int error;
		// Call a virtual overload, if it exists
		cy_call_evaluate_func_and_grad(this->m_obj, (char*)"evaluate", &error,
						               xi, lambda, objval, constr, gradObj, constrJac);
	}
}
}
