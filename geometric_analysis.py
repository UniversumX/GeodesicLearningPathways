# geometric_analysis.py

import numpy as np

def calculate_fisher_matrix(model, data):
    """
    Calculate the Fisher Information Matrix for a given model and data.
    Supports linear and non-linear models, including regression models with a jacobian method.

    Parameters:
    - model: str or callable, the best-fit model (e.g., 'linear', 'quadratic', callable regression model object).
    - data: ndarray, the input data used to fit the model.

    Returns:
    - fisher_matrix: ndarray, the calculated Fisher Information Matrix.
    """
    try:
        if isinstance(model, str):
            if model == 'linear':
                # Calculate Fisher matrix for linear regression
                if np.linalg.cond(data) > 1e10:
                    raise ValueError("Input data matrix is poorly conditioned or singular.")
                fisher_matrix = np.dot(data.T, data)
            elif model in ['quadratic', 'cubic', 'exponential']:
                # Non-linear Fisher information approximation
                gradients = np.gradient(data, axis=0)
                if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
                    raise ValueError("Gradient calculation for non-linear model produced invalid values.")
                fisher_matrix = np.dot(gradients.T, gradients)
            else:
                raise ValueError(f"Model type {model} is not supported.")
        elif callable(model):
            # Handle callable models like regression models
            if not hasattr(model, 'jacobian'):
                raise AttributeError("Callable model must implement a 'jacobian' method.")
            jacobian = model.jacobian(data)
            if jacobian is None or not isinstance(jacobian, np.ndarray):
                raise ValueError("The 'jacobian' method did not return a valid numpy array.")
            fisher_matrix = np.dot(jacobian.T, jacobian)
        else:
            raise TypeError("Model must be a string (predefined) or a callable (custom).")
    except Exception as e:
        print(f"Error calculating Fisher Information Matrix: {e}")
        raise

    return fisher_matrix

def compute_geometric_structures(fisher_matrix):
    """
    Compute geometric structures (e.g., Christoffel symbols, Riemann tensor) from Fisher Information Matrix.

    Parameters:
    - fisher_matrix: ndarray, the Fisher Information Matrix.

    Returns:
    - structures: dict, contains calculated geometric structures.
    """
    try:
        # Check for numerical stability before inversion
        if np.linalg.cond(fisher_matrix) > 1e10:
            raise ValueError("Fisher matrix is nearly singular and may result in unstable inversions.")

        # Inverse Fisher matrix as a metric tensor
        metric_tensor = np.linalg.inv(fisher_matrix)

        # Refined calculation for Christoffel symbols
        christoffel_symbols = np.zeros(metric_tensor.shape + (metric_tensor.shape[0],))
        for i in range(metric_tensor.shape[0]):
            for j in range(metric_tensor.shape[0]):
                for k in range(metric_tensor.shape[0]):
                    christoffel_symbols[i, j, k] = 0.5 * (
                        np.gradient(metric_tensor[:, j], axis=0)[k] +
                        np.gradient(metric_tensor[:, k], axis=0)[j] -
                        np.gradient(metric_tensor[:, :], axis=1)[i, k]
                    )

        # Refined computation for Riemann tensor
        riemann_tensor = np.zeros(metric_tensor.shape + (metric_tensor.shape[0], metric_tensor.shape[0]))
        for i in range(metric_tensor.shape[0]):
            for j in range(metric_tensor.shape[0]):
                for k in range(metric_tensor.shape[0]):
                    for l in range(metric_tensor.shape[0]):
                        riemann_tensor[i, j, k, l] = (
                            np.gradient(christoffel_symbols[:, j, l], axis=0)[k]
                            - np.gradient(christoffel_symbols[:, j, k], axis=0)[l]
                            + np.sum(
                                christoffel_symbols[i, :, k] * christoffel_symbols[:, j, l]
                                - christoffel_symbols[i, :, l] * christoffel_symbols[:, j, k]
                            )
                        )

        # Collect results
        structures = {
            "metric_tensor": metric_tensor,
            "christoffel_symbols": christoffel_symbols,
            "riemann_tensor": riemann_tensor,
        }
    except Exception as e:
        print(f"Error computing geometric structures: {e}")
        raise

    return structures

