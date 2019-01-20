__author__ = "Felix Buettner"

from utils.kernels import possible_kernel
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import numpy as np


class SVR:
    
    """Support Vector Regression."""

    def __init__(self, c, epsilon=0.2, kernel="rbf", kernel_arg=0.1):
        # Penalty parameter C of the error term.
        self.C = c

        # Margin.
        self.epsilon = epsilon

        # Kernel settings.
        self.kernel = possible_kernel[kernel]
        self.kernel_arg = kernel_arg

        # Root-mean-square error.
        self.rms = None

    def train(self, X, y):
        """Fit the SVR model according to the given training data."""
        # Convert y into a two dimensional array.
        y = y.reshape(y.size, 1)

        # Number of data sets N.
        N = X.shape[0]

        # Gram matrix K.
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i,j] = self.kernel(X[i], X[j], self.kernel_arg)

        # Submatrices.
        I_tilde = np.eye(2*N)
        J = np.ones((N, 1))
        J_tilde = np.ones((2*N, 1))
        O_tilde = np.zeros((2*N, 1))

        # Dual problem - Matrices.
        P = cvxopt.matrix(np.bmat([[K, -K], [-K, K]]))
        q = cvxopt.matrix(np.bmat([[np.subtract(self.epsilon, y)],
                                   [np.add(self.epsilon, y)]]))
        G = cvxopt.matrix(np.bmat([[I_tilde], [-I_tilde]]))
        h = cvxopt.matrix(np.bmat([[np.multiply(self.C, J_tilde)], [O_tilde]]))
        A = cvxopt.matrix(np.bmat([J.T, -J.T]))
        b = cvxopt.matrix(0.0)

        # Solve the pair of primal and dual convex quadratic programs.
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-10  # default: 1e-7
        cvxopt.solvers.options['reltol'] = 1e-10  # default: 1e-6
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers.
        x_hat = np.array(sol['x']).reshape((2, -1))
        
        # Deal with numerical erros.
        idx = np.where(x_hat < 1e-3)
        x_hat[idx] = 0.0

        idx = np.where(x_hat > 0.999*self.C)
        x_hat[idx] = self.C

        # Define both alpha variables.
        alpha = x_hat[0]
        alpha_star = x_hat[1]

        # Find support vectors.
        self.sv_idx = np.where(np.logical_or(alpha > 0.0, alpha_star > 0.0))
        self.sv_X = X[self.sv_idx]
        self.sv_y = y[self.sv_idx]
        self.sv_alpha_dif = np.subtract(alpha[self.sv_idx],
                                        alpha_star[self.sv_idx])
        N_sv = len(self.sv_idx[0])  # Number of support vectors.

        # Calculate intercept term b.
        self.b = 0.0
        idx = np.where(np.logical_and(alpha > 0.0, alpha < self.C))
        N_alpha = len(idx[0])

        if N_alpha != 0:
            for i in idx[0]:
                sums = 0.0
                for j in range(N_sv):
                    K = self.kernel(self.sv_X[j], X[i], self.kernel_arg)
                    sol = np.multiply(self.sv_alpha_dif[j], K)
                    sums = np.add(sums, sol)
                self.b += y[i] - self.epsilon - sums

            self.b = np.divide(self.b, N_alpha)

        # Alternative: 
        # Take the solution for y (bias) of the dual convex quadratic program 
        # defined in cvxopt.solvers.qp
        # self.b = float(np.asarray(sol['y']))

    def predict(self, X_p):
        """Perform regression on samples in X_p."""
        # Save predictions in a list.
        predictions = []

        # Number of support vectors.   
        N_sv = len(self.sv_idx[0])

        # Evaluate new points.
        for x_p in X_p:
            sol = 0.0
            for i in range(N_sv):
                K = self.kernel(self.sv_X[i], x_p, self.kernel_arg)
                sol_tmp = np.multiply(self.sv_alpha_dif[i], K)
                sol = np.add(sol, sol_tmp)
            sol = np.add(sol, self.b)
            predictions.append(sol)

        # Convert into array.
        predictions = np.array(predictions)

        return predictions

    def plot(self, X, y, X_p, y_t=None):
        """Plot results, if d=1."""
        # Collapse arrays into one dimension.
        X = X.flatten()
        y = y.flatten()
        X_p = X_p.flatten()

        # Plot predictions.
        y_p = self.predict(X_p).flatten()
        plt.plot(X_p, y_p, color='royalblue', label='Prediction')

        # Plot true function if available.
        if y_t is not None:
            plt.plot(X_p, y_t, color='red', label='True function $f$')

        # Plot margin.
        y_plus = np.add(y_p, self.epsilon).flatten()
        y_minus = np.subtract(y_p, self.epsilon).flatten()
        plt.plot(X_p, y_plus, color='grey', label='Margin, $f \pm \epsilon$')
        plt.plot(X_p, y_minus, color='grey')
        plt.fill_between(X_p, y_minus, y_plus, color='lightgrey')

        # Plot support vectors.
        plt.plot(self.sv_X, self.sv_y, 'o', color='white', markersize=10, markeredgewidth=1.5, markeredgecolor='black', 
            label="Support vector")

        # Plot data.
        plt.plot(X, y, 'ko', markersize=4, label='Sample')

        # Add labels.
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        # Add title.
        plt.title('Support Vector Regression')

        # Add legend.
        plt.legend()

        # Show plot.
        plt.show()

    def calc_rms(self, X_p, y_t):
        """Calculates the root-mean-square error."""
        # Convert y_t into a two dimensional array.
        y_t = y_t.reshape(y_t.size, 1)

        # Calculate predictions.
        y_p = self.predict(X_p)

        # Calculate RMS.
        self.rms = np.sqrt(np.mean(np.power(np.subtract(y_p, y_t), 2)))

        return self.rms

    def get_number_of_relevance_vectors(self):
        """Get number of support vectors."""
        return len(self.sv_idx[0])

    def print_info(self):
        """Show interesting infos."""
        print(50*"-")
        print("SVR")
        print('Indices of support vectors = ', self.sv_idx)
        print('Number of support vectors = ', len(self.sv_idx[0]))
        print('b = ', self.b)
        if self.rms is not None:
            print('RMS = ', self.rms)
