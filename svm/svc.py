__author__ = "Felix Buettner"

from utils.kernels import possible_kernel
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import numpy as np

             
class SVC:

    """Support Vector Classifier."""

    def __init__(self, c=0.0, kernel="pol", kernel_arg=3):
        # Penalty parameter C of the error term.
        self.C = c

        # Kernel settings.
        self.kernel = possible_kernel[kernel]
        self.kernel_arg = kernel_arg

    def train(self, X, y):
        """Fit the SVM model according to the given training data."""
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
        I = np.eye(N)
        J = np.ones((N, 1))
        O = np.zeros((N, 1))
        diag_y = np.diag(y.flatten())

        # Dual problem - Matrices.
        P = cvxopt.matrix(np.dot(diag_y, np.dot(K, diag_y)))
        q = cvxopt.matrix(-J)
        if self.C > 0.0:
            G = cvxopt.matrix(np.bmat([[I], [-I]]))
            h = cvxopt.matrix(np.bmat([[np.multiply(self.C, J)], [O]]))
        else:
            G = cvxopt.matrix(np.bmat([-I]))
            h = cvxopt.matrix(np.bmat([O]))
        A = cvxopt.matrix(y.T)
        b = cvxopt.matrix(0.0)

        # Solve the pair of primal and dual convex quadratic programs.
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-7  # default: 1e-7
        cvxopt.solvers.options['reltol'] = 1e-6  # default: 1e-6
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers.
        alpha = np.array(sol['x'])

        # Deal with numerical erros.
        idx = np.where(alpha < 1e-5)
        alpha[idx] = 0.0

        if self.C > 0.0:
            idx = np.where(alpha > 0.999*self.C)
            alpha[idx] = self.C

        # Find support vectors.
        self.sv_idx = np.where(alpha > 0.0)[0]
        self.sv_X = X[self.sv_idx]
        self.sv_y = y[self.sv_idx]
        self.sv_alpha = alpha[self.sv_idx]
        N_sv = len(self.sv_idx)  # Number of support vectors.

        # Calculate intercept term b.
        self.b = 0.0
        for i in range(N_sv):
            sums = 0.0
            for j in range(N_sv):
                k = K[self.sv_idx[i], self.sv_idx[j]]
                ay = np.multiply(self.sv_alpha[j], self.sv_y[j])
                ayk = np.multiply(ay, k)
                sums = np.add(sums, ayk)
            self.b += self.sv_y[i] - sums
        self.b = np.divide(self.b, N_sv) 

    def predict(self, X_p):
        """Perform classification on samples in X_p."""
        # Save predictions in a list.
        predictions = []

        # Save the values in a list.
        values = []

        # Number of support vectors.   
        N_sv = len(self.sv_idx)

        # Evaluate new points.
        for x_p in X_p:
            sol = 0.0
            for i in range(N_sv):
                k = self.kernel(self.sv_X[i], x_p, self.kernel_arg)
                ay = np.multiply(self.sv_alpha[i], self.sv_y[i])
                ayk = np.multiply(ay, k)
                sol = np.add(sol, ayk)
            sol = np.add(sol, self.b)
            values.append(sol)
            sol = np.sign(sol)
            predictions.append(sol)

        # Convert to numpy array.
        predictions = np.array(predictions)
        values = np.array(values)

        return [predictions, values]

    def plot(self, X, y):
        """Plot results, if d=2."""
        # Plot support vectors.
        plt.plot(self.sv_X[:,0], self.sv_X[:,1], 'o', color='white', 
            markersize=10, markeredgewidth=1.5, markeredgecolor='black',
            label='Support vectors')

        # Plot data.
        plt.plot(X[y==1][:,0], X[y==1][:,1], "bo", markersize=4, 
            label='Samples of class 1')
        plt.plot(X[y==-1][:,0], X[y==-1][:,1], "ro", markersize=4,
            label='Samples of class 0')

        # Plot decision boundary.
        x1_min = np.round(np.amin(X[:,0])).astype(int)
        x1_max = np.round(np.amax(X[:,0])).astype(int)
        x2_min = np.round(np.amin(X[:,1])).astype(int)
        x2_max = np.round(np.amax(X[:,1])).astype(int)

        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50),
                             np.linspace(x2_min, x2_max, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = self.predict(X)[1].reshape(X1.shape)

        plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        # Add labels.
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

        # Add title.
        plt.title('Support Vector Classifier')

        # Add legend.
        plt.legend()

        # Show plot.
        plt.show()

    def get_number_of_correct_predictions(self, X_p, y_t):
        """Get number of correct predictions."""
        y_p = self.predict(X_p)[0]
        y_p = y_p.reshape(1, -1)
        correct = np.sum(y_p == y_t)

        return correct

    def get_number_of_relevance_vectors(self):
        """Get number of support vectors."""
        return self.sv_idx.size

    def print_info(self):
        """Show interesting infos."""
        print(50*"-")
        print("SVC")
        print('Indices of support vectors = ', self.sv_idx)
        print('Number of support vectors = ', self.sv_idx.size)
        print('b = ', self.b)
