__author__ = "Johan, Felix Buettner"

from utils.kernels import possible_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


class RVM:

    """Relevance Vector Machine."""

    def __init__(self, method, kernel, kernel_arg=None):
        # Bias
        self.bias_was_pruned = False

        # Classification or regression
        self.method = method

        # Kernel settings.
        self.kernel = possible_kernel[kernel]
        self.kernel_arg = kernel_arg

        # Other
        self.classes = None
        self.relevance_vectors = None
        self.t = None
        self.phi = None
        self.mean = None
        self.sigma = None
        self.alpha = None
        self.previous_alpha = None
        self.gamma = None
        self.beta = None
        self.idx = None

    def train(self, x, t):
        """Fit the RVM model according to the given training data."""
        if self.method == "classification":
            self.classes = np.unique(t)
        x = np.asarray(x)
        t = np.asarray(t)
        # Let all training data points (samples) be relevance vectors initially
        self.relevance_vectors = x
        self.t = t
        # Phi is the N * (N + 1) design matrix
        self.phi = self.compute_kernel(x, x)
        # Define Gaussian prior distribution over the weights
        self.define_prior(x, t)

        iteration = 0
        self.previous_alpha = self.alpha
        while True:
            iteration = iteration + 1
            # Compute the posterior distribution over the weights
            self.compute_posterior()
            # Optimize hyperparameters
            self.optimize_hyperparameters()
            # Prune the basis functions
            self.prune_basis_functions()
            # Check convergence criteria
            if self.convergence(iteration):
                if self.relevance_vectors.size < self.idx.size:
                    # Remove index of bias term
                    self.idx = self.idx[0:self.idx.size-1]
                break

    def compute_kernel(self, x, y):
        """Compute Kernel."""
        # Number of data sets N in x
        Nx = x.shape[0]

        # Number of data sets N in y
        Ny = y.shape[0]

        # Calculate phi
        phi = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                phi[i, j] = self.kernel(x[i], y[j], self.kernel_arg)

        if not self.bias_was_pruned:
            # Add 1 to each row in the phi matrix (bias basis function)
            phi = np.append(phi, np.ones((Nx, 1)), axis=1)

        return phi

    def define_prior(self, x, t):
        """Define prior through initial guesses."""
        # Number of data sets
        N = x.shape[0]

        # Number of basis functions
        number_of_basis_functions = self.phi.shape[1]

        # Fill the mean vector with 0s (zero-mean)
        self.mean = np.zeros(number_of_basis_functions)

        # Initiate alpha as a vector of N+1 hyperparameters
        # Initial guess of alpha = (1/N)**2
        self.alpha = np.ones(number_of_basis_functions) * ((1 / N) ** 2)

        if self.method == "regression":
            # Initial guess of 10% noise
            e = np.std(t) * 0.1
            # Beta = sigma^-2
            self.beta = 1 / (e ** 2)
        elif self.method == "classification":
            self.beta = 0

    def compute_posterior(self):
        """Compute posterior."""
        if self.method == "regression":
            A = np.diag(self.alpha)
            self.sigma = np.linalg.inv(self.beta * np.dot(self.phi.T,
                                                          self.phi) + A)
            self.mean = self.beta * np.dot(self.sigma, np.dot(self.phi.T,
                                                              self.t))
        elif self.method == "classification":
            maximize = minimize(fun=self.compute_log_posterior,
                                x0=self.mean,
                                args=(self.phi, self.t, self.alpha),
                                method='Newton-CG',
                                jac=self.compute_jacobian,
                                hess=self.compute_hessian)
            self.mean = maximize.x
            self.sigma = np.linalg.inv(self.compute_hessian(self.mean, self.phi,
                                                            self.t, self.alpha))

    def compute_log_posterior(self, mean, phi, t, alpha):
        """Compute log posterior."""
        y = self.sigmoid_function_tr(np.dot(phi, mean))
        log_posterior = 0
        np.seterr(divide="ignore")
        for c in self.classes:
            log_y = np.log(y[t == c])
            log_y[np.isneginf(log_y)] = 0
            log_1_minus_y = np.log(1 - y[t == c])
            log_1_minus_y[np.isneginf(log_1_minus_y)] = 0
            log_posterior = log_posterior \
                            + np.sum(c * log_y + (1 - c) * log_1_minus_y)
        np.seterr(divide="warn")
        A = np.diag(alpha)
        log_posterior = log_posterior - 0.5 * np.dot(mean.T, np.dot(A, mean))
        # We will minimize -log_posterior instead of maximizing log_posterior.
        log_posterior = -log_posterior
        return log_posterior

    def compute_jacobian(self, mean, phi, t, alpha):
        """Compute Jacobian."""
        y = self.sigmoid_function_tr(np.dot(phi, mean))
        A = np.diag(alpha)
        jacobian = np.dot(mean, A) - np.dot(phi.T, (t - y))
        return jacobian

    def compute_hessian(self, mean, phi, t, alpha):
        """Compute hessian matrix."""
        A = np.diag(alpha)
        y = self.sigmoid_function_tr(np.dot(phi, mean))
        B = np.diag(y * (1 - y))
        hessian = -(np.dot(phi.T, np.dot(B, phi)) + A)
        return -hessian

    def sigmoid_function_tr(self, dot_product):
        """Transform sigmoid function."""
        # Logistic sigmoid function: 1/(1+np.exp(-np.dot(phi, m)))
        return expit(dot_product)

    def optimize_hyperparameters(self):
        """Optimize hyper-parameters."""
        self.gamma = 1 - np.multiply(self.alpha, np.diag(self.sigma))
        self.alpha = self.gamma / (self.mean ** 2)
        n_samples = self.t.shape[0]
        self.beta = np.divide((n_samples - np.sum(self.gamma)),
                              (np.linalg.norm(self.t - np.dot(self.phi,
                                                              self.mean))**2))

    def convergence(self, iteration):
        """Check if alpha converges."""
        max_number_of_iterations = 5000
        convergence_criteria = 1e-3
        delta = np.amax(np.absolute(np.log(self.previous_alpha) -
                                    np.log(self.alpha)))
        
        if delta < convergence_criteria or \
                iteration == max_number_of_iterations:
            return True
        else:
            self.previous_alpha = self.alpha
            return False

    def prune_basis_functions(self):
        """Pruning."""
        infinity = 1e+9
        # Prune all large alpha values
        indices_to_keep = self.alpha < infinity

        # Save indices which stay.
        if self.idx is None:
            self.idx = np.where(self.alpha < infinity)[0]
        else:
            idx0 = np.where(self.alpha < infinity)[0]
            tmp = []
            for i in idx0:
                tmp.append(self.idx[i])
            self.idx = np.array(tmp)

        # Update parameters
        self.prune_parameters(indices_to_keep)
        # Update relevance vectors
        self.update_relevance_vectors(indices_to_keep)

    def prune_parameters(self, indices_to_keep):
        """Prune parameters."""
        self.phi = self.phi[:, indices_to_keep]
        self.mean = self.mean[indices_to_keep]
        self.gamma = self.gamma[indices_to_keep]
        self.previous_alpha = self.previous_alpha[indices_to_keep]
        self.alpha = self.alpha[indices_to_keep]
        self.sigma = self.sigma[:, indices_to_keep]

    def update_relevance_vectors(self, indices_to_keep):
        """Update relevance vectors."""
        if self.bias_was_pruned:
            self.relevance_vectors = self.relevance_vectors[indices_to_keep]
        else:
            # Number of relevance vectors (input points) < number of alpha
            # values until the alpha value corresponding to the bias (basis
            # function) is pruned
            indices = indices_to_keep[0:len(indices_to_keep)-1]
            self.relevance_vectors = self.relevance_vectors[indices]

            # The bias is the last item
            if not indices_to_keep[-1]:
                self.bias_was_pruned = True

    def compute_class_probabilities(self, x):
        """Compute class probabilities."""
        phi = self.compute_kernel(x, self.relevance_vectors)
        class0_probability = 1 - self.sigmoid_function_tr(np.dot(phi,
                                                                 self.mean))
        class1_probability = 1 - class0_probability

        return [class0_probability, class1_probability]

    def make_prediction(self, x):
        """Make predictions."""
        if self.method == "regression":
            phi = self.compute_kernel(x, self.relevance_vectors)
            y = np.dot(phi, self.mean)
            return y
        elif self.method == "classification":
            class0_probability, class1_probability = \
                self.compute_class_probabilities(x)
            class_prediction = \
                np.empty(class0_probability.shape[0], dtype=np.int)
            class_prediction.fill(self.classes[0])
            class_prediction[class1_probability > class0_probability] \
                = self.classes[1]
            return class_prediction

    def plot(self, x, y, x_p=None, y_t=None):
        """Plot results."""
        if self.method == "regression":
            self.plot_regression(x, y, x_p, y_t)
        elif self.method == "classification":
            self.plot_classification(x, y)

    def plot_regression(self, x, y, x_p, y_t=None):
        """Plot regression, if d=1."""
        # Plot predictive mean.
        y_p = self.make_prediction(x_p)
        plt.plot(x_p, y_p, 'royalblue', markersize=2, label="Predictive mean")

         # Plot true function if available.
        if y_t is not None:
            plt.plot(x_p, y_t, color='red', label='True function')

        # Plot support vectors.
        sv_x = self.relevance_vectors
        sv_y = list()
        for i in self.idx:
            sv_y.append(y[i])
        sv_y = np.array(sv_y)
        plt.plot(sv_x, sv_y, 'o', color='white', markersize=10,
                 markeredgewidth=1.5, markeredgecolor='black',
                 label="Relevance vector")

        # Plot data.
        plt.plot(x, y, 'ko', markersize=4, label='Sample')

        # Add labels.
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        # Add title.
        plt.title('Relevance Vector Regression')

        # Add legend.
        plt.legend()

        # Show plot.
        plt.show()

    def plot_classification(self, x, t):
        """Plot classification, if d=2."""
        # Plot support vectors.
        sv_x = self.relevance_vectors[:, 0]
        sv_y = self.relevance_vectors[:, 1]
        plt.plot(sv_x, sv_y, 'o', color='white', markersize=10,
                 markeredgewidth=1.5, markeredgecolor='black',
                 label="Relevance vector")

        # Plot data.
        plt.plot(x[t == 1][:, 0], x[t == 1][:, 1], "bo", markersize=4,
                 label='Samples of class 1')
        plt.plot(x[t == 0][:, 0], x[t == 0][:, 1], "ro", markersize=4,
                 label='Samples of class 0')

        # Plot decision boundary.
        x1_min = min(x[:, 0])
        x1_max = max(x[:, 0])
        x2_min = min(x[:, 1])
        x2_max = max(x[:, 1])
        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 500),
                             np.linspace(x2_min, x2_max, 500))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        probabilities = self.compute_class_probabilities(X)

        # Find indices where p(x|class 0) approx p(x|class 1)
        dif = np.abs(probabilities[0] - probabilities[1])
        idx = np.where(dif < 0.001)
        X_b = X[idx]
        X_b = X_b[X_b[:, 0].argsort()]

        plt.plot(X_b[:,0], X_b[:,1], 'k', linestyle=":",
                 label="Decision boundary")

        # Add labels.
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

        # Add title.
        plt.title('Relevance Vector Classifier')

        # Add legend.
        plt.legend()

        # Show plot.
        plt.show()

    def calc_rms(self, x_p, y_t):
        """Calculates the root-mean-square error (Regression)."""
        # Convert y_t into a two dimensional array.
        y_t = y_t.reshape(y_t.size, 1)

        # Calculate predictions.
        y_p = self.make_prediction(x_p)
        y_p = y_p.reshape(y_p.size, 1)

        # Calculate RMS.
        self.rms = np.sqrt(np.mean(np.power(np.subtract(y_p, y_t), 2)))

        return self.rms

    def get_number_of_correct_predictions(self, x_p, y_t):
        """Get number of correct predictions (Classification)."""
        cl0 = self.compute_class_probabilities(x_p)[0]
        idx0 = np.where(cl0 >= 0.5)
        idx1 = np.where(cl0 < 0.5)
        cl0[idx0] = 0
        cl0[idx1] = 1
        y_p = cl0

        y_p = y_p.reshape(1, -1)
        correct = np.sum(y_p == y_t)

        return correct

    def get_number_of_relevance_vectors(self):
        """Get number of support vectors."""
        return len(self.relevance_vectors)
