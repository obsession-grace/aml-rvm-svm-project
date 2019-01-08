__author__ = "Johan"

import math
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


class RVM():

    def __init__(self, method, kernel_type):
        self.method = method
        self.kernel_type = kernel_type
        self.bias_was_pruned = False
        self.relevance_vectors = None

    def compute_kernel(self, x, y):
        phi = []
        if self.kernel_type == 'linear':
            for x_m in x:
                row = []
                x_m = x_m[0]
                for x_n in y:
                    x_n = x_n[0]
                    K = 1 + x_m*x_n + x_m*x_n*min(x_m, x_n) - ((x_m+x_n)/2)*min(x_m, x_n)*min(x_m, x_n) + (min(x_m, x_n)*min(x_m, x_n)*min(x_m, x_n))/3
                    row.append(K)
                phi.append(row)
            phi = np.asarray(phi)
        elif self.kernel_type == 'gaussian':
            for x_m in x:
                row = []
                x_m = np.array(x_m)
                for x_n in y:
                    x_n = np.array(x_n)
                    #K = math.exp(((-0.5)**(-2)) * (np.linalg.norm(x_m - x_n))**2)
                    K = math.exp(-(1/len(x_m))*(np.linalg.norm(x_m - x_n))**2)
                    row.append(K)
                phi.append(row)
            phi = np.asarray(phi) 
        if not self.bias_was_pruned:
            #Add 1 to each row in the phi matrix (bias basis function)
            phi = np.append(np.ones((phi.shape[0], 1)), phi, axis=1)
        return phi

    def define_prior(self):
        number_of_basis_functions = self.phi.shape[1]
        #Fill the mean vector with 0s (zero-mean)
        self.mean = np.zeros(number_of_basis_functions)
        #Initiate alpha as a vector of N+1 hyperparameters
        #self.alpha = self.alpha * np.ones(number_of_basis_functions)
        self.alpha = 1/number_of_basis_functions * np.ones(number_of_basis_functions)
        #Beta = sigma^-2
        self.beta = 1/number_of_basis_functions

    def sigmoid_function_transform(self, dot_product):
        #Logistic sigmoid function: 1/(1+np.exp(-np.dot(phi, m)))
        return expit(dot_product)

    def compute_log_posterior(self, mean, phi, t, alpha):
        y = self.sigmoid_function_transform(np.dot(phi, mean))
        log_posterior = 0
        np.seterr(divide = "ignore")
        for c in self.classes:
            log_y = np.log(y[t == c])
            log_y[np.isneginf(log_y)] = 0
            log_1_minus_y = np.log(1 - y[t == c])
            log_1_minus_y[np.isneginf(log_1_minus_y)] = 0
            log_posterior = log_posterior + np.sum(c * log_y + (1 - c) * log_1_minus_y)
        np.seterr(divide = "warn")
        A = np.diag(alpha)
        log_posterior = log_posterior - 0.5 * np.dot((mean).T, np.dot(A, mean))
        #We will minimize -log_posterior instead of maximizing log_posterior.
        log_posterior = -log_posterior
        return log_posterior

    def compute_jacobian(self, mean, phi, t, alpha):
        y = self.sigmoid_function_transform(np.dot(phi, mean))
        A = np.diag(alpha)
        jacobian = np.dot(mean, A) - np.dot((phi).T, (t - y))
        return jacobian

    def compute_hessian(self, mean, phi, t, alpha):
        A = np.diag(alpha)
        y = self.sigmoid_function_transform(np.dot(phi, mean))
        B = np.diag(y * (1 - y))
        hessian = -(np.dot((phi).T, np.dot(B, phi)) + A)
        return -hessian

    def compute_posterior(self):
        if self.method == "regression":
            A = np.diag(self.alpha)
            self.sigma = np.linalg.inv(self.beta * np.dot(self.phi.T, self.phi) + A)
            self.mean = self.beta * np.dot(self.sigma, np.dot(self.phi.T, self.t))
        elif self.method == "classification":
            maximize = minimize(fun = self.compute_log_posterior, x0 = self.mean, args = (self.phi, self.t, self.alpha),
                              method = 'Newton-CG', jac = self.compute_jacobian, hess = self.compute_hessian)
            self.mean = maximize.x
            self.sigma = np.linalg.inv(self.compute_hessian(self.mean, self.phi, self.t, self.alpha))

    def optimize_hyperparameters(self):
            self.gamma = 1 - self.alpha * np.diag(self.sigma)
            self.alpha = self.gamma / (self.mean ** 2)
            number_of_samples = (self.t).shape[0]
            self.beta = (number_of_samples - np.sum(self.gamma)) / (np.linalg.norm(self.t - np.dot(self.phi, self.mean)) ** 2)

    def prune_parameters(self, indices_to_keep):
        self.phi = self.phi[:, indices_to_keep]
        self.mean = self.mean[indices_to_keep]
        self.gamma = self.gamma[indices_to_keep]
        self.previous_alpha = self.previous_alpha[indices_to_keep]
        self.alpha = self.alpha[indices_to_keep]
        self.sigma = self.sigma[:, indices_to_keep]

    def update_relevance_vectors(self, indices_to_keep):
        if self.bias_was_pruned:
            self.relevance_vectors = self.relevance_vectors[indices_to_keep]
        else:
            #Number of relevance vectors (input points) < number of alpha values
            #until the alpha value corresponding to the bias (basis function) is pruned
            self.relevance_vectors = self.relevance_vectors[indices_to_keep[1:]]
            #The bias has index 0
            if not indices_to_keep[0]:
                self.bias_was_pruned = True

    def prune_basis_functions(self):
        infinity = 1000
        #Prune all large alpha values
        indices_to_keep = self.alpha < infinity
        #Update parameters
        self.prune_parameters(indices_to_keep)
        #Update relevance vectors
        self.update_relevance_vectors(indices_to_keep)

    def convergence(self, iteration):
        max_number_of_iterations = 5000
        convergence_criteria = 0.005
        delta = np.amax(np.absolute(self.previous_alpha - self.alpha))
        if (iteration > 500 and delta < convergence_criteria) or iteration == max_number_of_iterations:
            return True
        self.previous_alpha = self.alpha
        return False

    def train(self, x, t):
        if self.method=="classification":
            self.classes = np.unique(t)
        x = np.asarray(x)
        t = np.asarray(t)
        #Let all training data points (samples) be relevance vectors initially
        self.relevance_vectors = x
        self.t = t
        #Phi is the N * (N + 1) design matrix
        self.phi = self.compute_kernel(x, x)
        #Define Gaussian prior distribution over the weights
        self.define_prior()
        
        iteration = 0
        self.previous_alpha = self.alpha
        while True:
            iteration = iteration + 1
            #Compute the posterior distribution over the weights
            self.compute_posterior()
            #Optimize hyperparameters
            self.optimize_hyperparameters()
            #Prune the basis functions
            self.prune_basis_functions()
            #Check convergence criteria
            if self.convergence(iteration):
                break
        return self

    def compute_class_probabilities(self, x):
        phi = self.compute_kernel(x, self.relevance_vectors)
        class0_probability = 1 - self.sigmoid_function_transform(np.dot(phi, self.mean))
        class1_probability = 1 - class0_probability
        return class0_probability, class1_probability

    def make_prediction(self, x):
        if self.method == "regression":
            phi = self.compute_kernel(x, self.relevance_vectors)
            y = np.dot(phi, self.mean)
            return y
        elif self.method == "classification":
            class0_probability, class1_probability = self.compute_class_probabilities(x)
            class_prediction = np.empty(class0_probability.shape[0], dtype=np.int)
            class_prediction.fill(self.classes[0])
            class_prediction[class1_probability > class0_probability] = self.classes[1]
            return class_prediction
