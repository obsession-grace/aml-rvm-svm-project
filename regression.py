__author__ = "Johan, Felix Buettner"

from rvm.rvm import RVM
from svm.svr import SVR
import numpy as np


"""Data."""
# Sample Settings
x_min = -10
x_max = 10
number_samples = 100
noise = True


def get_samples(number_samples, x_min, x_max, noise):
    x_samples = np.linspace(x_min, x_max, number_samples)
    y_samples = []
    for x in x_samples:
        y = np.sin(x)/x
        y_samples.append(y)
    if noise==True:
        y_samples = y_samples + np.random.normal(0, 0.2, number_samples)

    # Save as array.
    x_samples = np.array(x_samples)
    y_samples = np.array(y_samples).flatten()

    return x_samples, y_samples


# Create samples.
x_samples, y_samples = get_samples(number_samples, x_min, x_max, noise)

# True function.
x_input = np.linspace(x_min, x_max, 100)
y_true = []
for x in x_input:
    y_true.append(np.sin(x)/x)

y_true = np.array(y_true)

"""RVM for Regression."""
print("------ RVR ------")
# RVM for Regression
rvm = RVM(method="regression", kernel="rbf", kernel_arg=0.11111)
rvm.train(x_samples, y_samples)
rvm.plot(x_samples, y_samples, x_input, y_true)

# Root-mean-square error
rms = rvm.calc_rms(x_input, y_true)
print("RMS: ", rms)

# Number of relevance vectors
n_rv = rvm.get_number_of_relevance_vectors()
print("Number of RVs: ", n_rv)

"""SVM for Regression (SVR)."""
print("------ SVR ------")
# SVR
svr = SVR(c=1e3, kernel="rbf", kernel_arg=0.1)
svr.train(x_samples, y_samples)
svr.plot(x_samples, y_samples, x_input, y_true)

# Root-mean-square error
rms = svr.calc_rms(x_input, y_true)
print("RMS: ", rms)

# Number of relevance vectors
n_rv = svr.get_number_of_relevance_vectors()
print("Number of RVs: ", n_rv)