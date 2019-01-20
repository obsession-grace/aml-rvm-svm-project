__author__ = "Johan, Felix Buettner"

from rvm.rvm import RVM
from svm.svc import SVC
import numpy as np

# Sample Settings.
number_samples_class0_mix1 = 25
number_samples_class0_mix2 = 25
number_samples_class1_mix1 = 25
number_samples_class1_mix2 = 25


def get_samples(number_samples, x, y, data, targets, t):
    for _ in range(number_samples):
        x_sample = x + np.random.normal(0, 0.15)
        y_sample = y + np.random.normal(0, 0.15)

        data.append([x_sample, y_sample])
        targets.append(t)

    return data, targets


# Create samples.
data = list()
targets = list()

x = 0
y = 0.4
data, targets = get_samples(number_samples_class0_mix1, x, y, data, targets, 0)

x = -1
y = 0.1
data, targets = get_samples(number_samples_class0_mix2, x, y, data, targets, 0)

x = 0.5
y = 0.7
data, targets = get_samples(number_samples_class1_mix1, x, y, data, targets, 1)

x = -0.5
y = 0.6
data, targets = get_samples(number_samples_class1_mix2, x, y, data, targets, 1)

# Convert to numpy array.
data = np.asarray(data)
targets = np.asarray(targets).flatten()

"""RVM for Classification (RVC)."""
print("------ RVC ------")
# RVC
rvm = RVM(method="classification", kernel="rbf", kernel_arg=1/(0.5**2))
rvm.train(data, targets)
rvm.plot(data, targets)

# Number of correct classifications
correct = rvm.get_number_of_correct_predictions(data, targets)
print("Correct classifications: ", correct, " out of ", targets.size)

# Number of relevance vectors
n_rv = rvm.get_number_of_relevance_vectors()
print("Number of RVs: ", n_rv)

"""SVM for Classification (SVC)."""
print("------ SVC ------")
# Small adjustments to the classes (class 0 -> class -1.0, class 1 -> class 1.0)
targets = targets.astype(float)
targets[targets==0] = -1.0
targets[targets==1] = 1.0

# SVC
svc = SVC(c=1e3, kernel="rbf", kernel_arg=1/(0.5**2))
svc.train(data, targets)
svc.plot(data, targets)

# Number of correct classifications
correct = svc.get_number_of_correct_predictions(data, targets)
print("Correct classifications: ", correct, " out of ", targets.size)

# Number of relevance vectors
n_rv = svc.get_number_of_relevance_vectors()
print("Number of RVs: ", n_rv)
