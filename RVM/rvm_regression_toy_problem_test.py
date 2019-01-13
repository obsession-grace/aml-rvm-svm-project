__author__ = "Johan"

import random
import matplotlib.pyplot as plt
import numpy as np
from rvm import RVM

def get_samples(number_samples, x_min, x_max, noise):
    x_samples = []
    y_samples = []
    x_samples = np.linspace(x_min, x_max, number_samples)
    for x in x_samples:
        y = np.sin(x)/x
        y_samples.append(y)
    if noise==True:
        y_samples = y_samples + np.random.normal(0, 0.2, number_samples)
    return x_samples, y_samples

def get_predictions(x_input):
    t_predictions = []
    for x in x_input:
        t_predictions.append(rvm.make_prediction([[x]]))
    return t_predictions

def get_relevance_vectors():
    relevance_vectors_x = []
    relevance_vectors_y = []
    for relevance_vector in rvm.relevance_vectors:
        relevance_vectors_x.append(relevance_vector[0])
        for i in range (len(x_samples)):
            if x_samples[i] == relevance_vector[0]:
                relevance_vectors_y.append(y_samples[i])
                break
    return relevance_vectors_x, relevance_vectors_y

#Get samples
x_min = -10
x_max = 10
number_samples = 100
noise = True
x_samples, y_samples = get_samples(number_samples, x_min, x_max, noise)

x_samples_temp = []
for x in x_samples:
    x_samples_temp.append([x])
x_samples = x_samples_temp
rvm = RVM(method="regression", kernel_type="linear")
rvm.train(x_samples, y_samples)

#Get predictive mean
x_input = np.linspace(x_min, x_max, 100)
t_predictions = get_predictions(x_input)

#Get relevance vectors
relevance_vectors_x, relevance_vectors_y = get_relevance_vectors()

#Plot relevance vectors
plt.plot(relevance_vectors_x, relevance_vectors_y, 'ko', markersize=10, label="Relevance vector")
plt.plot(relevance_vectors_x, relevance_vectors_y, 'wo', markersize=6)
#Plot samples
plt.plot(x_samples, y_samples, 'ko', markersize=2, label="Sample")
plt.xlim([x_min, x_max])
#Plot true function.
y_true = []
for x in x_input:
    y_true.append(np.sin(x)/x)
plt.plot(x_input, y_true, 'r', markersize=2, linestyle=":", label="sinc(x)")
#Plot predictive mean.
plt.plot(x_input, t_predictions, 'royalblue', markersize=2, label="Predictive mean")
plt.legend()
plt.show()
