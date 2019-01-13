__author__ = "Johan"

import numpy as np
import matplotlib.pyplot as plt
from rvm import RVM

number_samples = 100
number_samples_class0_mix1 = 25
number_samples_class0_mix2 = 25
number_samples_class1_mix1 = 25
number_samples_class1_mix2 = 25
data = []
targets = []
data0_x = []
data0_y = []
x = 0
y = 0.4
for _ in range(number_samples_class0_mix1):
    x_sample = x + np.random.normal(0, 0.15)
    y_sample = y + np.random.normal(0, 0.15)
    data0_x.append(x_sample)
    data0_y.append(y_sample)
    data.append([x_sample, y_sample])
    targets.append(0)
x = -1
y = 0.1
for _ in range(number_samples_class0_mix2):
    x_sample = x + np.random.normal(0, 0.15)
    y_sample = y + np.random.normal(0, 0.15)
    data0_x.append(x_sample)
    data0_y.append(y_sample)
    data.append([x_sample, y_sample])
    targets.append(0)
data1_x = []
data1_y = []
x = 0.5
y = 0.7
for _ in range(number_samples_class1_mix1):
    x_sample = x + np.random.normal(0, 0.15)
    y_sample = y + np.random.normal(0, 0.15)
    data1_x.append(x_sample)
    data1_y.append(y_sample)
    data.append([x_sample, y_sample])
    targets.append(1)
x = -0.5
y = 0.6
for _ in range(number_samples_class1_mix2):
    x_sample = x + np.random.normal(0, 0.15)
    y_sample = y + np.random.normal(0, 0.15)
    data1_x.append(x_sample)
    data1_y.append(y_sample)
    data.append([x_sample, y_sample])
    targets.append(1)
data = np.asarray(data)
targets = np.asarray(targets)
print(data)
print(targets)

rvm = RVM(method="classification", kernel_type="gaussian")
rvm.train(data, targets)
#Get relevance vectors
relevance_vectors = rvm.relevance_vectors
print(relevance_vectors)
#Plot relevance vectors
plt.plot([element[0] for element in relevance_vectors], [element[1] for element in relevance_vectors], 'ko', markersize=12, label="Relevance vector")
plt.plot([element[0] for element in relevance_vectors], [element[1] for element in relevance_vectors], 'wo', markersize=8)
#Plot data
plt.plot(data0_x, data0_y, 'ro', markersize=4, label="Class0 sample")
plt.plot(data1_x, data1_y, 'bo', markersize=4, label="Class1 sample")
#Plot decision boundary.
decision_boundary_x = []
decision_boundary_y = []
x_min = min(min(data0_x), min(data1_x))
x_max = max(max(data0_x), max(data1_x))
y_min = min(min(data0_y), min(data1_y))
y_max = max(max(data0_y), max(data1_y))
x_axis = np.linspace(x_min * 1.5, x_max * 1.5, 1000)
y_axis = np.linspace(y_min * 1.5, y_max * 1.5, 1000)
for x in x_axis:
    for y in y_axis:
        class0_probability, class1_probability = rvm.compute_class_probabilities([[x, y]])
        class0_probability = class0_probability[0]
        class1_probability = class1_probability[0]
        if abs(class0_probability-class1_probability)<0.001:
            decision_boundary_x.append(x)
            decision_boundary_y.append(y)
#print(decision_boundary_x)
#print(decision_boundary_y)
plt.plot(decision_boundary_x, decision_boundary_y, 'k', linestyle=":", label="Decision boundary")
plt.xlim([x_min*1.2, x_max*1.2])
plt.ylim([y_min*1.2, y_max*1.2])
plt.legend()
plt.show()
