__author__ = "Johan"

import numpy as np
from rvm import RVM

def get_samples_classification(dataset, train_test, number):
    with open("../datasets/" + dataset + "/" + dataset + "_" + train_test + "_data_" + str(number) + ".txt") as file:
        data_lines = file.readlines()
    file.close()
    data_lines = [line.strip() for line in data_lines]
    data_lines = [line.replace("   ", " ").replace("  ", " ") for line in data_lines]
    samples = []
    for sample in data_lines:
        sample = sample.split(" ")
        parameters = []
        for parameter in sample:
            parameters.append(float(parameter))
        samples.append(parameters)
    return np.asarray(samples)

def get_sample_targets_classification(dataset, train_test, number):
    with open("../datasets/" + dataset + "/" + dataset + "_" + train_test + "_labels_" + str(number) + ".txt") as file:
        data_lines = file.readlines()
    file.close()
    data_lines = [line.strip() for line in data_lines]
    data_lines = [line.replace("   ", " ").replace("  ", " ").replace("-1.0000000e+00","0").replace("1.0000000e+00", "1") for line in data_lines]
    sample_targets = []
    for sample in data_lines:
        sample_targets.append(int(sample))
    return np.asarray(sample_targets)

#-----------------Regression datasets-----------------:

#---------------Classification datasets---------------:
datasets = ["banana", "breast-cancer", "titanic", "waveform", "german"]
number_of_iterations = 10
d = []
result_n_train_samples = []
result_n_test_samples = []
result_average_n_relevance_vectors = []
result_average_error = []

for dataset in datasets:
    print("Dataset: " + dataset)
    total_number_of_relevance_vectors = 0
    total_number_of_test_samples = 0
    total_number_of_correct_predictions = 0
    for iteration in range(number_of_iterations):
        print("Iteration: " + str(iteration+1))
        #Get data.
        train_samples = get_samples_classification(dataset, "train", iteration+1)
        train_sample_targets = get_sample_targets_classification(dataset, "train", iteration+1)
        test_samples = get_samples_classification(dataset, "test", iteration+1)
        total_number_of_test_samples = total_number_of_test_samples + len(test_samples)
        test_sample_targets = get_sample_targets_classification(dataset, "test", iteration+1)
        if iteration == 0:
            d.append(len(train_samples[0]))
        #Fit the data.
        rvm = RVM(method = "classification", kernel_type = "gaussian")
        rvm.train(train_samples , train_sample_targets)
        #Get relevance vectors.
        relevance_vectors = rvm.relevance_vectors
        total_number_of_relevance_vectors = total_number_of_relevance_vectors + len(relevance_vectors)
        #Make predictions.
        predicted_targets = rvm.make_prediction(test_samples)
        total_number_of_correct_predictions = total_number_of_correct_predictions + np.sum(test_sample_targets == predicted_targets)
    result_n_train_samples.append(str(len(train_samples)))
    result_n_test_samples.append(str(len(test_samples)))
    result_average_n_relevance_vectors.append(str(total_number_of_relevance_vectors / number_of_iterations))
    result_average_error.append(str(100 * ((total_number_of_test_samples - total_number_of_correct_predictions) / total_number_of_test_samples)))

for i in range (len(datasets)):
    print("\nDataset: " + datasets[i])
    print("d: " + str(d[i]))
    print("Number of '" + datasets[i] + "' datasets: " + str(number_of_iterations))
    print("Number of training samples in each dataset: " + result_n_train_samples[i])
    print("Number of test samples in each dataset: " + result_n_test_samples[i])
    print("Average number of relevance vectors: " + result_average_n_relevance_vectors[i])
    print("Average error (%): " + result_average_error[i])
