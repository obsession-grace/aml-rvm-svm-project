__author__ = "Johan"

import numpy as np
from rvm.rvm import RVM
from svm.svr import SVR
from svm.svc import SVC
import random

#-----------------------------------------------------
#-----------------Regression datasets-----------------
#-----------------------------------------------------
def print_result(function_name, d, number_of_train_samples, average_number_of_relevance_vectors, average_error):
    print("Dataset: " + function_name)
    print("d: " + str(d))
    print("Number of training samples: " + str(number_of_train_samples))
    print("Average number of relevance vectors: " + str(average_number_of_relevance_vectors))
    print("Average error: " + str(average_error) + "\n")
    
def get_sinc_data(noise):
    x_samples = np.linspace(-10, 10, 100)
    y_samples = np.sin(x_samples) / x_samples
    x_samples = np.vstack(x_samples)
    if noise == "gaussian":
        y_samples = y_samples + np.random.normal(0, 0.1, 100)
    elif noise == "uniform":
        y_samples = y_samples + np.random.uniform(-0.1, 0.1, 100)
    return x_samples, y_samples

def get_boston_housing_data():
    with open("datasets/boston_housing.txt") as file:
        data_lines = file.readlines()
    file.close()
    data_lines = [line.strip() for line in data_lines]
    samples = []
    targets = []
    for line in data_lines:
        line = line.split(" ")
        input_values = []
        for input_value in line[:13]:
            input_values.append(float(input_value))
        samples.append(input_values)
        targets.append(float(line[-1]))
    return np.asarray(samples), np.asarray(targets)

def regression():
    for method in ["SVM", "RVM"]:
        print("----------REGRESSION----------")
        print("METHOD: " + method)
        #--------sinc with gaussian noise and uniform noise--------
        noises = ["gaussian", "uniform"]
        iterations = 100
        average_number_of_relevance_vectors = 0
        average_error = 0
        for noise in noises:
            for _ in range(iterations):
                #Get samples.
                x_samples, y_samples = get_sinc_data(noise)
                #Fit data.
                ################################################################################
                ################################################################################
                ################################################################################
                if method=="SVM":
                    rm = SVR(c=1e3, kernel="rbf", kernel_arg=0.1)
                elif method =="RVM":
                    rm = RVM(method="regression", kernel="rbf", kernel_arg=0.11111)
                ################################################################################
                ################################################################################
                ################################################################################
                rm.train(x_samples, y_samples)
                #Get relevance vectors
                average_number_of_relevance_vectors = average_number_of_relevance_vectors + rm.get_number_of_relevance_vectors() / iterations
                #Compute error
                x_input = np.linspace(-10, 10, 1000)
                y_true = np.sin(x_input) / x_input
                error = rm.calc_rms(x_input, y_true)
                average_error = average_error + error / iterations
            print_result("Sinc (" + noise + " noise)", 1, 100, average_number_of_relevance_vectors, average_error)

        #-------------------Boston housing data--------------------
        iterations = 100
        average_number_of_relevance_vectors = 0
        average_error = 0
        #Get data.
        samples, targets = get_boston_housing_data()
        for iteration in range(iterations):
            #Pick 481 random training samples and (506-481)=25 remaining test samples.
            train_indices = random.sample(range(0, 506), 481)
            test_indices = range(0, 505)
            test_indices = [x for x in test_indices if x not in train_indices]
            train_samples = samples[train_indices]
            train_targets = targets[train_indices]
            test_samples = samples[test_indices]
            test_targets = targets[test_indices]
            #Fit data.
            ################################################################################
            ################################################################################
            ################################################################################
            if method=="SVM":
                rm = SVR(c=1e3, kernel="rbf", kernel_arg=0.1)
            elif method =="RVM":
                rm = RVM(method="regression", kernel="rbf", kernel_arg=0.11111)
            ################################################################################
            ################################################################################
            ################################################################################
            rm.train(train_samples, train_targets)
            #Get relevance vectors
            average_number_of_relevance_vectors = average_number_of_relevance_vectors + rm.get_number_of_relevance_vectors() / iterations
            #Compute error.
            error = rm.calc_rms(test_samples, test_targets)
            average_error = average_error + error / iterations
        print_result("Boston Housing", len(samples[0]), 481, average_number_of_relevance_vectors, average_error)    


#-----------------------------------------------------
#---------------Classification datasets---------------
#-----------------------------------------------------
def get_samples_classification(dataset, train_test, number):
    with open("datasets/" + dataset + "/" + dataset + "_" + train_test + "_data_" + str(number) + ".txt") as file:
        data_lines = file.readlines()
    file.close()
    data_lines = [line.strip() for line in data_lines]
    samples = []
    for sample in data_lines:
        sample = sample.split(" ")
        parameters = []
        for parameter in sample:
            parameters.append(float(parameter))
        samples.append(parameters)
    return np.asarray(samples)

def get_sample_targets_classification(dataset, train_test, number):
    with open("datasets/" + dataset + "/" + dataset + "_" + train_test + "_labels_" + str(number) + ".txt") as file:
        data_lines = file.readlines()
    file.close()
    data_lines = [line.strip() for line in data_lines]
    sample_targets = []
    for sample in data_lines:
        sample_targets.append(int(sample))
    return np.asarray(sample_targets)

def classification():
    for method in ["SVM", "RVM"]:
        print("--------CLASSIFICATION--------")
        print("METHOD: " + method)
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
                number_of_features = len(train_samples[0])
                if iteration == 0:
                    d.append(number_of_features)
                #Fit the data.
                ################################################################################
                ################################################################################
                ################################################################################
                if method=="SVM":
                    cm = SVC(c=1e3, kernel="rbf", kernel_arg=1/number_of_features**2)
                    train_sample_targets = train_sample_targets.astype(float)
                    train_sample_targets[train_sample_targets==0] = -1.0
                    train_sample_targets[train_sample_targets==1] = 1.0
                    test_sample_targets = test_sample_targets.astype(float)
                    test_sample_targets[test_sample_targets==0] = -1.0
                    test_sample_targets[test_sample_targets==1] = 1.0
                elif method=="RVM":
                    cm = RVM(method = "classification", kernel = "rbf", kernel_arg=1/number_of_features**2)
                ################################################################################
                ################################################################################
                ################################################################################
                cm.train(train_samples, train_sample_targets)
                #Get relevance vectors.
                total_number_of_relevance_vectors = total_number_of_relevance_vectors + cm.get_number_of_relevance_vectors()
                #Make predictions.
                total_number_of_correct_predictions = total_number_of_correct_predictions + cm.get_number_of_correct_predictions(test_samples, test_sample_targets)
            result_n_train_samples.append(str(len(train_samples)))
            result_n_test_samples.append(str(len(test_samples)))
            result_average_n_relevance_vectors.append(str(total_number_of_relevance_vectors / number_of_iterations))
            result_average_error.append(str(100 * (total_number_of_test_samples - total_number_of_correct_predictions) / total_number_of_test_samples))

        for i in range (len(datasets)):
            print("\nDataset: " + datasets[i])
            print("d: " + str(d[i]))
            print("Number of '" + datasets[i] + "' datasets: " + str(number_of_iterations))
            print("Number of training samples in each dataset: " + result_n_train_samples[i])
            print("Number of test samples in each dataset: " + result_n_test_samples[i])
            print("Average number of relevance vectors: " + result_average_n_relevance_vectors[i])
            print("Average error (%): " + result_average_error[i])

regression()
classification()
