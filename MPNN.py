"""
Group Number : 21

Roll Numbers : Names of members
20CS60R68    : Trapti Singh
20CS60R70    : Ram Kishor Yadav
20CS60R71    : Rohit

Project number : 2
Project code   : MPNN
Project title : Popularity of Music Records Prediction using Artificial Neural Network
"""

# Including necessary libraries

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# Reading music data
data = pd.read_csv('music_data.csv', encoding='latin-1')

X = data.iloc[:, 6 : 38].values       # Feature variable
Y = data.iloc[:, 38].values

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Normalizing train and test data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

plt.title("Plot for accuracy blue line for Train and Red line for Test")
plt.xlabel("No of epochs") 
plt.ylabel("% Accuracy")
eph = []
train_test_acc = []
class NeuralNetwork:
    def __init__(self,X, Y, X_test, Y_test,mini_batch_size=32, n_hidden_layers=2, learninig_rate=0.01, hidden_nodes=32, epochs=200):

        self.X = X
        self.Y = Y[:, None]
        self.X_test = X_test
        self.Y_test = Y_test

        # defining parameters

        np.random.seed(4)
        self.input_nodes = len(X[0])     # number of features in the training data
        self.hidden_nodes = hidden_nodes
        self.output_npdes = 1
        self.learning_rate = learninig_rate
        # Weights
        self.w = []
        self.w.append('NA')
        for ini in range(0,n_hidden_layers):
            self.w.append('NA')
        self.w.append('NA')
        # initializing the weights for our network
        self.w[0] = 2 * np.random.random((self.input_nodes, self.hidden_nodes)) - 1
        for i in range(1, n_hidden_layers+1):
            self.w[i] = 2 * np.random.random((self.hidden_nodes, self.hidden_nodes)) - 1
        self.w[n_hidden_layers+1] = 2 * np.random.random((self.hidden_nodes, self.output_npdes)) - 1

        self.train(epochs, X, Y, mini_batch_size, n_hidden_layers)  # Since we have to train our model for many times we here pass epochs count
        self.test(n_hidden_layers)
    # in between input and hidden layers
    # Defining the activation function as a relu function
    def sigmoid(self, X):
        return (1 / (1 + np.exp(-X)))

    def sigmoid_prime(self, X):
        return X * (1 - X)
    
    def relu(self, x):
        return np.maximum(0 , x)

    def relu_derivative(self, x):
      return np.greater(x, 0).astype(int)

    def train(self, epochs, X, Y, mini_batch_size, n_hidden_layers):

        n = len(X)

        for e in range(epochs):

            # Creation of mini batches
            mini_batches_X = [X[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches_Y = [Y[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_y = 0
            for mini_batch_X in mini_batches_X:
                # FORWARDPROPAGATION
                l = []
                for ini in range(0,n_hidden_layers+2):
                    l.append('NA')
                l[0] = self.relu(np.dot(mini_batch_X, self.w[0]))
                # in between hidden and output
                for i in range(1,(n_hidden_layers+1)):
                    l[i] = self.relu(np.dot(l[i-1], self.w[i]))

                l[n_hidden_layers+1] = self.sigmoid(np.dot(l[n_hidden_layers], self.w[n_hidden_layers+1]))

                # BACKPROPAGATION
                # Network error (True value - Predicted value)
                mini_batches_Y[mini_y] = mini_batches_Y[mini_y][:,None]
                error = mini_batches_Y[mini_y] - l[n_hidden_layers+1]

                # error for each of the layers
                l_delta = []
                for inid in range(0,n_hidden_layers+2):
                    l_delta.append('NA')
                l_delta[n_hidden_layers+1] = error * self.sigmoid_prime(l[n_hidden_layers+1])
                for idel in range(n_hidden_layers,0,-1):
                    l_delta[idel] = l_delta[idel+1].dot(self.w[idel+1].T) * self.relu_derivative(l[idel])
                l_delta[0] = l_delta[1].dot(self.w[1].T) * self.relu_derivative(l[0])

                self.w[n_hidden_layers+1] = np.add(self.w[n_hidden_layers+1], l[n_hidden_layers].T.dot(l_delta[n_hidden_layers+1]) * self.learning_rate)
                for ierr in range(n_hidden_layers,0,-1):
                    self.w[ierr] = np.add(self.w[ierr], l[ierr-1].T.dot(l_delta[ierr]) * self.learning_rate)
                self.w[0] = np.add(self.w[0], mini_batch_X.T.dot(l_delta[0]) * self.learning_rate)

                mini_y += 1
            if e % 10 == 0 :

                correct = 0
                pred_list = []
                l[0] = self.relu(np.dot(self.X_test, self.w[0]))
                for ipred in range(1,n_hidden_layers+1):
                    l[ipred] = self.relu(np.dot(l[ipred-1], self.w[ipred]))
                l[n_hidden_layers+1] = self.sigmoid(np.dot(l[n_hidden_layers], self.w[n_hidden_layers+1]))

                for i in range(len(l[n_hidden_layers+1])):
                    if l[n_hidden_layers+1][i] >= 0.5:
                        pred = 1
                    else:
                        pred = 0

                    if pred == self.Y_test[i]:
                        correct += 1

                    pred_list.append(pred)
                train_test_acc.append((1 - (abs(error)).mean(), (correct / len(Y_test))))
                eph.append(e)

        plt.plot(eph, train_test_acc)
        plt.show()

        print('Training accuracy : ', (1 - (abs(error)).mean())*100,'%')


    # testing and evaluation


    def test(self, n_hidden_layers):

        correct = 0
        pred_list = []
        l = []
        for ini in range(0,n_hidden_layers+2):
            l.append('NA')
        l[0] = self.relu(np.dot(self.X_test, self.w[0])) #replace relu
        for itest in range(1,n_hidden_layers+1):
            l[itest] = self.relu(np.dot(l[itest-1], self.w[itest])) #replace relu
        l[n_hidden_layers+1] = self.sigmoid(np.dot(l[n_hidden_layers], self.w[n_hidden_layers+1]))

        for i in range(len(l[n_hidden_layers+1])):
            if l[n_hidden_layers+1][i] >= 0.5:
                pred = 1
            else:
                pred = 0

            if pred == self.Y_test[i]:
                correct += 1

            pred_list.append(pred)

        print('Test Accuracy : ', ((correct / len(Y_test)) * 100), '%')

nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))