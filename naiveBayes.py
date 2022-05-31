import numpy as np
import pandas as pd
from scipy.stats import lognorm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import sys

# Import the digits dataset
digits = datasets.load_digits()
X = digits.data[:, :]
y = digits.target
# Using train_test_split split the dataset 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#------------ My implementation of the Naive Bayes algorithm -------------------
np.seterr(divide = 'ignore')
class NaiveBayes(object):

    def __init__(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        self.num_features = len(X_train[0])
        # Total number of each class and total samples, digit 0 = class_count[0]
        self.class_counts = []
        for i in range(self.num_classes):
            self.class_counts.append(str(np.count_nonzero(y_train == (i))))

        # Find the total number of samples in the training set
        self.total_samples=len(y_train)

    def fit(self, X_train, y_train):
        # Turn the training set into a Pandas DataFrame
        self.df = pd.DataFrame(X_train)
        self.df += 1

        self.df['Class'] = y_train
        self.df_sorted = self.df.sort_values('Class')

        # Find the probabilities of finding each class in the training set
        self.class_prob = []
        for i in range(len(self.class_counts)):
            self.class_prob.append(str(int(self.class_counts[i]) / self.total_samples))

        # Calculate the feature_class mean and variance and hold themn in 2 2D arrays
        self.feature_class_means = np.zeros((self.num_classes, self.num_features))
        self.feature_class_variance = np.zeros((self.num_classes, self.num_features))
        for i in range(self.num_classes):
            self.df_label = self.df_sorted.loc[self.df_sorted['Class'] == i]
            for j in range(self.num_features):
                self.feature_class_means[i, j] = self.df_label[j].sum()/float(self.class_counts[i])
                self.feature_class_variance[i, j] = self.df_label[j].var()

    def density_function(self, x, mean, variance):
        # input the arguments into the probability density function
        try:
            self.p = 1/(np.sqrt(2*np.pi*variance)) * np.exp((-(x-mean)**2)/(2*variance))
        except ZeroDivisionError:
            self.p = 0
        return self.p

    def predict(self, X_test):
        self.X_test = X_test
        self.X_test += 1
        self.y_pred = []
        for x in range(len(X_test)):
            self.y_probabilities = []
            for i in range(self.num_classes):
                self.probability = 0
                for j in range(self.num_features):
                    self.mean = self.feature_class_means[i, j]
                    self.variance = self.feature_class_variance[i, j]
                    if self.variance != 0:
                        if self.X_test.ndim == 1:
                            self.probability = self.probability + np.log(np.nan_to_num(self.density_function(self.X_test[j], self.mean , self.variance)))
                        else:
                            self.probability = self.probability + np.log(np.nan_to_num(self.density_function(self.X_test[x, j], self.mean , self.variance)))
                self.probability = np.log(float(self.class_prob[i])) + self.probability
                self.y_probabilities.append(self.probability)
            self.y_pred.append(np.argmax(self.y_probabilities))
        if self.X_test.ndim == 1:
            return self.y_pred[0]
        return self.y_pred

def train_My_NB_Model():
    # Call and train my implementation of Naive Bayes
    my_nb = NaiveBayes(X_train, y_train)
    my_nb.fit_(X_train, y_train)
    return my_nb
my_filename = "myNaiveBayes.pkl"

#------------ sklearn implementation of that Naive Bayes algorithm -------------

# Train and fit
def train_sklearn_nb():
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb
sk_learn_filename = "sklearnNaiveBayes.pkl"

#---------------------------- Save and call models -----------------------------
def save_Model(filename, model):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_and_predict(filename):
    # Load from file
    with open(filename, 'rb') as file:
        saved_model = pickle.load(file)
    # Predict from saved model
    pred_test = saved_model.predict(X_test)
    pred_train = saved_model.predict(X_train)
    return pred_test, pred_train

def get_metrics(pred, y):
    # Accuracy metrics
    miss_class = ((pred != y).sum())
    accuracy = accuracy_score(y, pred)
    return miss_class, accuracy

# -------------------------------- Interface -----------------------------------
def interface():
    inputs=['f2', 'f3', 'f4', 'f5', 'train', 'q']
    print("---------------------------------------------------------------------------------------------------------")
    print("")
    print("The dataset used is the Optical recognition of handwritten digits dataset imported from sklearn.")
    print("The number of samples in the dataset is : ", len(X))
    print("The number of features in each sample is : ", len(X_train[0]))
    print('The number of classes in the dataset is : ', len(np.unique(y_train)))
    print('The different class labels are : ', np.unique(y))
    print('The number of samples for each class is : ', np.bincount(y))
    print("The min and max for each feature is the same for all. It is min=0 and max=16.")
    print("The train and test split 0.7:0.3")
    print("The number of samples in the training set is", len(y_train), ", the number of samples in the test is ", len(y_test))
    print('Label counts in the training set: ', np.bincount(y_train))
    print('Label counts in the test set: ', np.bincount(y_test))
    print("")
    print("---------------------------------------------------------------------------------------------------------")
    print("")
    print("Enter 'f2' to load the Naive Bayes algorithm from sklearn")
    print("Enter 'f3' to load my own Naive Bayes algorithm implementation")
    print("Enter 'f5' to query the model")
    print("Enter 'train' to train the models")
    print("Enter 'q' to quit the program")
    print("")
    print("---------------------------------------------------------------------------------------------------------")
    x = input()
    if x == 'f2':
        print("sklearn Naive Bayes classifier")
        predictions = load_and_predict(sk_learn_filename)
        test_metrics = get_metrics(predictions[0], y_test)
        train_metrics = get_metrics(predictions[1], y_train)
        print("")
        print("Test set: ")
        print("Missclassified samples: ", test_metrics[0])
        print("Accuracy: %.2f" % test_metrics[1])
        print("")
        print("Training set: ")
        print("Missclassified samples: ", train_metrics[0])
        print("Accuracy: %.2f" % train_metrics[1])
        print("")

    if x == 'f3':
        print("My own Naive Bayes classifier")
        predictions = load_and_predict(my_filename)
        test_metrics = get_metrics(predictions[0], y_test)
        train_metrics = get_metrics(predictions[1], y_train)
        print("")
        print("Test set: ")
        print("Missclassified samples: ", test_metrics[0])
        print("Accuracy: %.2f" % test_metrics[1])
        print("")
        print("Training set: ")
        print("Missclassified samples: ", train_metrics[0])
        print("Accuracy: %.2f" % train_metrics[1])
        print("")

    if x == 'f5':
        print("")
        print("Query Models")
        print("Enter a sample index of the dataset to predict a particular sample")
        index = int(input())
        with open(sk_learn_filename, 'rb') as file:
            saved_model = pickle.load(file)
        pred_nb = saved_model.predict(X[[index]])
        print("The prediction :", pred_nb)
        print("The actual value :", y[index])
        with open(my_filename, 'rb') as file:
            saved_model = pickle.load(file)
        pred_my = saved_model.predict(X[index])
        print("The prediction :", pred_my)
        print("The actual value :", y[index])
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == 'train':
        print("")
        print("Which algorithm would you like to train? Please enter '1' for sklearn algorithm or '2' for my own implementation.")
        train = input()
        if train == '1':
            new_sk = train_sklearn_nb()
            print("Would you like to save this model? Enter 'yes' or 'no'.")
            save = input()
            if save == 'yes':
                print("Enter name for model - must be one word. ")
                name = input()
                name = name+".pkl"
                save_Model(name, new_sk)
                print("Enter any key to return to menu")
                c = input()
                interface()
            if save == 'no':
                interface()
        if train == '2':
            new_nb = train_My_NB_Model()
            print("Would you like to save this model? Enter 'yes' or 'no'.")
            save = input()
            if save == 'yes':
                print("Enter name for model - must be one word. ")
                name = input()
                name = name+".pkl"
                save_Model(name, new_nb)
                print("Enter any key to return to menu")
                c = input()
                interface()
            if save == 'no':
                interface()
        elif train != '1' and train != '2':
            print("Invalid input")
            print("Enter any key to return to menu")
            c = input()
            interface()
    if x == 'q':
        sys.exit(0)
    if x not in inputs:
        print("Invalid input")
        print("Enter any key to return to menu")
        c = input()
        interface()
interface()
