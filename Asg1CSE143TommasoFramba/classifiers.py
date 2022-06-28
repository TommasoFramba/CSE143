"""
Classifier.py by Tommaso Framba April 25, 2022

Contains the classes for different classifiers
Naive Bayes fully implemented
Logistic Regression fully implemented
"""

import math
import numpy as np

np.set_printoptions(threshold=np.inf)

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.k = 20
        self.hate0_word_count = 0
        self.hate1_word_count = 0
        self.hate0_class_count = 0
        self.hate1_class_count = 0
        self.hate0_prior = 0
        self.hate1_prior = 0
        self.total_count = 0
        self.words_in_sentence = [[]]
        self.hate0_word_prob_dict = {}
        self.hate1_word_prob_dict = {}
        #raise Exception("Must be implemented")

    #Detect the prior probabilities
    def prior_prob(self, X, Y):
        num_rows, num_cols = X.shape

        # get prior probabilities
        hate0_class_count = 0
        hate1_class_count = 0
        for r in range(len(X)):
            if Y[r] == 1:
                hate1_class_count += 1
            elif Y[r] == 0:
                hate0_class_count += 1

        # Save word count to class
        self.hate0_class_count = hate0_class_count
        self.hate1_class_count = hate1_class_count
        self.total_count = self.hate0_class_count + self.hate1_class_count

        # Save prior prob to class
        hate0_prior = hate0_class_count / self.total_count
        hate1_prior = hate1_class_count / self.total_count

        # Return prior probabilities
        return hate0_prior, hate1_prior

    # detect word probabilities
    def world_probabilities(self, X, Y):

        #Loop through columns then rows
        for c in range(len(X[0])):
            for r in range(len(X)):
                if Y[r] == 1:
                    self.hate1_word_count += X[r][c]
                    self.hate1_word_prob_dict[c]= X[r][c] + self.hate1_word_prob_dict.get(c, 0)
                elif Y[r] == 0:
                    self.hate0_word_count += X[r][c]
                    self.hate0_word_prob_dict[c] = X[r][c] + self.hate0_word_prob_dict.get(c, 0)


        lst = []
        #add1 smoothing
        for i, c in self.hate0_word_prob_dict.items():
            self.hate0_word_prob_dict[i] = ((self.hate0_word_prob_dict[i]+1)/(self.hate0_word_count+len(X[0])))
            self.hate1_word_prob_dict[i] = ((self.hate1_word_prob_dict[i]+1)/(self.hate1_word_count+len(X[0])))
            lst.append((i,(self.hate1_word_prob_dict[i]/self.hate0_word_prob_dict[i])))

        # lst.sort(key = lambda tup: tup[1])
        #
        # print(lst[:10])




    #Get prior probabilities then individual word probabilities
    def fit(self, X, Y):
        self.hate0_prior, self.hate1_prior = self.prior_prob(X, Y)
        self.world_probabilities(X, Y)
        
    #Predict results from word probabilities
    def predict(self, X):
        y_pred = np.array([])
        correct = 0
        for r in range(len(X)):
            log_prob_hate0, log_prob_hate1 = 0, 0
            for c in range(len((X[r]))):
                if X[r][c] != 0:
                    log_prob_hate0 += np.log(self.hate0_word_prob_dict[c])
                    log_prob_hate1 += np.log(self.hate1_word_prob_dict[c])

            hate0_pred = self.hate0_prior*np.exp(log_prob_hate0)
            hate1_pred = self.hate1_prior*np.exp(log_prob_hate1)

            if hate0_pred >= hate1_pred:
                y_pred = np.append(y_pred, 0)
            else:
                y_pred = np.append(y_pred, 1)
        return y_pred


class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        self.bias = 1

    def sigmoid(self, row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row) - 1):
            yhat += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + math.exp(-yhat))

    #perform an l2 derived stochastic gradient descent
    def stochasticGD(self, X, Y, alpha, iterations, lamb):
        coefficients = np.zeros(X.shape[1])
        index = 0
        for epoch in range(iterations):
            index = 0
            print('>epoch=%d' % (epoch))
            for r in (X):
                #obatin initial yhat
                prob = self.sigmoid(r, coefficients)
                #calculate error per step
                error = r[-1] - prob
                #update initial coefficient
                coefficients[0] = coefficients[0] - alpha * error * (1/lamb) * prob * (Y[index]-prob)
                #update rest of coefficients
                for i in range(len(r) - 1):
                    coefficients[i + 1] = coefficients[i + 1] - alpha * error * (1/lamb) * prob * (Y[index]-prob) * r[i]
                index += 1

        return coefficients

    #fit the model with proper learning rate and coefficients
    def fit(self, X, Y):
        iterations = 40
        alpha = 0.1
        lamb = 1
        self.weights = self.stochasticGD(X, Y, alpha, iterations, lamb)

    #predict results
    def predict(self, X):
        predictions = np.array([])
        for row in X:
            probability = self.sigmoid(row, self.weights)
            predictions = np.append(predictions, round(probability))

        #return results
        return predictions


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
