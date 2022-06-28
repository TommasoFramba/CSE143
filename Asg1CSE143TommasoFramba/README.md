# Assignment 1 by Tommaso Framba

This assignment implements a NaiveBayes and Logistic Regression classifier to the classifiers.py file. Init, Fit, and Predict methods were fully implemented to handle the tasks provided by the assignment specifications

## Usage

For Naive Bayes in main.py be sure to call fit with the text and label numpy arrays that you want to train the model with. Then call predict on the model inside of the accuracy method to see your score.

```python
#in main.py
#create the NaiveBayesClassifier model
model = NaiveBayesClassifier()

# Fit the text and label numpy arrays
model.fit(X_train, Y_train)

# Predict the model and return its accuracy
accuracy(model.predict(X_train), Y_train)
```

For Logistic Regression in main.py be sure to call fit with the text and label numpy arrays that you want to train the model with. Then call predict on the model inside of the accuracy method to see your score.

```python
#in main.py
#create the LogisticRegressionClassifier model
model = LogisticRegressionClassifier()

# Fit the text and label numpy arrays
model.fit(X_train, Y_train)

# Predict the model and return its accuracy
accuracy(model.predict(X_train), Y_train)
```
The logistic regression classifier class has three fields that can be used when calculating its weights through the stochastic gradient descent algorithm. In order to change these parameters you must edit them manually inside of the classifiers.py method. 
The three fields that can be changed are iterations, alpha, and lamb. Iterations is the number of epochs that the gradient descent algorithm will run through. Alpha is the learning rate that the algorithm will use for every epoch. Lamb is the lambda for l2 regularization change.
```python
#in classifiers.py
#in class LogisticRegressionClassifier

#fit the model with proper learning rate and coefficients
    def fit(self, X, Y):
        #for best result use iter = 40, alpha = 0.1, lamb = 1
        #for quick result use iter = 15, alpha = 0.3, lamb = 1
        iterations = 40
        alpha = 0.1
        lamb = 1
        self.weights = self.stochasticGD(X, Y, alpha, iterations, lamb)
```
## Authors
This assignment in its entirety was implemented and coded in Python 3 by Tommaso Framba (tframba@ucsc.edu): 1815342
