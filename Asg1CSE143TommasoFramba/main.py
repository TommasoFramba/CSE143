import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(labels)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))


def read_data(path):
    train_frame = pd.read_csv(path + 'train.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        test_frame = train_frame

    try:
        dev_frame = pd.read_csv(path + 'dev.csv')
    except:
        dev_frame = train_frame

    return train_frame, test_frame, dev_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, test_frame, dev_frame = read_data(args.path)


    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']

    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']

    # form dev set for evaluation
    tokenized_text = []
    for i in range(0, len(dev_frame['text'])):
        tokenized_text.append(tokenize(dev_frame['text'][i]))
    X_dev = feat_extractor.transform_list(tokenized_text)
    Y_dev = dev_frame['label']


    if args.model == "AlwaysPredictZero":
        model = AlwaysPreditZero()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    elif args.model == 'BonusClassifier':
        model = BonusClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")

    from sklearn.naive_bayes import MultinomialNB

    model3 = MultinomialNB()
    model3.fit(X_train, Y_train)
    print("Multinomial sikit score for X_train is: {}".format(model3.score(X_train, Y_train)))
    print("Multinomial sikit score for X_test is: {}".format(model3.score(X_test, Y_test)))
    print("Multinomial sikit score for X_dev is: {}".format(model3.score(X_dev, Y_dev)))
    print("Sikit score for comparison purposes only\n")

    model2 = LogisticRegression()
    model2.fit(X_train, Y_train)
    print("Logistic sikit score for X_train is: {}".format(model2.score(X_train, Y_train)))
    print("Logistic sikit score for X_test is: {}".format(model2.score(X_test, Y_test)))
    print("Logistic sikit score for X_dev is: {}".format(model2.score(X_dev, Y_dev)))
    print("Sikit score for comparison purposes only\n")

    start_time = time.time()
    print("===== Train Accuracy =====")
    model.fit(X_train, Y_train)
    accuracy(model.predict(X_train), Y_train)

    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("===== Dev Accuracy =====")
    accuracy(model.predict(X_dev), Y_dev)

    print("Time for training, test, and dev: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()