"""
 Tommaso Framba, tframba@ucsc.edu, 1815342
 CSE 143 Assignment 2
 May 3, 2022
"""

import math

if __name__ == "__main__":

    # get unigram data
    def unigramData(wordCount, removeWords3Times):
        train = open("A2-Data/1b_benchmark.train.tokens", "r", encoding="utf8")

        # get word count per sentence in train data
        for s in train:
            # add start to beginning and end to end of each sentence
            # break up sentence into individual word tokens
            tokens = s.split()
            tokens.insert(0, "<start>")
            tokens.append("<end>")

            # append word count
            for i in tokens:
                wordCount[i] = 1 + wordCount.get(i, 0)

        # close data
        train.close()

        # Convert words that occur less than three times to UNK
        wordCount["UNK"] = 0
        for i, c in wordCount.items():
            # if count is less than 3
            if c < 3:
                # append to remove words list
                removeWords3Times.append(i)
                # increment unkown word count
                wordCount["UNK"] += c

        for word in removeWords3Times:
            # remove word count from original dictionary
            del wordCount[word]


    # Maximum likelihood estimate
    def MLE(wordCount, mleProb):
        # get sum of start token
        sumStart = wordCount["<start>"]
        # get sum of all tokens without start
        sumNotStart = sum(wordCount.values()) - sumStart

        # calc mle if token is not start
        for i, c in wordCount.items():
            if i == "<start>":
                continue
            # count of word over total count of words
            mleProb[i] = c / sumNotStart


    # MLE with additive smoothing
    def additiveSmoothingProb(wordCount, mleProb, alpha):
        # get sum of start token
        sumStart = wordCount["<start>"]
        # get sum of all tokens without start
        sumNotStart = sum(wordCount.values()) - sumStart

        # calc mle if token is not start
        for i, c in wordCount.items():
            if i == "<start>":
                continue
            # count of word over total count of words
            mleProb[i] = (c + alpha) / (c + alpha * sumNotStart)


    # Bigram probability with additive smoothing
    def bigramAdditiveSmoothing(data, mleProb, unigramProb, alpha):
        sentenceSum, totalSum, senLen = 0, 0, 0
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each element and the next element in the tokens list add tuple to bigrams
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            # for each bigram in bigrams list
            for bigr in bigrams:
                # if start occurs twice as a bigram just ignore it
                if (bigr[0] == "<start>" and bigr[1] == "<start>"):
                    continue
                # if the bigram prob is not zero add to the sentenceSum
                if bigramProb(bigr, mleProb, unigramProb) != 0:
                    sentenceSum += math.log(bigramProbSmoothing(bigr, mleProb, unigramProb, 1))
            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
            # sentence length minus start and end tokens
            senLen += len(tokens) - 2

        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * (totalSum)
        per = math.exp(exponent)
        return per


    # MLE for a bigram with additive smoothing
    def bigramProbSmoothing(featureCount, mleProb, unigramProb, alpha):
        try:
            # prob of bigram
            bigramProb = mleProb[featureCount]
            sumWordCount = unigramProb[featureCount[0]]

            # calc mle for bigram
            return float(bigramProb + alpha) / float(bigramProb + alpha * sumWordCount)
        except:
            return 0


    # MLE for a bigram with additive smoothing
    def bigramProb(featureCount, mleProb, unigramProb):
        try:
            # prob of bigram
            bigramProb = mleProb[featureCount]
            sumWordCount = unigramProb[featureCount[0]]

            # calc mle for bigram
            return float(bigramProb) / float(sumWordCount)
        except:
            return 0


    # Bigram perplexity
    def bigramPer(data, mleProb, unigramProb):
        sentenceSum, totalSum, senLen = 0, 0, 0
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each element and the next element in the tokens list add tuple to bigrams
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            # for each bigram in bigrams list
            for bigr in bigrams:
                # if start occurs twice as a bigram just ignore it
                if (bigr[0] == "<start>" and bigr[1] == "<start>"):
                    continue
                # if the bigram prob is not zero add to the sentenceSum
                if bigramProb(bigr, mleProb, unigramProb) != 0:
                    sentenceSum += math.log(bigramProb(bigr, mleProb, unigramProb))
            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
            # sentence length minus start and end tokens
            senLen += len(tokens) - 2

        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * (totalSum)
        per = math.exp(exponent)
        return per


    # unigram perplexity
    def unigramPer(data, mleProb):
        sentenceSum, totalSum, senLen = 0, 0, 0
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each unigram in tokens
            for unigram in tokens:
                # if unigram is start ignore
                if unigram == "<start>":
                    continue
                # add prob to the sentenceSum
                sentenceSum += math.log(mleProb[unigram])
                senLen += 1
            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * (totalSum)
        per = math.exp(exponent)
        return per


    # get trigram additive smoothing
    def trigramProbSmoothing(trigram, mleProb, unigramProb, wordCount, alpha):
        # if the key exists
        try:
            # get prob of the trigram
            trigramProb = mleProb[trigram]
            # if the trigram starts with start twice
            if trigram[0] == "<start>" and trigram[1] == "<start>":
                # get sum of just start
                sumWordCount = wordCount["<start>"]
            else:
                # get sum of first two
                sumWordCount = unigramProb[trigram[0:2]]
            # return mle
            return float(trigramProb + alpha) / float(trigramProb + alpha * sumWordCount)
        # if key does not exist make it 1
        except:
            return 1


    # get trigram additive smoothing
    def trigramProb(trigram, mleProb, unigramProb, wordCount):
        # if the key exists
        try:
            # get prob of the trigram
            trigramProb = mleProb[trigram]
            # if the trigram starts with start twice
            if trigram[0] == "<start>" and trigram[1] == "<start>":
                # get sum of just start
                sumWordCount = wordCount["<start>"]
            else:
                # get sum of first two
                sumWordCount = unigramProb[trigram[0:2]]
            # return mle
            return float(trigramProb) / float(sumWordCount)
        # if key does not exist make it 1
        except:
            return 1


    # get trigram additive smoothing
    def trigramAdditiveSmoothing(data, mleProb, unigramProb, wordCount, alpha):
        sentenceSum, totalSum, senLen = 0, 0, 0
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each element and the next two in tokens add triple to trigrams
            trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
            # for each trigram in trigrams list
            for trigr in trigrams:
                # if trigram probability is not zero add it to the sentenceSum
                if trigramProb(trigr, mleProb, unigramProb, wordCount) != 0:
                    sentenceSum += math.log(
                        trigramProbSmoothing(trigr, mleProb, unigramProb, wordCount, alpha))

            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
            # sentence length minus start and end tokens
            senLen += len(tokens) - 2

        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * totalSum
        per = math.exp(exponent)
        return per


    # get trigram additive smoothing
    def getTrigramPerplexity(data, mleProb, unigramProb, wordCount):
        sentenceSum, totalSum, senLen = 0, 0, 0
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each element and the next two in tokens add triple to trigrams
            trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
            # for each trigram in trigrams list
            for trigr in trigrams:
                # if trigram probability is not zero add it to the sentenceSum
                if trigramProb(trigr, mleProb, unigramProb, wordCount) != 0:
                    sentenceSum += math.log(trigramProb(trigr, mleProb, unigramProb, wordCount))

            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
            # sentence length minus start and end tokens
            senLen += len(tokens) - 2

        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * totalSum
        per = math.exp(exponent)
        return per


    # Trigram linear smoothing
    def trigramLinearSmoothing(data, mleProb, unigramProb, wordCount, l):
        sentenceSum, totalSum, senLen = 0, 0, 0
        startCount = wordCount["<start>"]
        totalTokens = sum(wordCount.values()) - startCount
        # for each sentence in the data
        for s in data:
            # break up sentence into individual word tokens
            tokens = s.split()
            # for each element and the next two in tokens add triple to trigrams
            trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
            # for each trigram in trigrams list
            for trigr in trigrams:
                inner = 0
                # add lambda to each result
                inner += (l[0] * float(wordCount[trigr[2]] / totalTokens)) \
                         + (l[1] * (bigramProb(trigr[1:3], unigramProb, wordCount))) \
                         + (l[2] * (trigramProb(trigr, mleProb, unigramProb, wordCount)))
                sentenceSum += math.log(inner)

            # increment totalSum of probability
            totalSum += sentenceSum
            sentenceSum = 0
            # sentence length minus start and end tokens
            senLen += len(tokens) - 2

        # Calculate perplexity with perplexity equation
        inverse = float(-1) / float(senLen)
        exponent = inverse * totalSum
        per = math.exp(exponent)
        return per


    # get data from file
    def getData(words, sentences, doc):
        # read document data into data
        if doc == "dev":
            data = open("A2-Data/1b_benchmark.dev.tokens", "r", encoding="utf8")
        elif doc == "test":
            data = open("A2-Data/1b_benchmark.test.tokens", "r", encoding="utf8")
        else:
            data = open("A2-Data/1b_benchmark.train.tokens", encoding="utf8")

        # for each sentence in data
        for s in data:
            # split sentence into tokens
            tokens = s.split()
            # for each word in tokens
            for word in tokens:
                # if the word is not in word dictionary
                if word not in words:
                    # set the word to UNKOWN
                    tokens[tokens.index(word)] = "UNK"

            # add start twice to detect start of sentence outside and internally
            tokens.insert(0, "<start>")
            tokens.insert(0, "<start>")

            # add end to detect end of sentence internally
            tokens.append("<end>")

            # add sentence modified to sentences array
            sentences.append(" ".join(tokens))

        # close the data
        data.close()


    # get bigrams and trigrams
    def bigramAndTrigram(data, wordCount, tokenCount):
        # for each sentence in data
        for s in data:
            # split tokens up
            tokens = s.split()
            if tokenCount == 2:
                ngram = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            if tokenCount == 3:
                ngram = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in
                         range(len(tokens) - 2)]

            # for each in n gram
            for i in ngram:
                # increment if not in wordcount set to 1
                wordCount[i] = 1 + wordCount.get(i, 0)

    unigram, bigram, trigram, prob,  = {}, {}, {}, {}
    unk, train, dev, test = [], [], [], []

    unigramData(unigram, unk)

    val = input("Linear(L) or Additive Smoothing(A): ")
    while True:
        if val == 'L' or val == 'l' or val == 'a' or val == 'A':
            break
        print("Incorrect Selection try again")
        val = input("Linear(L) or Additive Smoothing(A): ")

    # make a selection
    if val == 'A' or val == 'a':
        print("\nModel Vocabulary Unique Tokens: {}".format(len(unigram) - 1))
        print("Additive Smoothing with alpha=1:")
        getData(unigram, train, doc="train")
        additiveSmoothingProb(unigram, prob, 0.70)
        print("Train Unigram Perplexity: {}".format(unigramPer(train, prob)))
        getData(unigram, dev, doc="dev")
        # DEBUG ONLY - getData(unigramCount, devData, type="debug")
        print("Dev Unigram Perplexity: {}".format(unigramPer(dev, prob)))
        getData(unigram, test, doc="test")
        print("Test Unigram Perplexity: {}\n".format(unigramPer(test, prob)))

        bigramAndTrigram(train, bigram, 2)
        print("Model Vocabulary Unique Bigram Count: {}".format(len(bigram) - 1))
        print("Train Bigram Perplexity: {}".format(bigramAdditiveSmoothing(train, bigram, unigram, 0.70)))
        print("Dev Bigram Perplexity: {}".format(bigramAdditiveSmoothing(dev, bigram, unigram, 0.70)))
        print("Test Bigram Perplexity: {}\n".format(bigramAdditiveSmoothing(test, bigram, unigram, 0.70)))

        bigramAndTrigram(train, trigram, 3)
        print("Model Vocabulary Unique Trigram Count: {}".format(len(trigram) - 1))
        print("Train Trigram Perplexity: {}".format(trigramAdditiveSmoothing(train, trigram, bigram, unigram, 0.70)))
        print("Dev Trigram Perplexity: {}".format(trigramAdditiveSmoothing(dev, trigram, bigram, unigram, 0.70)))
        print("Test Trigram Perplexity: {}\n".format(trigramAdditiveSmoothing(test, trigram, bigram, unigram, 0.70)))

    else:
        print("\nModel Vocabulary Unique Tokens: {}".format(len(unigram) - 1))

        print("No Smoothing:")
        getData(unigram, train, doc="train")
        MLE(unigram, prob)
        print("Train Unigram Perplexity: {}".format(unigramPer(train, prob)))
        getData(unigram, dev, doc="dev")
        # DEBUG ONLY - getData(unigramCount, devData, type="debug")
        print("Dev Unigram Perplexity: {}".format(unigramPer(dev, prob)))
        getData(unigram, test, doc="test")
        print("Test Unigram Perplexity: {}\n".format(unigramPer(test, prob)))

        bigramAndTrigram(train, bigram, 2)
        print("Model Vocabulary Unique Bigram Count: {}".format(len(bigram) - 1))
        print("Train Bigram Perplexity: {}".format(bigramPer(train, bigram, unigram)))
        print("Dev Bigram Perplexity: {}".format(bigramPer(dev, bigram, unigram)))
        print("Test Bigram Perplexity: {}\n".format(bigramPer(test, bigram, unigram)))

        bigramAndTrigram(train, trigram, 3)
        print("Model Vocabulary Unique Trigram Count: {}".format(len(trigram) - 1))
        print("Train Trigram Perplexity: {}".format(getTrigramPerplexity(train, trigram, bigram, unigram)))
        print("Dev Trigram Perplexity: {}".format(getTrigramPerplexity(dev, trigram, bigram, unigram)))
        print("Test Trigram Perplexity: {}\n".format(getTrigramPerplexity(test, trigram, bigram, unigram)))

        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .2, .3])))
        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(train, trigram, bigram, unigram, [.3, .4, .5])))
        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .3, .6])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .2, .3])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.3, .4, .5])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .3, .6])))

        print("Test Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.4, .5, .6, trigramLinearSmoothing(test, trigram, bigram, unigram, [.4, .5, .6])))

