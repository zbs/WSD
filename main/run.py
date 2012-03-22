train_filename = "../train.txt"
test_filename = "../test.txt"
kaggle_filename = "../kaggle.txt"

import re
from Word import Word
import features
# http://inclass.kaggle.com/c/cornell-cs4740-word-sense-disambiguation

FEATURE_FUNS = (features.posNeighbors, ) #add some more
LIMIT_WORDS = True
LIMIT = 2000

def parse(filename):
    samples = []
    count = 0
    for line in open(filename):
        if LIMIT_WORDS and count > LIMIT: break
        count +=1
        #find tag, classes, and context in line of file
        data = re.match('^(?P<tag>[a-z]+\.[a-z]) (?P<classes>[01 ]+) @ (?P<context>.+)', line)
        tag = data.group('tag')
        #get the classes into an int array
        classes = map(int, re.split(' ', data.group('classes')))
        context = data.group('context')
        samples.append( (tag, classes, context) )
        #print lines[-1]
    return samples

def buildModels(examples, feature_funs):
    #the samples are arranged in the data file by word
    #it's useful to preserve this order (as opposed to using a dict)
    #since the kaggle results require the same order.
    words = []
    curr_tag = None
    for sample in examples:
        next_tag, classes, context = sample
        #when sample has a previously unseen tag
        if curr_tag != next_tag:
            curr_tag = next_tag
            words.append( Word(curr_tag, feature_funs) )
        words[-1].add_sample( classes, context)
        
    map(lambda word: word.splitCV(.10), words)
    map(lambda word: word.classify(), words)
    return words

#If using test data, then set the "tests" variable. Otherwise cv data is used
def testModels(words, tests = None):
    isTest = tests is not None
    if isTest: #then add test samples to Words
        curr_tag = None
        word_ptr = -1
        for test in tests:
            next_tag, classes, context = test
            if curr_tag != next_tag:
                word_ptr += 1
                curr_tag = next_tag
                words[word_ptr].add_sample(classes, context, isTest = True)
    predicted = sum (map(lambda word: word.predict(isTest), words), []) # map and flatten
    if isTest:
        kaggle_file = open(kaggle_filename, 'w+')
        kaggle_file.write("\n".join(predicted))
        return predicted
    actual = sum( map(lambda word: word.get_actual(isTest), words), [])
    return actual, predicted

def analyze(predicted, actual):
    tp, fp, fn, tn = 0., 0., 0., 0.
    for (p,a) in zip(predicted, actual):
        if p == a and a == 1.:   tp += 1.
        elif p == a and a == 0.: tn += 1.
        elif p == 1. and a == 0.: fp += 1.
        else:                   fn += 1.
    #http://en.wikipedia.org/wiki/Precision_and_recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #http://en.wikipedia.org/wiki/F1_score
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    return f_measure, accuracy
               
if __name__ == '__main__':
'''
    bass = Word('bass.v',())
    bass.add_sample([1,0],"I went @bass@ fishing")
    bass.add_sample([0,1],"I am playing @bass@ ball")
    bass.build_context_list()
    for t in bass.tokens:
	    print (t,bass.tokens[t])
    f = features.cooccurrances(bass,"I were playing @bass@ fish")
    print f
'''
    examples = parse(train_filename)
    words = buildModels(examples, FEATURE_FUNS)
    actual, predicted = testModels(words)
    print analyze(predicted, actual)
   

    