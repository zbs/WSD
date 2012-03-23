train_filename = "../train.txt"
test_filename = "../test.txt"
kaggle_filename = "../kaggle.txt"

train_pickle = "train_pickle.txt"
test_pickle = "test_pickle.txt"

import re, pickle, os
from Word import Word
import features
# http://inclass.kaggle.com/c/cornell-cs4740-word-sense-disambiguation

FEATURE_FUNS = (features.posNeighbors, ) #add some more
LIMIT_WORDS = True #true to test and/or train with limited # words
LIMIT = 2
PICKLE_FILES = False #set to True to reparse data files
TEST = False #true to run test set and write kaggle file

def parse(filename, pickle_file):
    if not PICKLE_FILES and os.path.isfile(pickle_file):
        return pickle.load(open(pickle_file, 'rb')) 
    samples = []
    for line in open(filename):
        #find tag, classes, and context in line of file
        data = re.match('^(?P<tag>[a-z]+\.[a-z]) (?P<classes>[01 ]+) @ (?P<context>.+)', line)
        tag = data.group('tag')
        #get the classes into an int array
        classes = map(int, re.split(' ', data.group('classes')))
        context = data.group('context')
        samples.append( (tag, classes, context) )
        #print lines[-1]
    pickle.dump(samples, open(pickle_file, 'wb'))
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
            if LIMIT_WORDS and len(words) >= LIMIT: break;
            curr_tag = next_tag
            words.append( Word(curr_tag, feature_funs) )
        words[-1].add_sample( classes, context)
        
    map(lambda word: word.setup(), words)
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
                if LIMIT_WORDS and word_ptr >= LIMIT:
                    break
                curr_tag = next_tag
            words[word_ptr].add_sample(classes, context, isTest = True)
    predicted = sum (map(lambda word: word.predict(isTest), words), []) # map and flatten
    if isTest:
        kaggle_file = open(kaggle_filename, 'w+')
        map(lambda p: kaggle_file.write(str(p)+'\n'), predicted)
        kaggle_file.close();
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

def get_word(words, tag):
    for word in words:
        if word.tag == tag:
            return word
    return None

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
    #look at run options at top of file.
    examples = parse(train_filename, train_pickle)
    words = buildModels(examples, FEATURE_FUNS)
    if TEST:
        test_samples = parse(test_filename, test_pickle)
        actual, predicted = testModels(words, tests = test_samples)
    else:
        actual, predicted = testModels(words)
        print analyze(predicted, actual)
   

    