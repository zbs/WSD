train_filename = "../train.txt"
test_filename = "../test.txt"
kaggle_filename = "../kaggle.txt"

train_pickle = "train_pickle.txt"
test_pickle = "test_pickle.txt"
pos_tag_pickle  ="pos_tag_pickle.txt"

import re, cPickle, os, cProfile
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from Word import Word
import features
# http://inclass.kaggle.com/c/cornell-cs4740-word-sense-disambiguation

FEATURE_FUNS = (features.pos_tags, features.cooccurrances,) #add some more
LIMIT_WORDS = False #true to test and/or train with limited # words

LIMIT = 3
REPICKLE_FILES = False #set to True to reparse data files
REPICKLE_POS_TAGS = False #set to True to reset pos-tags
TEST = True #true to run test set and write kaggle file
# the percent of words that is put in the stop_word list
STOP_WORD_PERCENT = 0.1


def parse(filename, pickle_file):
    if not REPICKLE_FILES and os.path.isfile(pickle_file):
        return cPickle.load(open(pickle_file, 'rb')) 
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
    cPickle.dump(samples, open(pickle_file, 'wb'))
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
    if len(actual) != len(predicted): print "actual and predicted different length"
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
    
def get_stop_words(examples):
    # key: word
    # value: number of times seen
    count_dict = {}
    
    # key: integer {1,2...}
    # value: words that have been seen this many times
    words_by_count = {1:[]}
    
    stemmer = PorterStemmer()
    word_count = 0
    top_words = []
    # generate word counts and words per count
    for (_,_,context) in examples:
        tokens = word_tokenize(clean_string(context))
        for word in tokens:
            word_count += 1
            word = stemmer.stem(word)
            if word not in count_dict:
                count_dict[word] = 1
                words_by_count[1].append(word)
            else:
                i = count_dict[word]
                words_by_count[i].remove(word)
                if i+1 not in words_by_count:
                    words_by_count[i+1] = []
                words_by_count[i+1].append(word)
                count_dict[word] += 1
    
    stop_list_count = int(word_count * STOP_WORD_PERCENT)
    print stop_list_count
    while stop_list_count > 0:
        max = len(words_by_count)
        if not words_by_count[max]:
            del words_by_count[max]
        else:
            top_words.append(words_by_count[max].pop())
            stop_list_count -= 1
    return top_words

def clean_string(str):
    temp = re.sub('@\S+@', '', str)
    return re.sub("[^a-z0-9A-Z\ ']",'',temp)

def main():
    #load pos_tag pickle file if it exists
    if os.path.isfile(pos_tag_pickle) and not REPICKLE_POS_TAGS:
        features.pos_tags = cPickle.load(open(pos_tag_pickle, 'rb'))
        
    examples = parse(train_filename, train_pickle)
    words = buildModels(examples, FEATURE_FUNS)
    if TEST:
        test_samples = parse(test_filename, test_pickle)
        predicted = testModels(words, tests = test_samples)
    else:
        actual, predicted = testModels(words)
        print analyze(predicted, actual)
        
    if features.pos_tags_changed:
        cPickle.dump(features.pos_tags, open(pos_tag_pickle, 'wb'))
    print "Done!"
        
if __name__ == '__main__':
    '''
    lst = []
    lst.append((1,[1,0],"I went @bass@ fishing"))
    lst.append((1,[0,1],"I am playing @bass@ ball"))
    lst.append((1,[0,1],"I fish for @bass@ fish"))
    counts = get_stop_words(lst)
    print counts
    
    '''
    #look at run options at top of file.
    #cProfile.run('main()')
    main()
    