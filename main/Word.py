#import numpy as np
#import scipy as sp
from nltk import word_tokenize
from sklearn import svm
from nltk.stem.porter import *
import re, heapq
import Collocation

CACHE_SIZE = 1000
PENALTY = .1 #svm penalty parameter
STOP_WORDS = ["i","a","about","an","are","as","at","be","by","for","from","how",
              "in","is","it","of","on","or","that","the","this","to","was","what",
              "when","where","who","will","with","the"]

'''
A Word object contains one binary SVM classifying model for each word sense.
'''
class Word(object):
    
    def __init__(self, tag, feature_funs):
        self.tag = tag
        self.feature_funs = feature_funs
        self.classez = [] # [n_samples, n_classes], "z" stands for the plural of classes
        self.contexts = []
        self.cv_contexts = []
        self.cv_classez = None
        self.num_classes = None
        self.test_contexts = []
        self.test_classez = []
        self.model = None
        self.tokens = None
        self.collocation_reference_vector = None
        
    #isTest determines whether the data is train or test.   
    def add_sample(self, classes, context, isTest = False):
        if isTest:
            self.test_contexts.append(context)
            self.test_classez.append(classes)
        else:
            self.contexts.append(context)
            self.classez.append(classes)
    
    #Calls all the subroutines necessary after Word object populated with samples
    def setup(self, isTest):
        if not isTest:
            self.splitCV(.10)
        self.build_context_list()
        self.classify()
        #garbage collect
        self.classez = self.contexts = None
        
    #split up the train data by word for cross-validation
    #we could make this more advanced
    def splitCV(self, percent):
        p = int( len(self.contexts) * percent)
        self.cv_contexts = self.contexts[:p]
        self.contexts = self.contexts[p:]
        self.cv_classez = self.classez[:p]
        self.classez = self.classez[p:]
            
    #apply each feature to each example and normalize
    def calc_features(self, contexts):
        '''
        Only need to normalize if not using binary features
        features = map(lambda f: map(lambda c: f(self.tag, c), contexts), self.feature_funs) #[n_features, n_samples]
        avgs = map(sp.mean, features)
        stds = map(sp.std, features)
        for std in stds: #sanity check
            if std == 0:
                print "Div/0 Error: feature has standard deviation = 0."
        normalized = map( lambda feature, avg, std: map( lambda X: (X-avg)/std, feature), features, avgs, stds)
        return zip(*normalized)  # 2-dim array [n_samples, n_features]
        '''
        #For each context - concatenates the vectors from each feature function.
        return [ sum ( [f(self, c) for f in self.feature_funs], [] ) for c in contexts] #[n_samples, n_features]
            
    def classify(self):
        self.num_classes = len(self.classez[0])
        X = self.calc_features(self.contexts)
        # Converts binary classification to an int.
        # If there are multiple classifications, takes first one.
        Y = map( lambda c: c.index(1), self.classez) #[n_samples] of classification for each sample
        if(len(X) != len(Y)): print "X and Y different lengths in classify()"
        #linear multiclass svm
        self.model = svm.LinearSVC( C = PENALTY, multi_class = True, scale_C = True)
        #self.model = svm.SVC( C = PENALTY, scale_C = True)
        self.model.fit(X,Y)
        
        
    #isTest decides whether to use the cv or test data   
    def predict(self, isTest):
        T = self.calc_features( self.test_contexts if isTest else self.cv_contexts )
        #convert int classification to binary
        #OK fine, it was easier to not use a map here
        predictions = self.model.predict(T)
        return sum([[int(i == j) for i in range(self.num_classes)] for j in predictions], [])
    
    def get_actual(self, isTest):
        C = self.test_classez if isTest else self.cv_classez
        return sum( C, [])
    
    def get_vocabulary(self):
        """ 
            Return a list representing all unique words encountered in all contexts of a given word, 
            sorted lexicographically
        """
        all_words = set()
        #Get all contexts by concatenating the partitioned groups
        for context in (self.contexts + self.cv_contexts):
            for word in context.split(' '):
                #Don't include target word
                if word != "@"+self.tag+"@":
                    all_words.add(word)
        list.sort(all_words)
        return all_words
        
    # builds dictionary of all tokens in all contexts
    #   key: token (root form, lowercase)
    #   value: index for converting to list
    def build_context_list(self):
        stemmer = PorterStemmer()
        self.tokens = {}
        index_count = 0
        for i in range(len(self.classez)):
            context = self.remove_stop_words(
                        word_tokenize(
                          self.clean_string(
                            self.contexts[i])))
            _class = self.classez[i]

            for word in context:
                root = (stemmer.stem(word)).lower()
                if root not in self.tokens:
                    self.tokens[root] = index_count
                    index_count += 1
                    
    #removes @word@ from context
    #takes in a tokenized string, not the string itself..
    def clean_string(self,string):
        temp = re.sub('@\S+@', '', string)
        return re.sub("[^a-z0-9A-Z\ ']",'',temp)
        
    def remove_stop_words(self,tokens):
        for word in tokens:
            if word.lower() in STOP_WORDS:
                tokens.remove(word)
        return tokens
    
    def get_collocation_reference_vector(self):
        if not self.collocation_reference_vector:
                self.collocation_reference_vector = \
                    Collocation.get_reference(self.contexts + self.cv_contexts, self.tag)
        return self.collocation_reference_vector



    
                        
