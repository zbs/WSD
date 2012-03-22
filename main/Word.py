#import numpy as np
#import scipy as sp
from sklearn import svm
from re import match

CACHE_SIZE = 1000
PENALTY = 1 #svm penalty parameter

'''
A Word object contains one binary SVM classifying model for each word sense.
'''
class Word(object):
    tag = None
    feature_funs = None
    classez = [] # [n_samples, n_classes], "z" stands for the plural of classes
    contexts = []
    cv_contexts = None
    cv_classez = None
    num_classes = None
    test_contexts = []
    test_classez = []
    model = None
    
    def __init__(self, tag, feature_funs):
        self.tag = tag
        self.feature_funs = feature_funs
    
    #isTest determines whether the data is train or test.   
    def add_sample(self, classes, context, isTest = False):
        if isTest:
            self.test_context.append(context)
            self.test_classez.append(classes)
        else:
            self.contexts.append(context)
            self.classez.append(classes)
    
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
        return [ sum ( [f(self, c) for f in self.feature_funs], [] ) for c in contexts] #[n_features, n_samples]
            
    def classify(self):
        self.num_classes = len(self.classez[0])
        X = self.calc_features(self.contexts)
        # Converts binary classification to an int.
        # If there are multiple classifications, takes first one.
        Y = map( lambda c: c.index(1), self.classez) #[n_samples] of classification for each sample
        if(len(X) != len(Y)): print "X and Y different lengths in classify()"
        #linear multiclass svm
        self.model = svm.LinearSVC( C = PENALTY, multi_class = True, scale_C = True)
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
                #Ensure only alphabetical words are included in vocabulary
                if match("^[a-zA-Z]$", word) and word != self.tag:
                    all_words.add(word)
        list.sort(all_words)
        return all_words
        
        
        