import numpy as np
import scipy as sp
from sklearn.svm.sparse import SVC

CACHE_SIZE = 1000
PENALTY = 1.0 #svm penalty parameter

'''
A Word object contains one binary SVM classifying model for each word sense.
'''
class Word(object):
    tag = None
    feature_funs = None
    classez = [] # [n_samples, n_classes]
    contexts = []
    cv_contexts = None
    cv_classez = None
    test_contexts = []
    test_classez = []
    models = []
    
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
        features = map(lambda f: map(lambda c: f(self.tag, c), contexts), self.feature_funs) #[n_features, n_samples]
        avgs = map(sp.mean, features)
        stds = map(sp.std, features)
        for std in stds:
            if std == 0:
                print "Div/0 Error: feature has standard deviation = 0."
        normalized = map( lambda feature, avg, std: map( lambda X: (X-avg)/std, feature), features, avgs, stds)
        return zip(*normalized) # 2-dim array [n_samples, n_features]
        
    def classify(self):
        X = self.calc_features(self.contexts)  
        inv_classez = zip(*self.classez) # 2-dim array [n_classes, n_samples]
        for col in inv_classez:
            #vector [n_samples] of classification for each sample
            Y = np.array(col)
            model = SVC(cache_size = 1000, C = PENALTY, scale_C = True)
            model.fit(X,Y)
            self.models.append(model)
        
    #isTest decides whether to use the cv or test data   
    def predict(self, isTest):
        T = None
        if isTest:
            T = self.calc_features(self.test_contexts)
        else:
            T = self.calc_features(self.cv_contexts)
        results = map (lambda model: model.predict(T), self.models)
        return sum( [list(x) for x in zip(*results)], []) #invert and flatten to 1D array
    
    def get_actual(self, isTest):
        C = None
        if isTest:
            C = self.test_classez
        else:
            C = self.cv_classez
        return sum( C, [])
        
        
        
        
        