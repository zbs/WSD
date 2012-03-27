'''
Need to run "nltk.download()" in Python and install "maxent_treebank_pos_tagger" under "Models" to use posNeighbors()

Feature functions are of form f(word, context) => [features]
'''
import re, cPickle
import nltk as nl
import Collocation

pos_tags = {}
pos_tags_changed = False

POS_TYPES = { 'CC':0, 'CD':1, 'DT':2, 'EX':3, 'FW':4, 'IN':5, 'JJ':6, 'JJR':7, 'JJS':8, 'LS':9, 'MD':10, 'NN':11, 'NNS':12, 'NNP':13,
              'NNPS':14, 'PDT':15, 'POS':16, 'PRP':17, 'PRP$':18, 'RB':19, 'RBR':20, 'RBS':21, 'RP':22, 'SYM':23, 'TO':24, 'UH':25,
              'VB':26, 'VBD':27, 'VBG':28, 'VBN':29, 'VBP':30, 'VBZ':31, 'WDT':32, 'WP':33, 'WP$':34, 'WRB':35, ',':36, ':':36, '``':36, 
              "''":36, '-NONE-':37 }
NUM_POS_TYPES = 38

def posNeighbors(word, context):
    #find the target and up to 3 words on each side
    result =  re.search( '(?P<left>(\S+ ){1,5})@(?P<target>\S+)@(?P<right>( \S+){1,5})', context)
    left = re.split( '[?!\.]', result.group('left'))[-1] #don't include words in another sentence
    right = re.split( '[?!\.]', result.group('right'))[0]
    target = result.group('target')
    sequence = left + target + right
    sequence = sequence.replace('@', '')  # '@bank@ to @bank@' would otherwise cause error  
    pos_neighbors = []
    if sequence in pos_tags: #if it's been pickled
        pos_neighbors = pos_tags[sequence]
    else:
        pos_tags_changed = True #there's been a change so will have to dump pickle data later
        tokens = nl.word_tokenize(sequence)
        pos = nl.pos_tag( tokens )
        target_index = tokens.index(target)
        for i in range(3): #find the three pos to left of target
            if target_index - i - 1 >= 0:
                pos_neighbors.append( pos[target_index - i -1])
            else:
                pos_neighbors.append(('', '-NONE-'))
        for i in range(3): # find 3 to right
            if target_index + i + 1 < len(pos):
                pos_neighbors.append( pos[target_index + i + 1])
            else:
                pos_neighbors.append(('', '-NONE-'))
        pos_tags[sequence] = pos_neighbors
    #print sequence
    #print pos_neighbors
    features = [ 0 for _ in range( NUM_POS_TYPES*len(pos_neighbors) ) ] #binary feature vector - 36+ POS for each neighbor
    for i in range( len(pos_neighbors) ):
        #turn on the corresponding feature for the pos of each neighbors
        features[ NUM_POS_TYPES*i + POS_TYPES[ pos_neighbors[i][1] ] ] = 1
    return features
    
def cooccurrences(word,context):
    context_tokens = nl.word_tokenize(word.clean_string(context))
    stemmer = nl.stem.porter.PorterStemmer()
    
    features = [0]*len(word.tokens)
    
    #if context_token has been seen before, increment the associated counter
    for i in context_tokens:
        root = (stemmer.stem(i)).lower()
        if root in word.tokens:
            features[word.tokens[root]] = 1
    return features
    
def collocations(word, context):
    return Collocation.get_vector(context, word)

# serves as a bypass so that we can keep code structure untouched
# (for use in baseline calc)
def most_frequent(word,context):
    return []
