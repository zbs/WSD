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
    result =  re.search( '(?P<left>(\S+ ){1,3})@(?P<target>\S+)@(?P<right>( \S+){1,3})', context)
    left = re.split( '[?!\.]', result.group('left'))[-1] #don't include words in another sentence
    right = re.split( '[?!\.]', result.group('right'))[0]
    left =left.replace('@', '') # '@bank@ to @bank@' would otherwise cause error
    right = right.replace('@', '')
    target = result.group('target')
    sequence = left + target + right
    pos = None
    if sequence in pos_tags:
        pos = pos_tags[sequence]
    else:
        pos_tags_changed = True
        left_tokens = nl.word_tokenize(left)
        right_tokens = nl.word_tokenize(right)
        if len(left_tokens) >3 : left_tokens = left_tokens[-3:] # tokenizer splits "cannot" => "can" not"
        if len(right_tokens) > 3 : right_tokens = right_tokens[:3] 
        #tokenize left, target, and right together
        pos = nl.pos_tag( left_tokens + [target] + right_tokens) #Uses Penn Treebank with 36+ POS 
        #Add '-NONE-' tags to pos if there are less than 3 left or right neighbors
        for i in range(3 - len(left_tokens)):
            pos.insert(0, ('', '-NONE-'))
        for i in range(3 - len(right_tokens)):
            pos.append( ('', '-NONE-'))
        pos.pop(3) #remove the target word
        pos_tags[sequence] = pos
    #print pos
    if (len(pos) != 6):
        print "len(pos)!=6 in in posNeighbors, figure out what went wrong"
    features = [ 0 for _ in range( NUM_POS_TYPES*len(pos) ) ] #binary feature vector - 36+ POS for each neighbor
    for i in range( len(pos) ):
        #turn on the corresponding feature for the pos of each neighbors
        features[ NUM_POS_TYPES*i + POS_TYPES[ pos[i][1] ] ] = 1
    return features
    
def cooccurrances(word,context):
    context_tokens = nl.word_tokenize(word.clean_string(context))
    stemmer = nl.stem.porter.PorterStemmer()
    
    features = [0]*len(word.tokens)
    
    #if context_token has been seen before, increment the associated counter
    for i in context_tokens:
        root = (stemmer.stem(i)).lower()
        if root in word.tokens:
            features[word.tokens[root]] += 1
    return features
    
def collocations(word, context):
    return Collocation.get_collocation_vector(context, word.tag, word)
