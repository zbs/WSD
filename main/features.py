'''
Need to run "nltk.download()" in Python and install "maxent_treebank_pos_tagger" under "Models" to use posNeighbors()

Feature functions are of form f(word, context) => [features]
'''
import re
import nltk as nl

POS_TYPES = { 'CC':0, 'CD':1, 'DT':2, 'EX':3, 'FW':4, 'IN':5, 'JJ':6, 'JJR':7, 'JJS':8, 'LS':9, 'MD':10, 'NN':11, 'NNS':12, 'NNP':13,
              'NNPS':14, 'PDT':15, 'POS':16, 'PRP':17, 'PRP$':18, 'RB':19, 'RBR':20, 'RBS':21, 'RP':22, 'SYM':23, 'TO':24, 'UH':25,
              'VB':26, 'VBD':27, 'VBG':28, 'VBN':29, 'VBP':30, 'VBZ':31, 'WDT':32, 'WP':33, 'WP$':34, 'WRB':35, ',':36, ':':36, '``':36, '-NONE-':37, 'NULL':38}
NUM_POS_TYPES = 39

def posNeighbors(word, context):
    #find the target and up to 3 words on each side
    result =  re.search( '(?P<left>(\S+ ){1,3})@(?P<target>\S+)@(?P<right>( \S+){1,3})', context)
    left = re.split( '[?!\.]', result.group('left'))[-1] #don't include words in another sentence
    right = re.split( '[?!\.]', result.group('right'))[0]
    left =left.replace('@', '') # '@bank@ to @bank@' would otherwise cause error
    right = right.replace('@', '')
    target = result.group('target')
    left_tokens = nl.word_tokenize(left)
    right_tokens = nl.word_tokenize(right)
    #'cannot' is tokenized as ['can', 'not'], which is annoying
    if 'not' in left_tokens: left_tokens.remove('not')
    if 'not' in right_tokens: right_tokens.remove('not')
    #tokenize left, target, and right together
    pos = nl.pos_tag( left_tokens + [target] + right_tokens) #Uses Penn Treebank with 36+ POS 
    #Add 'NULL' tags to pos if there are less than 3 left or right neighbors
    for i in range(3 - len(left_tokens)):
        pos.insert(0, ('', 'NULL'))
    for i in range(3 - len(right_tokens)):
        pos.append( ('', 'NULL'))
    pos.pop(3) #remove the target word
    if (len(pos) != 6):
        print "len(pos)!=6 in in posNeighbors, figure out what went wrong"
    features = [ 0 for _ in range( NUM_POS_TYPES*len(pos) ) ] #binary feature vector - 36+ POS for each neighbor
    for i in range( len(pos) ):
        #turn on the corresponding feature for the pos of each neighbors
        features[ NUM_POS_TYPES*i + POS_TYPES[ pos[i][1] ] ] = 1
    return features