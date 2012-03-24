#import run
#import Word

COLLOCATION_BOUNDS = [(-1,-1), (1,1), (-2, -2), (2,2), (-2, -1), (-1, 1), \
                      (1,2), (-3,-1), (-2,1), (-1,2), (1,3)]
MAX_OFFSET = 0
MIN_OFFSET = 0

EPSILON = "^!^"

def get_colloc_vector(tag, context, words):
    """
    word = run.get_word(words, tag)
    vocabulary = word.get_vocabulary()
    vocab_size = len(vocabulary)
    context_words = context.split(" ")
    try:
        target_index = context_words.index('@'+tag+'@')
    except ValueError:
        raise Exception("Target word not contained in context")
        
    if not word:
        raise Exception("Tag not found in list of words")
    vector = [0] * vocab_size ** (UPPER_BOUND - LOWER_BOUND - 1) #-1 for excluding the target word

    for i in range(LOWER_BOUND, UPPER_BOUND+1): #shift everything to the right by LOWER_BOUND
        if i == 0:
            continue
    """

def format_tag(tag):
    return '@' + tag + '@'

#how do you test for physical equality?
def extract_sentence_array(context, tag):
    tag_encountered = False
    
    context_array = context.split(' ')
    sub_array = []
    
    for item in context_array:
        
        sub_array.append(item)
        if item == format_tag(tag):
            tag_encountered = True
        if item == '.':
            if tag_encountered:
                return sub_array 
            else:
                sub_array = []
    return sub_array

def epsilon_pad(extracted_sentence_array, tag):
    
    try:
        index = extracted_sentence_array.index(format_tag(tag))
    except ValueError:
        raise Exception("Malformed context")
        
    min_lower_index, max_upper_index = index + MIN_OFFSET, index + MAX_OFFSET

    extracted_sentence_array += [EPSILON] * (max_upper_index - len(extracted_sentence_array))
    extracted_sentence_array = [EPSILON] * (-1 * min_lower_index ) + extracted_sentence_array
    
    return extracted_sentence_array

def remove_target(array, tag):
    try:
        index = array.index(format_tag(tag))
    except ValueError:
        raise Exception("Malformed context")
    
    del array[index]
    return array

def get_collocation_vector(context, tag, word):
    
    reference_vector = word.get_collocation_reference_vector()
    # collocations_list is a binary feature vector. We initialize it to all zeroes, and change an entry
    # only if it corresponds to a collocation that exists in the given context
    collocations_list = [0] * len(reference_vector)

    context = epsilon_pad(extract_sentence_array(context, tag), tag)
    
    try:
        target_index = context.index(format_tag(tag))
    except IndexError:
        raise Exception("Malformed context")
    #Iterate across all collocations as defined in the Collocation module
    for lower, upper in COLLOCATION_BOUNDS:
        relative_lower, relative_upper = target_index + lower, target_index + upper
        if relative_lower < 0 and relative_upper > 0:
            collocation = ' '.join(remove_target(context[relative_lower:relative_upper + 1], tag))
        else:
            collocation = ' '.join(context[relative_lower:relative_upper + 1])
        if collocation in reference_vector:
            collocation_index = reference_vector[collocation]
            collocations_list[collocation_index] = 1
    return collocations_list
    
    

def get_collocation_reference(contexts, tag):
    index = 0
    colloc_locations = {}
    
    for context in contexts: # Extract the sentence containing the target word, as for collocations,
        # we do not cross sentence boundaries.
        # Additionally, pad the sentence with epsilon so that no collocation will incur
        # an IndexError.
        context = epsilon_pad(extract_sentence_array(context, tag), tag)
        try:
            target_index = context.index(format_tag(tag))
        except IndexError:
            raise Exception("Malformed context")
        #Iterate across all collocations as defined in the Collocation module
        for lower, upper in COLLOCATION_BOUNDS:
            relative_lower, relative_upper = target_index + lower, target_index + upper
            if relative_lower < 0 and relative_upper > 0:
                collocation = ' '.join(remove_target(context[relative_lower:relative_upper + 1], tag))
            else:
                collocation = ' '.join(context[relative_lower:relative_upper + 1])
            #Do not add duplicates
            if collocation not in colloc_locations:
                colloc_locations[collocation] = index
                index += 1
    return colloc_locations

min, max = COLLOCATION_BOUNDS[0]
for lower, upper in COLLOCATION_BOUNDS:
    min, max = lower if lower < min else min, upper if upper > max else max
MIN_OFFSET, MAX_OFFSET = min, max