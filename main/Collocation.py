#import run
#import Word
import re

COLLOCATION_BOUNDS = [(-1,-1), (1,1), (-2, -2), (2,2), (-2, -1), (-1, 1), \
                      (1,2), (-3,-1), (-2,1), (-1,2), (1,3)]
MAX_OFFSET = 0
MIN_OFFSET = 0

OUTPUT = True
ONCE = True
WORD = None

EPSILON = '\xcf\xb5'

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

def get_tag_from_context(context):
    result = re.search( '@(?P<target>\S+)@', context )
    return result.group('target')

def remove_target(array, tag):
    try:
        index = array.index(format_tag(tag))
    except ValueError:
        raise Exception("Malformed context")
    
    del array[index]
    return array


def get_vector(context, word):
    global OUTPUT
    
    reference_vector = word.get_collocation_reference_vector()
    # collocations_list is a binary feature vector. We initialize it to all zeroes, and change an entry
    # only if it corresponds to a collocation that exists in the given context
    collocations_list = [0] * len(reference_vector)

    tag = get_tag_from_context(context)
    context_array = epsilon_pad(extract_sentence_array(context, tag), tag)
    if OUTPUT:
        print "Producing collocation vector for tag %s with context %s."%(tag, context)
        print "Extracted sentence: %s"%(" ".join(context_array))
    try:
        target_index = context_array.index(format_tag(tag))
    except IndexError:
        raise Exception("Malformed context")
    #Iterate across all collocations as defined in the Collocation module
    for lower, upper in COLLOCATION_BOUNDS:
        relative_lower, relative_upper = target_index + lower, target_index + upper
        if lower < 0 and upper > 0:
            collocation = ' '.join(remove_target(context_array[relative_lower:relative_upper + 1], tag))
        else:
            collocation = ' '.join(context_array[relative_lower:relative_upper + 1])
        
        collocation_key = ((lower,upper), collocation)
        if OUTPUT:
            print "C_%d_%d = %s"%(lower, upper, collocation)
        if collocation_key in reference_vector:
            collocation_index = reference_vector[collocation_key]
            collocations_list[collocation_index] = 1
    if OUTPUT:
        OUTPUT = False
    return collocations_list

def get_reference(contexts):
    index = 0
    colloc_locations = {}
    
    for context in contexts: # Extract the sentence containing the target word, as for collocations,
        # we do not cross sentence boundaries.
        # Additionally, pad the sentence with epsilon so that no collocation will incur
        # an IndexError.
        tag = get_tag_from_context(context)
        context_array = epsilon_pad(extract_sentence_array(context, tag), tag)
        try:
            target_index = context_array.index(format_tag(tag))
        except IndexError:
            raise Exception("Malformed context")
        #Iterate across all collocations as defined in the Collocation module
        for lower, upper in COLLOCATION_BOUNDS:
            relative_lower, relative_upper = target_index + lower, target_index + upper
            if lower < 0 and upper > 0:
                collocation = ' '.join(remove_target(context_array[relative_lower:relative_upper + 1], tag))
            else:
                collocation = ' '.join(context_array[relative_lower:relative_upper + 1])
            #Do not add duplicates
            collocation_key = ((lower,upper), collocation)
            if collocation_key not in colloc_locations:
                colloc_locations[collocation_key] = index
                if collocation == "Limewashing":
                    print collocation_key
                index += 1
            
    return colloc_locations

MIN_OFFSET, MAX_OFFSET = COLLOCATION_BOUNDS[0]
for lower, upper in COLLOCATION_BOUNDS:
    MIN_OFFSET, MAX_OFFSET = lower if lower < MIN_OFFSET else MIN_OFFSET, upper if upper > MAX_OFFSET else MAX_OFFSET
