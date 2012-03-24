'''
    Issues:
        - Figure out how to do multi-line strings without newlines being inserted automatically
'''
import unittest
from main import Collocation

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass
    
    def testFormatTag(self):
        tag = "baby"
        self.assertEquals('@baby@', Collocation.format_tag(tag))
        
    def testGetCollocationReference(self):
        # This test is handwritten, and is dependent on COLLOCATION_BOUNDS being as they were
        # on 3/23/2012
        if Collocation.COLLOCATION_BOUNDS != [(-1,-1), (1,1), (-2, -2), (2,2), (-2, -1), (-1, 1), \
                      (1,2), (-3,-1), (-2,1), (-1,2), (1,3)]:
            raise Exception("COLLOCATION_BOUNDS need to be as they were on 3/23/2012")
        
        tag = 'began'
        context_string = '''He focused , stared , reached out and picked it up. 
        It was light as filigree in his hand , the heel held to the shoe only by 
        one fragment of cleanly cut leather . He held it in his hands and @began@ to blub . 
        Oh master Conroy , do n't ! Don't upset yourself , lovie , do n't take on so !'''
        
        epsilon = Collocation.EPSILON
        
        sentence = "He held it in his hands and @began@ to blub ."
        calculated_dict = Collocation.get_reference([context_string])
        
        self.assertEquals(11, len(calculated_dict))
        expected_dict = {((-1,-1),'and'):0, ((1,1),'to'):1, ((-2,-2), 'hands'):2, ((2,2),'blub'):3, ((-2,-1), 'hands and'):4, ((-1,1), 'and to'):5, \
                        ((1,2), 'to blub'): 6, ((-3,-1), 'his hands and'):7, ((-2,1),'hands and to'):8, ((-1,2), 'and to blub'):9, ((1,3), 'to blub .'):10 }
        self.assertEquals(expected_dict, calculated_dict)
        
    def testGetCollocationVector(self):
        # This test is handwritten, and is dependent on COLLOCATION_BOUNDS being as they were
        # on 3/23/2012
        if Collocation.COLLOCATION_BOUNDS != [(-1,-1), (1,1), (-2, -2), (2,2), (-2, -1), (-1, 1), \
                      (1,2), (-3,-1), (-2,1), (-1,2), (1,3)]:
            raise Exception("COLLOCATION_BOUNDS need to be as they were on 3/23/2012")
        
        tag = 'began'
        context_string = '''He focused , stared , reached out and picked it up. 
        It was light as filigree in his hand , the heel held to the shoe only by 
        one fragment of cleanly cut leather . He held it in his hands and @began@ to blub . 
        Oh master Conroy , do n't ! Don't upset yourself , lovie , do n't take on so !'''
        
        calculated_dict = Collocation.get_reference([context_string])

        class Word():
            def get_collocation_reference_vector(self):
                return calculated_dict
        
        w = Word()
        sample_context = "and @began@ to"
            
        # For reference:    
        # {'and':0, 'to':1, 'hands':2, 'blub':3, 'hands and':4, 'and to':5, \
        #     'to blub': 6, 'his hands and':7, 'hands and to':8, 'and to blub':9, 'to blub .':10 }
        
        expected_collocation_vector = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        calculated_collocation_vector = Collocation.get_vector(sample_context, w)
        
        self.assertEquals(len(calculated_collocation_vector), 11)
        print ""
        self.assertEquals(expected_collocation_vector, calculated_collocation_vector)
        
    def testRemoveTag(self):
        self.assertEquals(['a', 'c'], Collocation.remove_target(['a', '@b@', 'c'], 'b'))

    def testExtractSentenceArray(self):
        context_string = '''He focused , stared , reached out and picked it up. 
        It was light as filigree in his hand , the heel held to the shoe only by 
        one fragment of cleanly cut leather . He held it in his hands and @began@ to blub . 
        Oh master Conroy , do n't ! Don't upset yourself , lovie , do n't take on so !'''
        expected_context_array = ['He', 'held', 'it', 'in', 'his', 'hands', 'and', '@began@', 'to', 'blub', '.']
        self.assertEquals(expected_context_array, Collocation.extract_sentence_array(context_string, 'began'))

    def testEpsilonPad(self):
        # This is a handwritten test for the Collocation bounds constants
        # as of 3/23/2012, 8:37 pm
        if Collocation.MAX_OFFSET != 3 or Collocation.MIN_OFFSET != -3:
            raise Exception("This test is only valid for bounds constants as of 3/23/2012")
        
        context_array = ['@test@', '.']
        
        epsilon = Collocation.EPSILON
        expected_padded_array = [epsilon, epsilon, epsilon, '@test@', '.', epsilon]
        
        self.assertEquals(expected_padded_array, Collocation.epsilon_pad(context_array, 'test'))
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()