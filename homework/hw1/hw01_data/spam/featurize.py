'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict, Counter
import glob
import re
import scipy.io

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_at_feature(text, freq):
    return text.count('@')

#---spam---#
def freq_td_feature(text, freq):
    return float(freq['td'])

def freq_nbsp_feature(text, freq):
    return float(freq['nbsp'])

def freq_src_feature(text, freq):
    return float(freq['src'])

def freq_cialis_feature(text, freq):
    return float(freq['cialis'])

def freq_img_feature(text, freq):
    return float(freq['img'])

def freq_htmlimg_feature(text, freq):
    return float(freq['htmlimg'])

def freq_intel_feature(text, freq):
    return float(freq['intel'])

def freq_free_feature(text, freq):
    return float(freq['free'])

def freq_or_feature(text, freq):
    return float(freq['or'])

def freq_3_feature(text, freq):
    return float(freq['3'])

def freq_underscore_feature(text, freq):
    return float(freq['_'])

def freq_our_feature(text, freq):
    return float(freq['our'])

def freq_http_feature(text, freq):
    return float(freq['http'])

def freq_all_feature(text, freq):
    return float(freq['all'])

#----Ham----#
def freq_ect_feature(text, freq):
    return float(freq['ect'])

def freq_xls_feature(text, freq):
    return float(freq['xls'])

def freq_strangers_feature(text, freq):
    return float(freq['strangers'])

def freq_cotten_feature(text, freq):
    return float(freq['cotten'])

def freq_spreadsheet_feature(text, freq):
    return float(freq['spreadsheet'])

def freq_replica_feature(text, freq):
    return float(freq['replica'])


def freq_2000_feature(text, freq):
    return float(freq['2000'])

def freq_that_feature(text, freq):
    return float(freq['that'])

def freq_be_feature(text, freq):
    return float(freq['be'])

def freq_on_feature(text, freq):
    return float(freq['on'])

def freq_i_feature(text, freq):
    return float(freq['i'])

def freq_we_feature(text, freq):
    return float(freq['we'])

def freq_from_feature(text, freq):
    return float(freq['from'])

def freq_have_feature(text, freq):
    return float(freq['have'])

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    feature.append(freq_at_feature(text, freq))
    '''
    feature.append(freq_td_feature(text, freq))
    feature.append(freq_nbsp_feature(text, freq))
    feature.append(freq_src_feature(text, freq))
    feature.append(freq_cialis_feature(text, freq))
    feature.append(freq_img_feature(text, freq))
    feature.append(freq_htmlimg_feature(text, freq))
    feature.append(freq_intel_feature(text, freq))
    feature.append(freq_ect_feature(text, freq))
    feature.append(freq_xls_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_strangers_feature(text, freq))
    feature.append(freq_replica_feature(text, freq))
    feature.append(freq_spreadsheet_feature(text, freq))
    '''
    feature.append(freq_2000_feature(text, freq))
    feature.append(freq_or_feature(text, freq))
    feature.append(freq_3_feature(text, freq))
    feature.append(freq_underscore_feature(text, freq))
    feature.append(freq_that_feature(text, freq))
    feature.append(freq_be_feature(text, freq))
    feature.append(freq_our_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_all_feature(text, freq))
    feature.append(freq_on_feature(text, freq))
    feature.append(freq_i_feature(text, freq))
    feature.append(freq_we_feature(text, freq))
    feature.append(freq_from_feature(text, freq))
    feature.append(freq_have_feature(text, freq))

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    whole_freq = defaultdict(int)
    
    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1
                whole_freq[word] += 1

            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)


    d = Counter(whole_freq)
    # Create a feature vector
    for k, v in d.most_common(50):
        print (k, v)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
print("spam")
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
print("ham")
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)

