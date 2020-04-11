import mailbox
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from bs4 import BeautifulSoup
import utils
import config
import re
import random
import textutils

import nltk
from nltk.corpus import brown



# BeautifulSoup does not work with Python 3, use bs4 instead

def get_company_names():
    company_names = ['paypal', 'pay pal', 'ebay', 'national city', 'egold',
    'credit union', 'amazon', 'bb t', 'western union', 'sears',
    'capital one', 'wells fargo', 'wellsfargo' ,'chase', 'washington mutual',
    'citizens bank', 'nationwide bank', 'north fork bank',
    'northfork', 'bank of the west', 'united bank',
    'bank of america', 'regions bank', 'rd bank', 'barclays',
    'fifth third bank', 'bancorp', 'citibank', 'bank of queensland',
    'm t bank', 'halifax', 'anz', 'bendigo bank', 'commonwealth bank',
    'pnc bank', 'onlinebanking', 'lasalle bank', 'sky bank',
    'armed forces bank', 'south bank', 'commercial federal bank',
    'colonial bank', 'key bank', 'peoples bank', 'hsbc', 'ohio savings',
    'midamerica bank', 'mbna', 'wachovia', 'california bank', 'visa',
    'tcf bank', 'america online', 'aol', 'national bank', 'sierra',
    'cuna', 'nafcu', 'bank']

    return company_names


def processFile(filepath, phishy=True, limit=500):
    mbox = mailbox.mbox(filepath)

    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'
    phish_body_file = processed_dir + 'phish.txt'

#    phish_f = open(phish_body_file, 'w+')

    index = 0
    counts = {
        'total' : 0,
        'html_parse' : 0,
        'non_alpha' : 0,
        'wspace_norm' : 0,
        'web_jargon' : 0,
        'cleaned' : 0,
        
    }
    
    for message in mbox:
        # index = 14 #random.randint(0,2278)
        # message = mbox[index]
        
        txt = process_message(message, counts)
        word_tokens = create_word_tokens(txt)
        txt = ' '.join(word_tokens)
        

#        if is_invalid_message(word_tokens):
#            continue
#        else:
        print('Total Char Count', counts['total'])
        print('HTML Count', counts['html_parse'])
        print('Non Alpha', counts['non_alpha'])
        print('Norm Whitespace', counts['wspace_norm'])
        print('Web Jargon', counts['web_jargon'])
        print('Clean', counts['cleaned'])
        
        


#        phish_f.write(txt)
#        phish_f.write('\n')
        index = index + 1

    print("Total Emails", index)
#    phish_f.close()

def compute_english_percentage(word_tokens):
    english_count = 0
    rand_indices = [random.randint(0, len(word_tokens)-1) for i in range(20)]

    for index in rand_indices:
        if word_tokens[index] in words.words():
             english_count = english_count + 1

    return (english_count / 20.0 ) * 100.0

def is_invalid_message(word_tokens):
    if len(word_tokens) < 20:
        return True
    english_perc = compute_english_percentage(word_tokens)
    # print(english_perc)
    if english_perc < 50.0:
        return True

    return False


def create_word_tokens(txt):
    tokens = word_tokenize(txt)
    tokens = list(filter(lambda s: textutils.is_valid_token(s), tokens))
    return tokens


def keep_whitelisted_words(line, whitelist):
    tokens = word_tokenize(line)
    tokens = [word for word in tokens if word in whitelist]
    processed_line = ' '.join(tokens)
    return processed_line
    
def remove_blacklisted_words(line, blacklist):
    tokens = word_tokenize(line)
    tokens = [word for word in tokens if word not in blacklist]
    processed_line = ' '.join(tokens)
    return processed_line
    
def create_black_list(all_lines):
    all_text = ' '.join(all_lines)

    stop_words = set(brown.words())


    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word not in stop_words]


    # Calculate frequency distribution
    fdist = nltk.FreqDist(tokens)

    index = 0
    blacklist = list()
    for word, frequency in fdist.most_common(1500):
        # print(index, u'{}\t{}'.format(word, frequency))
        # print(word)
        blacklist.append(word)
        index = index + 1

    return set(blacklist)

def create_white_list():
    whitelist = []

    cnames = get_company_names()

    for company in cnames:
        tokens = company.split()
        for token in tokens:
            whitelist.append(token)

    return set(whitelist)
    
def create_black_list2():
    blacklist = [
        'hex', 'engine', 'function', 'type', 'x', 'media', 'o', 'n', 'revision',
        'stack', 'file', 'f', 'o', 'm', 'l', 'path', 'formulas', 'banner',
        'blank', 'spacing', 'root', 'ex', 'start', 'interface', 'k', 'y', 'cu',
        'p', 'he', 'size', 'center', 'auto', 'label', 'none', 'level', 'background',
        'veranda', 'text', 'h', 'alert', 'sheet', 'red', 'blue', 'normal', 'ol',
        'ul', 'wrapper', 'component', 'dashboard', 'group', 'overflow', 'hidden',
        'global', 'position', 'relative', 'input', 'form', 'input', 'em', 'r',
        'left', 'right', 'middle', 'top', 'bottom', 'w', 'baseline', 'pad',
        'include', 'source', 'define', 'g', 'en'
    ]

    return set(blacklist)        

def process_message(message, counts):
    
    
    html = process_payload(message)
    current = len(html)
    counts['total'] = counts['total'] + current
    
    #print(html)
    #print()
    #print('-------------')
    txt = parse_html(html)
    txt = textutils.remove_hex(txt)
    txt = textutils.replace_url(txt)
    
    current = current - len(txt)
    counts['html_parse'] = counts['html_parse'] + current
    current = len(txt)
    
    #txt = remove_css_attr(txt)
    txt = textutils.remove_non_alpha(txt)
    current = current - len(txt)
    counts['non_alpha'] = counts['non_alpha'] + current
    current = len(txt)
    
    txt = textutils.strip_whitespace(txt)
    current = current - len(txt)
    counts['wspace_norm'] = counts['wspace_norm'] + current
    current = len(txt)
    
    txt = txt.lower()
    txt = keep_whitelisted_words(txt, whitelist | brown_corpus)
    txt = remove_blacklisted_words(txt, create_black_list2())
    txt = textutils.remove_consecutive_repeating(txt)
    
    current = current - len(txt)
    counts['web_jargon'] = counts['web_jargon'] + current
    current = len(txt) 
    
    counts['cleaned'] = counts['cleaned'] + current
    
    return txt



# code taken from
# https://matix.io/extract-text-from-webpage-using-beautifulsoup-and-python/
def parse_html(html):
    # Do not use this parser
    #soup = BeautifulSoup(html, 'html.parser')

    # This parser is much more lenient
    soup = BeautifulSoup(html, 'html5lib')
    text = soup.find_all(text=True)

    output = ''
    blacklist = [
    	'[document]',
    	'noscript',
    	'header',
    	'html',
    	'meta',
    	'head',
    	'input',
    	'script',
        #'font',
    ]

    for t in text:
    	if t.parent.name not in blacklist:
    		output += '{} '.format(t)

    return output

def process_payload(message):
    payload = utils.getpayload_dict(message)
    payload_str = ''

    for p in payload:
        payload_str += p['payload']
        payload_str += ' '

    return payload_str

def run():
    processFile("../../data/raw/phishing3.mbox", limit=2279)

whitelist = create_white_list()
brown_corpus = set(brown.words())
run()
