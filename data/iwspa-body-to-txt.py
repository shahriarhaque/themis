import re
from os import listdir
from os.path import isfile, join
import textutils

import nltk

from nltk.corpus import brown

def process_directory(legit_email_files):
    output_dir = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
    # legit_body_file = output_dir + 'legit.txt'
    legit_body_file = output_dir + 'phish-iwspa.txt'

    all_bodies = list()
    for legit_file in legit_email_files:
        body_text = get_body_text(legit_file)
        all_bodies.append(body_text)

    all_bodies = [process_body_text(body_text) for body_text in all_bodies]
    all_bodies = [body_text for body_text in all_bodies if is_acceptable_size(body_text)]

    legit_f = open(legit_body_file, 'w+')

    for body_text in all_bodies:
        legit_f.write(body_text)
        legit_f.write('\n')

    legit_f.close()

def is_acceptable_size(body_text):
    MAX_TOKENS = 1000
    MIN_TOKENS = 100
    MIN_LENGTH = 10
    tokens = body_text.strip().split()
    return len(body_text) > MIN_LENGTH and len(tokens) <= MAX_TOKENS and len(tokens) >= MIN_TOKENS

def get_body_text(legit_file):
    all_lines = []
    with open(legit_file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()

    last_header_line = 0
    current_line = -1

    for line in all_lines:
        current_line = current_line + 1
        if is_header_line(line):
            # print(current_line, is_header_line(line), line)
            last_header_line = current_line


    last_header_line = last_header_line + 2
    body_lines = all_lines[last_header_line:]
    body_text = ' '.join(body_lines).strip()

    return body_text

def process_file(legit_file):
    all_lines = []
    with open(legit_file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()

    body_text = process_lines(all_lines)
    return body_text

def process_body_text(body_text):

    body_text = textutils.remove_non_alpha(body_text)
    body_text = textutils.remove_non_words(body_text)
    body_text = textutils.strip_whitespace(body_text)
    body_text = body_text.lower()
    body_text = remove_consecutive_repeating(body_text)

    # body_text = remove_non_dictionary_words(body_text)

    return body_text

def remove_consecutive_repeating(body_text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', body_text)

def remove_non_dictionary_words(body_text):
    tokens = body_text.split()
    whitelist = set(brown.words())
    tokens = [t for t in tokens if t in whitelist]
    new_body = ' '.join(tokens)
    return new_body

def is_header_line(line):
    tokens = line.split()
    if(len(tokens) < 1):
        return False

    first_token = tokens[0]
    return first_token.endswith(':')

def run():
    iwspa_dir = '/Users/shahriar/Documents/Research/Code/themis/data/raw/Dataset_Submit_Legit/'
#    iwspa_dir = '/Users/shahriar/Documents/Research/Code/themis/data/raw/Dataset_Submit_Phish/'

     legit_email_files = [ str(num) + '.txt' for num in list(range(1,4083))]
#    legit_email_files = [ str(num) + '.txt' for num in list(range(1,504))]

    legit_email_files = [iwspa_dir + f for f in legit_email_files]
    process_directory(legit_email_files)

run()
