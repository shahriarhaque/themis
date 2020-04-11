# IWSPA Email to JSON
import json
from os import listdir
from os.path import isfile, join

# Update these constants as needed
OUTPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'

INPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/raw/Dataset_Submit_Legit/'
OUTPUT_FILE = 'legit-iwspa.txt'

# INPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/raw/Dataset_Submit_Phish/'
# OUTPUT_FILE = 'phish-iwspa.txt'

def process_directory(email_files):
    output_file = OUTPUT_DIRECTORY + OUTPUT_FILE

    all_emails = list()

    for email_file in email_files:
        email = get_email_lines(email_file)
        all_emails.append(email)

    output_list = list()

    for email in all_emails:
        output = dict()
        filename = email["filename"]
        header = construct_header_dict(email)
        body = ' '.join(email["body_lines"]).strip()
        output["filename"] = filename
        output["header"] = header
        output["body"] = body
        output_list.append(output)

    fh = open(output_file, 'w+')
    fh.write(json.dumps(output_list, indent=2))
    fh.close()

def construct_header_dict(email):
    header_lines = email["header_lines"]

    key_indices = [index for index in range(0,len(header_lines)) if is_header_line(header_lines[index])]

    start_end_indices = list()

    for i in range(0, len(key_indices)):
        start = key_indices[i]
        end = -1
        if i+1 < len(key_indices):
            end = key_indices[i+1]
        start_end_indices.append((start, end))

    lines = [' '.join(header_lines[start:end]).strip() if end > 0 else ' '.join(header_lines[start:]).strip() for (start,end) in start_end_indices]

    header_dict = dict()
    dict_keys = [line.split()[0][:-1] for line in lines]
    dict_values = [' '.join(line.split()[1:]) for line in lines]

    for i in range(0, len(dict_keys)):
        header_dict[dict_keys[i]] = dict_values[i]

    return header_dict

def is_header_line(line):
    tokens = line.split()
    if(len(tokens) < 1):
        return False

    first_token = tokens[0]
    return first_token.endswith(':')


def get_email_lines(file):
    all_lines = []
    with open(file) as fp:
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
    header_lines = all_lines[:last_header_line-1]
    body_lines = all_lines[last_header_line:]

    email = dict()
    email["filename"] = file
    email["header_lines"] = header_lines
    email["body_lines"] = body_lines

    return email

def run():
    email_files = [join(INPUT_DIRECTORY, f) for f in listdir(INPUT_DIRECTORY) if isfile(join(INPUT_DIRECTORY, f))]
    process_directory(email_files)

run()
