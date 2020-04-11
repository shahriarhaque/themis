# iwspa-preprocess.py
import json
import textutils


INPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'
# INPUT_FILE = 'legit-iwspa.txt'
INPUT_FILE = 'phish-iwspa.txt'

OUTPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
# OUTPUT_FILE = 'legit-iwspa.txt'
OUTPUT_FILE = 'phish-iwspa.txt'

MIN_HEADER_TOKENS = 1
MAX_HEADER_TOKENS = 20

def read_input_emails():
    input_file_path = INPUT_DIRECTORY + INPUT_FILE
    with open(input_file_path, "r") as file:
        email_list = json.load(file)

    return email_list

def write_output_email(email_list):
    output_file =  OUTPUT_DIRECTORY + OUTPUT_FILE
    fh = open(output_file, 'w+')
    fh.write(json.dumps(email_list, indent=2))
    fh.close()

def run():
    email_list = read_input_emails()

    body_preprocess_pipeline = [
        textutils.to_lower_case,
        textutils.remove_css_attr
    ]

    header_preprocess_pipeline = [
        (textutils.to_lower_case, 'Subject')
    ]

    for email in email_list:
        email['qualify'] = True
        for func in body_preprocess_pipeline:
            apply_function_to_body(email, func)
        for func_key in header_preprocess_pipeline:
            apply_function_to_header(email, func_key)

    for email in email_list:
        flag_unacceptable_size(email)
        flag_missing_critical_features(email)
        flag_unacceptable_header_size(email)

    write_output_email(email_list)


def apply_function_to_header(email, func_key):
    func, key = func_key
    if 'header' in email and key in email['header']:
        header_text = email['header'][key]
        header_text = func(header_text)
        email['header'][key] = header_text

def apply_function_to_body(email, func):
    body_text = email['body']
    body_text = func(body_text)
    email['body'] = body_text

def flag_missing_critical_features(email):
    if 'body' in email and 'header' in email and 'Subject' in email['header']:
        pass
    else:
        email['rejected-for'] = 'Missing Subject or Body'
        email['qualify'] = False

def flag_unacceptable_header_size(email):
    if 'body' in email and 'header' in email and 'Subject' in email['header']:
        subject_text = email['header']['Subject']
        subject_tokens = subject_text.split()

        if len(subject_tokens) < MIN_HEADER_TOKENS or len(subject_tokens) > MAX_HEADER_TOKENS:
            email['rejected-for'] = 'Unacceptable Header Size'
            email['qualify'] = False

def flag_unacceptable_size(email):
    qualify = textutils.is_acceptable_size(email['body'])
    if not qualify:
        email['rejected-for'] = 'Unacceptable Body Size'
        email['qualify'] = False


run()
