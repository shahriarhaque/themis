import re
import config

def is_acceptable_size(body_text):
    MAX_TOKENS = 1000
    MIN_TOKENS = 50
    MIN_LENGTH = 10
    tokens = body_text.strip().split()
    return len(body_text) > MIN_LENGTH and len(tokens) <= MAX_TOKENS and len(tokens) >= MIN_TOKENS

def remove_consecutive_repeating(body_text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', body_text)

def to_lower_case(txt):
    return txt.lower()

def remove_css_attr(txt):
    txt = re.sub('[\.#:].+ ', '', txt).strip()
    txt = re.sub('.+[\.#:] ', '', txt).strip()
    return txt

def remove_non_alpha(txt):
    return re.sub('[^A-Za-z\s@/\.:\?\$#&]', '', txt).strip()

def remove_hex(txt):
    return re.sub('(0x|0X)[a-fA-F0-9]+', '', txt).strip()

def replace_url(text):
    return re.sub(config.URLREGEX, 'URL', text)

def strip_whitespace(txt):
    txt = re.sub("[\n\r]+", " ", txt)
    txt = re.sub('\s+', ' ', txt).strip()
    return txt

def remove_non_words(txt):
    tokens = txt.split()
    tokens = list(filter(lambda s: is_valid_token(s), tokens))
    return ' '.join(tokens)

def is_acceptable_size(body_text):
    MAX_TOKENS = 1000
    MIN_TOKENS = 100
    MIN_LENGTH = 10
    tokens = body_text.strip().split()
    return len(body_text) > MIN_LENGTH and len(tokens) <= MAX_TOKENS and len(tokens) >= MIN_TOKENS

def is_valid_token(s):
    b = s.isalpha()
    stop_words = ['font', 'bold', 'untitled','margin', 'color',
    'serif', 'px' ,'arial', 'xsmall', 'table', 'cell', 'row',
    'header', 'tr', 'html', 'verdana', 'height', 'fff', 'ccc',
    'border', 'td', 'span', 'align', 'img', 'width', 'outset',
    'dotted', 'pt', 'helvetica', 'body', 'javascript', 'mouse',
    'aaa', 'image', 'img', 'loaded', 'padding', 'solid', 'collapse',
    'message', 'var', 'nav', 'menu', 'hover', 'visited', 'underline',
    'active', 'http', 'div', 'class', 'href', 'src', 'column', 'background'
    'roman', 'times', 'yiv', 'style', 'horizontal', 'microsoft',
    'server', 'smtp', 'communigate', 'cipher']

    full_stop_words = ['with', 'id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'sun', 'mon', 'tue', 'wed', 'thu',
    'fri', 'sat', 'mapi', 'tls', 'postfix', 'utc', 'pro', 'bst', 'cest',
    'pst', 'pdt', 'gmt', 'edt', 'cipher', 'bits']

    not_contains_stop = not(any(sw in s.lower() for sw in stop_words))
    not_contains_full_stop = not(any(sw == s.lower() for sw in full_stop_words))

    b = b and not_contains_stop
    b = b and not_contains_full_stop

    b = b and len(s) < 25

    return b
