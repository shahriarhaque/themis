def process_lines(file):
    all_lines = []
    with open(file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()

    filtered_lines = all_lines

    filtered_lines = filter_mails_by_company(all_lines)
    filtered_lines = [body_text for body_text in filtered_lines if is_acceptable_size(body_text)]

    print(len(filtered_lines))
#
#    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'
#    phish_body_file = processed_dir + 'phish-phase2.txt'
#
#    phish_f = open(phish_body_file, 'w+')
#
#    for line in filtered_lines:
#        phish_f.write(line)
#        phish_f.write('\n')
#
#    phish_f.close()



def is_acceptable_size(body_text):
    MAX_TOKENS = 1000
    MIN_TOKENS = 100
    MIN_LENGTH = 10
    tokens = body_text.strip().split()
    return len(body_text) > MIN_LENGTH and len(tokens) <= MAX_TOKENS and len(tokens) >= MIN_TOKENS



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

def filter_mails_by_company(all_lines):

    company_names = get_company_names()
    filtered_lines = list()

    for line in all_lines:
        familiar_company = False

        for company in company_names:
            if company in line:
                familiar_company = True

        if familiar_company:
            filtered_lines.append(line)

    return filtered_lines






def run():
    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'
    phish_body_file = processed_dir + 'phish.txt'
#    legit_body_file = processed_dir + 'legit.txt'


    process_lines(phish_body_file)
    #process_lines(legit_body_file)



run()
