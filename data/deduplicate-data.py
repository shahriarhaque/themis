import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def process_lines(file):
    all_lines = []
    with open(file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()

    similarity = dict()

    index1 = 0
    for line in all_lines:
        index2 = 0
        similarity[index1] = list()
        for line2 in all_lines:
            sim = bag_of_words_similarity(line, line2)
            similarity[index1].append((index2, sim))
            index2 = index2 + 1
        index1 = index1 + 1

    SIMILARITY_THRESHOLD = 0.9

    num_messages = len(all_lines)
    all_set = set(list(range(0, num_messages)))
    clusters = list()

    while all_set:
        msg_index = all_set.pop()
        clusters.append(msg_index)
        all_sim = similarity[msg_index]
        all_sim = [sim for sim in all_sim if sim[1] >= SIMILARITY_THRESHOLD]
        print((msg_index+1), "is similar to", [ (sim[0]+1) for sim in all_sim])
        for sim in all_sim:
            if sim[0] in all_set:
                all_set.remove(sim[0])

    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
    phish_body_file = processed_dir + 'phish.txt'
    phish_f = open(phish_body_file, 'w+')

    for index in clusters:
        phish_f.write(all_lines[index])

    phish_f.close()


def company_wise_dedup(all_lines):
    company_names = ['paypal', 'pay pal', 'ebay', 'national city', 'egold',
    'credit union', 'amazon', 'bb t', 'western union', 'sears',
    'capital one', 'wells fargo', 'wellsfargo' ,'chase', 'washington mutual',
    'citizens bank', 'nationwide bank', 'north fork bank',
    'northfork', 'bank of the west', 'united bank',
    'bank of america', 'regions bank', 'barclays',
    'fifth third bank', 'bancorp', 'citibank', 'bank of queensland',
    'm t bank', 'halifax', 'anz', 'bendigo bank', 'commonwealth bank',
    'pnc bank', 'onlinebanking', 'lasalle bank', 'sky bank',
    'armed forces bank', 'south bank', 'commercial federal bank',
    'colonial bank', 'key bank', 'peoples bank', 'hsbc', 'ohio savings',
    'midamerica bank', 'mbna', 'wachovia', 'california bank', 'visa',
    'tcf bank', 'america online', 'aol', 'national bank', 'sierra',
    'cuna', 'nafcu', 'rd bank', 'bank']

    company_counts = dict()

    for company in company_names:
        company_counts[company] = list()

    for line in all_lines:
        familiar_company = False

        for company in company_names:
            if company in line:
                familiar_company = True
                company_counts[company].append(line)
                break

    company_similarity = dict()




    for company in company_counts.keys():
        # print('[', company, ']', sep = '')
        company_lines = company_counts[company]
        company_similarity[company] = dict()
        msg_index = 0
        for line in company_lines:
            # print(msg_index, line)
            company_similarity[company][msg_index] = list()
            line2_index = 0
            for line2 in company_lines:
                similarity =  bag_of_words_similarity(line, line2)
                company_similarity[company][msg_index].append((line2_index, similarity))
                line2_index = line2_index + 1

            msg_index = msg_index + 1

    SIMILARITY_THRESHOLD = 0.7

    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
    phish_body_file = processed_dir + 'phish.txt'
    phish_f = open(phish_body_file, 'w+')

    for company in company_counts.keys():
        num_messages = len(company_similarity[company].keys())
        all_set = set(list(range(0, num_messages)))
        clusters = list()

        while all_set:
            msg_index = all_set.pop()
            clusters.append(msg_index)
            all_sim = company_similarity[company][msg_index]
            all_sim = [sim for sim in all_sim if sim[1] >= SIMILARITY_THRESHOLD]
            # print(msg_index, "is similar to", [sim[0] for sim in all_sim])
            for sim in all_sim:
                if sim[0] in all_set:
                    all_set.remove(sim[0])

        for index in clusters:
            phish_f.write(company_counts[company][index])

    phish_f.close()

def bag_of_words_similarity(a, b):
    a_words = set(a.split())
    b_words = set(b.split())

    all_words = a_words | b_words
    common_words = a_words & b_words

    return len(common_words) / len(all_words)


def calculate_frequency_distribution(all_lines):
    all_text = ' '.join(all_lines)

    stop_words = set(stopwords.words('english'))

    custom_stop_words = set(['would', 'said', 'us', 'link', 'e'])

    stop_words = stop_words | custom_stop_words

    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word not in stop_words]


    # Calculate frequency distribution
    fdist = nltk.FreqDist(tokens)

    # Output top 50 words
    for word, frequency in fdist.most_common(20):
        print(u'{}\t{}'.format(word, frequency))

def run():
    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/interim/'
    phish_body_file = processed_dir + 'phish-phase2.txt'
    legit_body_file = processed_dir + 'legit.txt'


    process_lines(phish_body_file)
    #process_lines(legit_body_file)



run()
