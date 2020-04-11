import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd

STOP_WORDS = set(stopwords.words('english'))


def calculate_frequency_distribution(file):
    all_lines = []
    with open(file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()    
    
    all_text = ' '.join(all_lines)

    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word not in STOP_WORDS]


    # Calculate frequency distribution
    fdist = nltk.FreqDist(tokens)

    word_freq_pairs = list()
    NUM_TOP_WORDS = 50

    # Output top 50 words
    for word, frequency in fdist.most_common(NUM_TOP_WORDS):
        word_freq_pairs.append((word, frequency))

    df = pd.DataFrame(word_freq_pairs, columns = ['Word', 'Frequency'])
    df.sort_values('Frequency')
    
    return df

def draw_bar_chart(df):
    print(df.head())
    
    
    words = df['Word'].tolist()
    freqs = df['Frequency'].tolist()
    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=90)
    plt.show()
    

def main():
    processed_dir = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
    phish_body_file = processed_dir + 'phish.txt'
    legit_body_file = processed_dir + 'legit.txt'


#    df = calculate_frequency_distribution(phish_body_file)
    df = calculate_frequency_distribution(legit_body_file)
    
    draw_bar_chart(df)
    
    



if __name__== "__main__":
    main()

