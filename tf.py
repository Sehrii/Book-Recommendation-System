import csv,os,copy
import numpy as np
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
    'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
    'wasn', 'weren', 'won', 'wouldn', 'okay', 'yeah', 'hi', 'hey'
]
stopwords = set(stopwords)

punctuation_set = {'!', '.', ',', '?', ':', ';', '-', '(', ')', '"', "'", '“', '”', '’', '—', '–'}
def stemmer(word):
    suffixes = ['ing', 'ed', 'ly', 'es', 's', 'ment', 'ness', 'tion']
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

vocab_keys = []
with open("vocab.csv",'r',encoding='utf-8',newline='') as file:
    reader = csv.reader(file)
    next(reader)
    for word in reader:
        vocab_keys.append(word[0])
print(vocab_keys)
vocab_size = len(vocab_keys)
vocab = {word:0 for word in vocab_keys}


with open("books.csv",'r',encoding='utf-8',newline='') as file:
    matrix =[]
    reader = csv.reader(file)
    next(reader)
    count =0
    for row in reader:
        count+=1
        total_count=0
        vocabulary = copy.deepcopy(vocab)

        summary = row[3]
        midway2 = ''.join(char for char in summary if char not in punctuation_set)
        words = midway2.lower().split()
        filtered = [word for word in words if word not in stopwords]
        stemmed = []
        for word in filtered:
            stemmed.append(stemmer(word))
        total_terms_in_summary = len(stemmed)
        for word in stemmed:
            total_count+=1
            if word in vocabulary.keys():
                vocabulary[word]+=1

        vector = np.zeros(vocab_size)
        for i in range(vocab_size):
            vector[i]=vocabulary[vocab_keys[i]]
        vector = vector/float(total_terms_in_summary)
        matrix.append(vector)
        print("book ",count," added")
matrix = np.array(matrix)
np.savetxt('output.csv', matrix, delimiter=',', fmt='%.10e')
np.save('tf.npy', matrix)









