import csv,os

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



vocab = dict()

with open('books.csv','r',encoding='utf-8',newline='') as file:
    read = csv.reader(file)
    next(read)
    for row in read:
        summary_data = row[3]
        midway2 = ''.join(char for char in summary_data if char not in punctuation_set)
        words = midway2.lower().split()
        filtered = [word for word in words if word not in stopwords]
        stemmed = []
        for word in filtered:
            stemmed.append(stemmer(word))
        stemmed = list(set(stemmed))

        soup = [row[1]] * 5 + [row[2]] * 3 + stemmed

        for word in soup:
            if word in vocab.keys():
                vocab[word]+=1
            else:
                vocab[word] = 1


with open("vocab.csv",'w',encoding='utf-8',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['word', 'frequency'])  # Write header

    for word, freq in vocab.items():
        writer.writerow([word, freq])


