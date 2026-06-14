import csv,numpy

import numpy as np

number_of_books = 5101
vocab_keys = []
idf = []
with open("vocab.csv",'r',encoding='utf-8',newline='') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        number_of_doc_with_term = int(row[1])
        idf.append(np.log(number_of_books/number_of_doc_with_term))
idf = np.array(idf)
print(idf)
np.save('idf.npy', idf)


