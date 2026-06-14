import numpy as np

tf = np.load('tf.npy')
idf = np.load(('idf.npy'))

TfIdf = tf*idf
print(TfIdf)
np.save('TfIdf.npy', TfIdf)
np.savetxt('TfIdf.csv', TfIdf, delimiter=',', fmt='%.10e')