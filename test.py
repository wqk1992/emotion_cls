from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale



model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
 

model1 = Word2Vec.load_word2vec_format('vectors.txt', binary=False) 
#model = Word2Vec.load_word2vec_format('vectors.bin', binary=True) 


print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5))
 
#[(u'queen', 0.711819589138031),
#(u'monarch', 0.618967592716217),
#(u'princess', 0.5902432799339294),
#(u'crown_prince', 0.5499461889266968),
#(u'prince', 0.5377323031425476)]



print(model.most_similar(positive=['biggest','small'], negative=['big'], topn=5))

 
#[(u'smallest', 0.6086569428443909),
#(u'largest', 0.6007465720176697),
#(u'tiny', 0.5387299656867981),
#(u'large', 0.456944078207016),
#(u'minuscule', 0.43401968479156494)]


#model.most_similar(positive=['ate','speak'], negative=['eat'], topn=5)
 
#[(u'spoke', 0.6965223550796509),
#(u'speaking', 0.6261293292045593),
#(u'conversed', 0.5754593014717102),
#(u'spoken', 0.570488452911377),
#(u'speaks', 0.5630602240562439)]


import numpy as np
 
with open('food_words.txt', 'r') as infile:
    food_words = infile.readlines()
 
with open('sports_words.txt', 'r') as infile:
    sports_words = infile.readlines()
 
with open('weather_words.txt', 'r') as infile:
    weather_words = infile.readlines()
 
def getWordVecs(words):
    vecs = []
    for word in words:
        word = word.replace('n', '')
        try:
            vecs.append(model[word].reshape((1,300)))
        except KeyError:
            continue
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float') 
 
food_vecs = getWordVecs(food_words)
sports_vecs = getWordVecs(sports_words)
weather_vecs = getWordVecs(weather_words)