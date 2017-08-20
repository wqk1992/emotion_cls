
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


#############################################
 
ts = TSNE(2)
reduced_vecs = ts.fit_transform(np.concatenate((food_vecs, sports_vecs, weather_vecs)))
 

for i in range(len(reduced_vecs)):
    if i &lt; len(food_vecs):
        #food words  blue
        color = 'b'
    elif i &gt;= len(food_vecs) and i &lt; (len(food_vecs) + len(sports_vecs)):
        #sports words  red
        color = 'r'
    else:
        #weather words  green
        color = 'g'
    plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=color, markersize=8)




 
with open('twitter_data/pos_tweets.txt', 'r') as infile:
    pos_tweets = infile.readlines()
 
with open('twitter_data/neg_tweets.txt', 'r') as infile:
    neg_tweets = infile.readlines()
 
# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))
 
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)
 
# 零星的预处理
def cleanText(corpus):
    corpus = [z.lower().replace('n','').split() for z in corpus]
    return corpus
 
x_train = cleanText(x_train)
x_test = cleanText(x_test)
 
n_dim = 300
# 初始化模型并创建词汇表
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)
 
# 训练模型 
imdb_w2v.train(x_train)


# 对训练数据集创建词向量，接着进行比例缩放（scale）。
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec



train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)
 
 
# 在测试推特数据集中训练 Word2Vec
imdb_w2v.train(x_test)



# 创建测试推特向量并缩放
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)



# 使用分类算法（逻辑回归 
 
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
 
print ('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))



# 创建 ROC 曲线
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
pred_probas = lr.predict_proba(test_vecs)[:,1]
 
fpr,tpr,_ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
 
 
plt.show()



