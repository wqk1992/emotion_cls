
import gensim
from sklearn.cross_validation import train_test_split
import numpy as np



LabeledSentence = gensim.models.doc2vec.LabeledSentence
 

 
with open('IMDB_data/pos.txt','r') as infile:
    pos_reviews = infile.readlines()
 
with open('IMDB_data/neg.txt','r') as infile:
    neg_reviews = infile.readlines()
 
with open('IMDB_data/unsup.txt','r') as infile:
    unsup_reviews = infile.readlines()
 
# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
 
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
 
# 预处理
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('n','') for z in corpus]
    corpus = [z.replace('&lt;br /&gt;', ' ') for z in corpus]
 
    # 将标点视为一个单词
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus
 
x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)
 
# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
 
x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')





import random
 
size = 400
 
# 实例化 DM 和 DBOW 模型
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
 
# 对所有评论创建词汇表
model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
 
# 多次传入数据集，通过每次滑动（shuffling）来提高准确率。
all_train_reviews = np.concatenate((x_train, unsup_reviews))
for epoch in range(10):
    perm = np.random.permutation(all_train_reviews.shape[0])
    model_dm.train(all_train_reviews[perm])
    model_dbow.train(all_train_reviews[perm])
 
# 从我们的模型中获得训练过的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)
 
train_vecs_dm = getVecs(model_dm, x_train, size)
train_vecs_dbow = getVecs(model_dbow, x_train, size)
 
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
 
# 训练测试数据集
x_test = np.array(x_test)
 
for epoch in range(10):
    perm = np.random.permutation(x_test.shape[0])
    model_dm.train(x_test[perm])
    model_dbow.train(x_test[perm])
 
# 创建测试数据集向量
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)
 
test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

#再次使用 SGDClassifier

from sklearn.linear_model import SGDClassifier
 
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
 
print ('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))



#ROC 
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

