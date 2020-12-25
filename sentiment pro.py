#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import string
import re
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean(tokens):

    result =[]

    for i in range(len(tokens)):
        for pun in pn:
            for c in pun:
                tokens[i] = tokens[i].replace(c, '')

    for tok in tokens:
        if tok not in stopwords_list:
            if tok != '':
                result.append(tok)

    return result
                
positive = twitter_samples.strings('positive_tweets.json')
negative = twitter_samples.strings('negative_tweets.json')
# train and test data
train_pos_len = (len(positive) * 3) // 4
train_positive = positive[:train_pos_len]
test_positive = positive[train_pos_len:]

train_neg_len = (len(negative) * 3) // 4
train_negative = negative[:train_neg_len]
test_negative = negative[train_neg_len:]

tp=" ".join(train_positive)
#print(tp)
tn=" ".join(train_negative)
postokens = tp.split()    # get token
negtokens = tn.split()
#print(postokens)
#print(negtokens)
stopwords_list = stopwords.words('english')
pn = list(string.punctuation)

#***********************/ lemmatization/*********
def lemm(tokeen,text):  
    sentence_words=nltk.word_tokenize(text)
    for word in sentence_words:
        if word in pn:
            sentence_words.remove(word)
    wnl = nltk.WordNetLemmatizer()
    myls =[wnl.lemmatize(t) for t in tokeen]
    return myls

postokens = clean(lemm(postokens,tp)) # postokens
negtokens = clean(lemm(negtokens,tn))

#print(postokens)
#print(negtokens)

def get_count_dict(lst):
    count = {}
    for item in lst:
        if item in count:
            count[item] = count[item] + 1
        else:
            count[item] = 1
    return count

#**********Binarized multinomial naive bayes ******

#***** priors*****
pos_count = get_count_dict(postokens)
neg_count = get_count_dict(negtokens)
vocab_count = len(pos_count) + len(neg_count)

pos_prior = len(pos_count) / vocab_count
# neg_prior can be 1 - pos_proir as well.
neg_prior = len(neg_count) / vocab_count

# ******conditional probabilities****

def calc_prob(count, count_all):
    return (count + 1) / (count_all + vocab_count)

def calc_prob_dict(tokens):   
    result = {}
    for token in tokens:
        result[token] = calc_prob(tokens[token], len(tokens))

    return result
        
pos_probs = calc_prob_dict(pos_count)
neg_probs = calc_prob_dict(neg_count)

def choose_class(words):
 
    pos_prob = pos_prior
    neg_prob = neg_prior
    
    def get_prob(word, probs):
        if word in probs.keys():
            return probs[word]
        else:
            return 1 / (len(probs) + vocab_count)
    
    for word in words:
        pos_prob = pos_prob * get_prob(word, pos_probs)
        neg_prob = neg_prob * get_prob(word, neg_probs)
    
    return pos_prob - neg_prob

def test():
    pos_correct = 0
    neg_correct = 0
    
    pos_wrong = 0
    neg_wrong = 0
    
    for pos in test_positive:
        if (choose_class(pos) < 0):
            pos_wrong = pos_wrong + 1
        else:
            pos_correct = pos_correct + 1
            
    for neg in test_negative:
        if (choose_class(pos) > 0):
            neg_wrong = neg_wrong + 1
        else:
            neg_correct = neg_correct + 1
        
    #print("Positive:")
    #print("Correct: " + str(pos_correct) + ", Wrong: " + str(pos_wrong))
    #print("Negative:")
    #print("Correct: " + str(neg_correct) + ", Wrong: " + str(neg_wrong))
    
    count_all = pos_correct + pos_wrong + neg_correct + neg_wrong
    percentage = ((pos_correct + neg_correct) / count_all) * 100
    print("Correct Percentage: " + str(percentage) + "%")
    

def classify(sentence):
    if (choose_class(sentence.split()) > 0):
        print("Positive")
    else:
        print("Negative")

test()
while True:
    classify(input())
    




# In[ ]:




