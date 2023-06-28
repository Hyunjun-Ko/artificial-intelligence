# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.05, pos_prior=0.8,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    pos = Counter()
    neg = Counter()
    for i in range(len(train_set)):
        rating = train_labels[i]
        if rating == 1: # if positive review
            for word in train_set[i]:
                if word in pos:
                    pos[word] = pos[word] + 1
                else:
                    pos[word] = 1
        else: # if negative review
            for word in train_set[i]:
                if word in neg:
                    neg[word] = neg[word] + 1
                else:
                    neg[word] = 1
    
    yhats = []
    pos_val = sum(pos.values()) # total number of words in the positive review dict
    neg_val = sum(neg.values()) # total number of words in the negative review dict
    p_wt = len(pos) # distinct word types
    n_wt = len(neg) # distinct word types 

    for doc in tqdm(dev_set,disable=silently):
        #print(doc)
        p_wp = math.log(1) # initializing the
        p_wn = math.log(1) # probability of word give positive and negative
        for word in doc: # for each word in the doc
            if word in pos.keys():
                p_wp = math.log((pos[word] + laplace)/(pos_val + (p_wt+1)*laplace)) + p_wp # p(word|positive) = (number of words in positive reviews + laplace) / 
            else:                                                              # (total number of words in positive review + word types in pos review * laplace) 
                p_wp = math.log(laplace/(pos_val + (p_wt+1)*laplace)) + p_wp

            if word in neg.keys():
                p_wn = math.log((neg[word] + laplace)/(neg_val + (n_wt+1)*laplace)) + p_wn
            else:
                p_wn = math.log(laplace/(neg_val + (n_wt+1)*laplace)) + p_wn
        
        p_wp = p_wp + math.log(pos_prior) # multiply by the probability of positive review to get finalized probability of the review being positive given the words
        p_wn = p_wn + math.log(1-pos_prior) 

        if p_wp > p_wn:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8, silently=False):

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    pos = Counter()
    neg = Counter()
    pair_pos = Counter()
    pair_neg = Counter()
    for i in range(len(train_set)):
        rating = train_labels[i]
        if rating == 1: # if positive review
            for word in train_set[i]:
                if word in pos:
                    pos[word] = pos[word] + 1
                else:
                    pos[word] = 1
            for j in range(len(train_set[i])-1):
                first = train_set[i][j]
                second = train_set[i][j+1]
                pair = [first,second]
                pair = tuple(pair)
                if pair in pair_pos:
                    pair_pos[pair] += 1
                else:
                    pair_pos[pair] = 1
        else: # if negative review
            for word in train_set[i]:
                if word in neg:
                    neg[word] = neg[word] + 1
                else:
                    neg[word] = 1
            for j in range(len(train_set[i])-1):
                first = train_set[i][j]
                second = train_set[i][j+1]
                pair = [first,second]
                pair = tuple(pair)
                if pair in pair_neg:
                    pair_neg[pair] += 1
                else:
                    pair_neg[pair] = 1


    yhats = []
    pos_val = sum(pos.values()) # total number of words in the positive review dict
    neg_val = sum(neg.values()) # total number of words in the negative review dict
    pair_pos_val = sum(pair_pos.values())
    pair_neg_val = sum(pair_neg.values())
    p_wt = len(pos) # distinct word types
    n_wt = len(neg) # distinct word types
    pair_p_wt = len(pair_pos)
    pair_n_wt = len(pair_neg)

    for doc in tqdm(dev_set,disable=silently):
        #print(doc)
        p_wp = math.log(1) # initializing the
        p_wn = math.log(1) # probability of word give positive and negative
        pair_p_wp = math.log(1)
        pair_p_wn = math.log(1)
        for word in doc: # for each word in the doc
            if word in pos.keys():
                p_wp = math.log((pos[word] + unigram_laplace)/(pos_val + (p_wt+1)*unigram_laplace)) + p_wp # p(word|positive) = (number of words in positive reviews + laplace) / 
            else:                                                              # (total number of words in positive review + word types in pos review * laplace) 
                p_wp = math.log(unigram_laplace/(pos_val + (p_wt+1)*unigram_laplace)) + p_wp

            if word in neg.keys():
                p_wn = math.log((neg[word] + unigram_laplace)/(neg_val + (n_wt+1)*unigram_laplace)) + p_wn
            else:
                p_wn = math.log(unigram_laplace/(neg_val + (n_wt+1)*unigram_laplace)) + p_wn
        
        p_wp = p_wp + math.log(pos_prior) # multiply by the probability of positive review to get finalized probability of the review being positive given the words
        p_wn = p_wn + math.log(1-pos_prior)

        for i in range(len(doc)-1):
            first = doc[i]
            second = doc[i+1]
            pair = [first,second]
            pair = tuple(pair)
            if pair in pair_pos.keys():
                pair_p_wp += math.log((pair_pos[pair] + bigram_laplace)/(pair_pos_val + (pair_p_wt+1)*bigram_laplace))
            else:
                pair_p_wp += math.log(bigram_laplace/(pair_pos_val + (pair_p_wt+1)*bigram_laplace))

            if pair in pair_neg.keys():
                pair_p_wn += math.log((pair_neg[pair] + bigram_laplace)/(pair_neg_val + (pair_n_wt+1)*bigram_laplace))
            else:
                pair_p_wn += math.log(bigram_laplace/(pair_neg_val + (pair_n_wt+1)*bigram_laplace))

        pair_p_wp += math.log(pos_prior)
        pair_p_wn += math.log(1-pos_prior)
        positive_p = (1-bigram_lambda)*p_wp + bigram_lambda*pair_p_wp                
        negative_p = (1-bigram_lambda)*p_wn + bigram_lambda*pair_p_wn
        if positive_p > negative_p:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats

