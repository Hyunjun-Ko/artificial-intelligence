# artificial-intelligence
The projects are written in python3 and while the core functionalities have been written by myself the templates and the environment have been configured by and originated from University of Illinois at Urbana-Champaign. 

## Search
requires python3, pygame, numpy

In .search.py,
def bfs(maze):
def astar_single(maze):
def astar_multiple(maze):
def fast(maze):


This project aims to solve mazes by using different search algorithms
1. Breadth-first search, with one waypoint.
2. A* search, with one waypoint.
3. A* search, with many waypoints.
4. Faster A* search, with many waypoints

The _data_/ directory contains the maze files. Each maze is a simple plaintext file. 

The main.py file is the primary entry point and with 

**python3 main.py --human data/part-1/small** 

it will open a pygame-based interactive visualization of the _data/part-1/small_ maze. 

The blue dot represents the agent. agent can be moved using the arrow keys. The black dots represent the maze waypoints. The agent has to go through a path that reaches all of the waypoints.

you can run and test the algorithms by **python3 main.py "path to maze file" --search "search algorithm name[bfs, astar_single, astar_multiple, fast]"**

## Naive Bayes
requires python3, pygame, numpy, nltk, scikit-learn

In naive_bayes.py,
def naiveBayes():
def bigramBayes():

Given a dataset containing positive and negative reviews, this project uses a Naive Bayes classifier that will predict the right class label given an unseen review. 

**Dataset**

The data set consists of 10,000 positive and 3,000 negative reviews, a subset of the [Stanford Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which was originally introduced by [this paper.](https://aclanthology.org/P11-1015/) The data in split into 5,000 development examples and 8,000 training examples. 

**Unigram Model**

The bag of words model in NLP is a simple unigram model which considers a text to be represented as a bag of independent words.

<img src="Naive_Bayes/img.png" style="height:300px; width:1000px;">

so, the probability of a review being positive given the words in the review. estimated the posterior probabilities and returned the type label with the higher priority. Log probabilities are used to avoid underflow.

**Bigram Model**

Unigram assumes that every word is independent which often does not provide accurate predictions in the real world. Instead, Bigram model processes every two adjacent words together.

we set up the posterior probabilities just as the unigram model into a mixture model defined with a parameter $\lambda$.

<img src="Naive_Bayes/img2.png" style="height:150px; width:800px;">

we need to find the best parameter $\lambda$ that gives the highest classification accuracy.

run **python3 main.py -h** for information on how to run. 