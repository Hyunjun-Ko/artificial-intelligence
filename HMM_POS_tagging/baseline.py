"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    vocab = {}
    for i in range(len(train)):
        for j in train[i]:
            word,tag = j
            if word in vocab:
                if tag in vocab[word]:
                    vocab[word][tag] += 1
                else:
                    vocab[word][tag] = 1
            else:
                dic = {}
                dic[tag] = 1
                vocab[word] = dic

    dic = {}    
    for word in vocab.keys():
        for tag in vocab[word]:
            if tag in dic.keys():
                dic[tag] += vocab[word][tag]
            else:
                dic[tag] = vocab[word][tag]
    max = 0
    f_tag = ''
    for tag in dic.keys():
        if dic[tag] > max:
            max = dic[tag]
            f_tag = tag  
            

    return_list=[]
    r_tag = ''
    for sentence in test:
        r_list = []
        for word in sentence:
            if word in vocab.keys():
                max = 0
                dic = vocab[word]
                for tag in dic.keys():
                    if dic[tag] > max:
                        max = dic[tag]
                        r_tag = tag
                tple = (word,r_tag)
                r_list.append(tple)
            else:
                tple = (word,f_tag)
                r_list.append(tple) 
        return_list.append(r_list)
    return return_list
