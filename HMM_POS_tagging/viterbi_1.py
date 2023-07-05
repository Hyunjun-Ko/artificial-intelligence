"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags = {}
    words = {}
    tag_pairs = {}
    word_tag = {}
    for i in range(len(train)):
        for j in train[i]:
            word,tag = j
            if tag in tags.keys():
                tags[tag] += 1
            else:
                tags[tag] = 1

            if j in word_tag.keys():
                word_tag[j] += 1
            else:
                word_tag[j] = 1

            if word in words.keys():
                words[word] += 1
            else:
                words[word] = 1
        for k in range(len(train[i])-1):
            ftag = train[i][k][1]
            stag = train[i][k+1][1]
            pair = (stag,ftag)
            if pair in tag_pairs.keys():
                tag_pairs[pair] += 1
            else:
                tag_pairs[pair] = 1
    wtlaplace = 0.00001
    wtUNK = {}
    ttlaplace = 0.00001
    ttUNK = {}
    uniquew = {}
    for tag in tags.keys():
        counter = 0
        for word in words.keys():
            if (word,tag) in word_tag.keys():
                counter +=1
        uniquew[tag] = counter
    uniquet = {}
    for tag1 in tags.keys():
        counter = 0
        for tag2 in tags.keys():
            if (tag2,tag1) in tag_pairs.keys():
                counter +=1
        uniquet[tag1] = counter
    

    for pair in word_tag.keys():
        tag = pair[1]
        word_tag[pair] = math.log((word_tag[pair]+wtlaplace)/(tags[tag] + (uniquew[tag]+1)*wtlaplace))   
    for pair in tag_pairs.keys():
        tag = pair[1]
        tag_pairs[pair] = math.log((tag_pairs[pair]+ttlaplace)/(tags[tag] + (uniquet[tag]+1)*ttlaplace))
    for tag  in tags.keys():
        wtUNK[tag] = math.log(wtlaplace/(tags[tag] + (uniquew[tag]+1)*wtlaplace))  
        ttUNK[tag] = math.log(ttlaplace/(tags[tag] + (uniquet[tag]+1)*ttlaplace))
    

    return_list = []
    for i in range(len(test)):
        vmatrix = [{} for col in range(len(test[i])-1)]
        bmatrix = [{} for col in range(len(test[i])-1)]
        
        for k in range(len(test[i])-1):
            
            column = {}
            for t2 in tags:
                tagA = {}
                if k == 0:
                    
                    if (t2,'START') in tag_pairs.keys():
                        transition = tag_pairs[(t2,'START')]
                    else:
                        transition = ttUNK['START']
                    
                    if (test[i][k+1],t2) in word_tag.keys(): 
                        emission = word_tag[(test[i][k+1],t2)]
                    else:
                        emission = wtUNK['START']
                    v_prev = word_tag[('START','START')]
                    probability = transition + emission + v_prev
                    vmatrix[k][t2] = probability
                    bmatrix[k][t2] = 'START'
                    column[t2] = probability

                else:
                    for t1 in tags:
                        if (t2,t1) in tag_pairs.keys():
                            transition = tag_pairs[(t2,t1)]
                        else:
                            transition = ttUNK[t1]
                        
                        if(test[i][k+1],t2) in word_tag.keys():
                            emission = word_tag[(test[i][k+1],t2)]
                        else:
                            emission = wtUNK[t2]
                        v_prev = vmatrix[k-1][t1]
                        probability = transition + emission + v_prev
                        tagA[t1] = probability
                    argmax = max(tagA, key=tagA.get)
                    maxval = tagA[argmax]
                    vmatrix[k][t2] = maxval
                    bmatrix[k][t2] = argmax
                    column[t2] = maxval
            final = max(column, key = column.get) 
        taglist = []
        taglist2 =[]
        taglist3 = []
        taglist.append(final)
        for k in reversed(range(len(test[i])-1)):
            taglist.append(bmatrix[k][final])
            final = bmatrix[k][final]

        while taglist:
            taglist2.append(taglist.pop())
        for k in range(len(test[i])):
            tple = (test[i][k],taglist2[k])
            taglist3.append(tple)
        return_list.append(taglist3) 
    return return_list
