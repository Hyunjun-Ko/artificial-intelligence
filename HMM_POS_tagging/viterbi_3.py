"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import defaultdict
import math
def construct (train,test):
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
    return tags,words,tag_pairs,word_tag

def trellis (test,tags,tag_pairs,word_tag,ttUNK,wtUNK,lytag,ingtag,edtag,fultag,nesstag,altag,bletag,menttag,iontag,esttag,acytag,fytag):
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
                        if test[i][k+1].endswith('ly'):
                            emission = lytag[t2]
                        elif test[i][k+1].endswith('ing'):
                            emission = ingtag[t2]
                        elif test[i][k+1].endswith('ed'):
                            emission = edtag[t2]
                        elif test[i][k+1].endswith('ful'):
                            emission = fultag[t2]
                        elif test[i][k+1].endswith('ness'):
                            emission = nesstag[t2]
                        elif test[i][k+1].endswith('al'):
                            emission = altag[t2]
                        elif test[i][k+1].endswith('able') or test[i][k+1].endswith('ible') :
                            emission = bletag[t2]
                        elif test[i][k+1].endswith('ment'):
                            emission = menttag[t2]
                        elif test[i][k+1].endswith('ion'):
                            emission = iontag[t2]
                        elif test[i][k+1].endswith('est'):
                            emission = esttag[t2]
                        elif test[i][k+1].endswith('acy'):
                            emission = acytag[t2]
                        elif test[i][k+1].endswith('ify') or test[i][k+1].endswith('fy'):
                            emission = fytag[t2]
                        else:
                            emission = wtUNK[t2]
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
                            if test[i][k+1].endswith('ly'):
                                emission = lytag[t2]
                            elif test[i][k+1].endswith('ing'):
                                emission = ingtag[t2]
                            elif test[i][k+1].endswith('ed'):
                                emission = edtag[t2]
                            elif test[i][k+1].endswith('ful'):
                                emission = fultag[t2]
                            elif test[i][k+1].endswith('ness'):
                                emission = nesstag[t2]
                            elif test[i][k+1].endswith('al'):
                                emission = altag[t2]
                            elif test[i][k+1].endswith('able') or test[i][k+1].endswith('ible') :
                                emission = bletag[t2]
                            elif test[i][k+1].endswith('ment'):
                                emission = menttag[t2]
                            elif test[i][k+1].endswith('ion'):
                                emission = iontag[t2]
                            elif test[i][k+1].endswith('est'):
                                emission = esttag[t2]
                            elif test[i][k+1].endswith('acy'):
                                emission = acytag[t2]
                            elif test[i][k+1].endswith('ify') or test[i][k+1].endswith('fy'):
                                emission = fytag[t2]
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

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags,words,tag_pairs,word_tag = construct(train, test)

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

    hapaxtag = {} # hapax dictionary to keep track of the hapax words count per tag
    lytag = {}
    ingtag = {}
    edtag = {}
    fultag = {}
    nesstag = {}
    altag = {}
    bletag = {}
    menttag = {}
    iontag = {}
    esttag = {}
    acytag = {}
    fytag = {}
    for tag in tags.keys():
        for word in words.keys():
            if words[word] == 1:    # if it is a hapax word then
                if(word,tag) in word_tag.keys():
                    if word.endswith('ly'):
                        if tag in lytag.keys():
                            lytag[tag] += 1
                        else:
                            lytag[tag] = 1
                    if word.endswith('ing'):
                        if tag in ingtag.keys():
                            ingtag[tag] += 1
                        else:
                            ingtag[tag] = 1
                    if word.endswith('ed'):
                        if tag in edtag.keys():
                            edtag[tag] += 1
                        else:
                            edtag[tag] = 1
                    if word.endswith('ful'):
                        if tag in fultag.keys():
                            fultag[tag] += 1
                        else:
                            fultag[tag] = 1
                    if word.endswith('ness'):
                        if tag in nesstag.keys():
                            nesstag[tag] += 1
                        else:
                            nesstag[tag] = 1
                    if word.endswith('al'):
                        if tag in altag.keys():
                            altag[tag] += 1
                        else:
                            altag[tag] = 1
                    if word.endswith('able') or word.endswith('ible'):
                        if tag in bletag.keys():
                            bletag[tag] += 1
                        else:
                            bletag[tag] = 1
                    if word.endswith('ment'):
                        if tag in menttag.keys():
                            menttag[tag] += 1
                        else:
                            menttag[tag] = 1
                    if word.endswith('ion'):
                        if tag in iontag.keys():
                            iontag[tag] += 1
                        else:
                            iontag[tag] = 1
                    if word.endswith('est'):
                        if tag in esttag.keys():
                            esttag[tag] += 1
                        else:
                            esttag[tag] = 1
                    if word.endswith('acy'):
                        if tag in acytag.keys():
                            acytag[tag] += 1
                        else:
                            acytag[tag] = 1
                    if word.endswith('ify') or word.endswith('fy'):
                        if tag in fytag.keys():
                            fytag[tag] += 1
                        else:
                            fytag[tag] = 1
                    if tag in hapaxtag.keys():
                        hapaxtag[tag] += 1 # count up the hapaxword counts for that given tag
                    else:
                        hapaxtag[tag] = 1
 
    for tag in tags.keys(): # to fill the 0 probabilities
        if tag not in hapaxtag.keys(): # if the tag was not seen with a hapax tag
            hapaxtag[tag] = 0.00001 # i assigned 1 a dummy value to prevent 0 probability
        if tag not in lytag.keys():
            lytag[tag] = 0.00001
        if tag not in ingtag.keys():
            ingtag[tag] = 0.00001
        if tag not in edtag.keys():
            edtag[tag] = 0.00001
        if tag not in fultag.keys():
            fultag[tag] = 0.00001
        if tag not in nesstag.keys():
            nesstag[tag] = 0.00001
        if tag not in altag.keys():
            altag[tag] = 0.00001
        if tag not in bletag.keys():
            bletag[tag] = 0.00001
        if tag not in menttag.keys():
            menttag[tag] = 0.00001
        if tag not in iontag.keys():
            iontag[tag] = 0.00001
        if tag not in esttag.keys():
            esttag[tag] = 0.00001
        if tag not in acytag.keys():
            acytag[tag] = 0.00001
        if tag not in fytag.keys():
            fytag[tag] = 0.00001
    for pair in word_tag.keys():
        tag = pair[1]
        
        wlaplace = wtlaplace*((hapaxtag[tag])/(sum(hapaxtag.values())))  # scaling the smoothing constant by the probability of the word being with that tag
        word_tag[pair] = math.log((word_tag[pair]+wlaplace)/(tags[tag] + (uniquew[tag]+1)*wlaplace))   # tags[tag] is the number of times that tag appeared and uniquew[tag] is the number of unique words seen with that tag
    for pair in tag_pairs.keys():
        tag = pair[1]
        tag_pairs[pair] = math.log((tag_pairs[pair]+ttlaplace)/(tags[tag] + (uniquet[tag]+1)*ttlaplace))
    for tag  in tags.keys():
        
        wlaplace = wtlaplace*((hapaxtag[tag])/(sum(hapaxtag.values())))# scaling the smoothing constant by the probability of the word being with that tag
        lylaplace = abs(wtlaplace*((lytag[tag]/(sum(lytag.values())))))
        inglaplace = abs(wtlaplace*((ingtag[tag]/(sum(ingtag.values())))))
        edlaplace = abs(wtlaplace*((edtag[tag]/(sum(edtag.values())))))
        fullaplace = abs(wtlaplace*((fultag[tag]/(sum(fultag.values())))))
        nesslaplace = abs(wtlaplace*((nesstag[tag]/(sum(nesstag.values())))))
        allaplace = abs(wtlaplace*((altag[tag]/(sum(altag.values())))))
        blelaplace = abs(wtlaplace*((bletag[tag]/(sum(bletag.values())))))
        mentlaplace = abs(wtlaplace*((menttag[tag]/(sum(menttag.values())))))
        ionlaplace = abs(wtlaplace*((iontag[tag]/(sum(iontag.values())))))
        estlaplace = abs(wtlaplace*((esttag[tag]/(sum(esttag.values())))))
        acylaplace = abs(wtlaplace*((acytag[tag]/(sum(acytag.values())))))
        fylaplace = abs(wtlaplace*((fytag[tag]/(sum(fytag.values())))))
        wtUNK[tag] = math.log(wlaplace/(tags[tag] + (uniquew[tag]+1)*wlaplace))
        lytag[tag] = math.log(lylaplace/(tags[tag] + (uniquew[tag]+1)*lylaplace))  
        ingtag[tag] = math.log(inglaplace/(tags[tag] + (uniquew[tag]+1)*inglaplace)) 
        edtag[tag] = math.log(edlaplace/(tags[tag] + (uniquew[tag]+1)*edlaplace))
        fultag[tag] = math.log(fullaplace/(tags[tag] + (uniquew[tag]+1)*fullaplace))
        nesstag[tag] = math.log(nesslaplace/(tags[tag] + (uniquew[tag]+1)*nesslaplace))
        altag[tag] = math.log(allaplace/(tags[tag] + (uniquew[tag]+1)*allaplace))
        bletag[tag] = math.log(blelaplace/(tags[tag] + (uniquew[tag]+1)*blelaplace))
        menttag[tag] = math.log(mentlaplace/(tags[tag] + (uniquew[tag]+1)*mentlaplace))
        iontag[tag] = math.log(ionlaplace/(tags[tag] + (uniquew[tag]+1)*ionlaplace))
        esttag[tag] = math.log(estlaplace/(tags[tag] + (uniquew[tag]+1)*estlaplace))
        acytag[tag] = math.log(acylaplace/(tags[tag] + (uniquew[tag]+1)*acylaplace))
        fytag[tag] = math.log(fylaplace/(tags[tag] + (uniquew[tag]+1)*fylaplace))
        ttUNK[tag] = math.log(ttlaplace/(tags[tag] + (uniquet[tag]+1)*ttlaplace))
    
    return trellis(test,tags,tag_pairs,word_tag,ttUNK,wtUNK,lytag,ingtag,edtag,fultag,nesstag,altag,bletag,menttag,iontag,esttag,acytag,fytag)

