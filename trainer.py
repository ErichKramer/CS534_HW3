#!/usr/bin/env python


#from collections import defaultdict
from modDict import myclass as defaultdict
import sys
from math import log
startsym, stopsym = "<s>", "</s>"

def readfile(filename):
    for line in open(filename):
        wordtags = [x.rsplit("/", 1) for x in line.split()]
        yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair
    
def mle(filename): # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda : defaultdict(int))
    ttfreq = defaultdict(lambda : defaultdict(int)) 
    tagfreq = defaultdict(int)    
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):
        last = startsym
        tagfreq[last] += 1
        for word, tag in list(zip(words, tags)) + [(stopsym, stopsym)]: #stop/stop is a word/tag pair?

            #if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1            
            ttfreq[last][tag] += 1
            dictionary[word].add(tag)
            tagfreq[tag] += 1
            last = tag            
    
    model = defaultdict(float)
    num_tags = len(tagfreq)
    for tag, freq in tagfreq.items(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].items():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2 # +1 smoothing
 
    return dictionary, model

def pad(strList):
    return [startsym] + strList + [stopsym]


def update(model, tags, mytags, template, i):
    if y!= z:
        model[y, x] += 1
        model[z, x] -= 1
    if y != z or tags[i-1] != mytag[i-1]:
        model[tags[i-1], y] += 1
        model[mytag[i-1], z] -=1



def perceptron(train, testF, AVG = False, epoch=10):
    dictionary, _ = mle(train)
    train_dat = list(readfile(train))
    updates = errTag = 0
    model = defaultdict(float)
    avgModel = defaultdict(float)
    c = 0
    for ep in range(epoch):
        updates = 0
        for words, tags in train_dat:
            mytag = decode(words, dictionary, model)
            if mytag != tags:
                updates += 1
                words   = pad(words)
                tags    = pad(tags)
                mytag   = pad(mytag)
                for i, (x, y, z) in enumerate(list(zip(words, tags, mytag))[1:], 1):
                    #put all of the inside of mytag!=tags into update func
#'''
                    if y != z:
                        model[y, x] += 1
                        model[z, x] -= 1
                    if y != z or tags[i-1] != mytag[i-1]:
                        model[tags[i-1], y] += 1
                        model[mytag[i-1], z] -=1
#'''
                if AVG:
                    avgModel += model
                    c+=1
        
        trainErr    = test(train, dictionary, model)
        devErr      = test(testF, dictionary, model)
        if AVG:
            avgTrain    = test(train, dictionary, avgModel)
            avgDev      = test(testF, dictionary, avgModel)
            averageString = " Average Train: {:0.2%} Average Dev: {:0.2%}".format(avgTrain, avgDev)

        weightSize = sum( x!=0 for x in model.values() )

        #print(trainErr, devErr)
        print("Epoch: {} Updates: {} |W| = {} TrainErr: {:0.2%} DevErr: {:0.2%}"\
                .format(ep, updates, weightSize, trainErr, devErr), end='' ) 

        print( ((AVG and averageString)  or ''))

def decode(words, dictionary, model):

    def backtrack(i, tag):
        if i == 0:
            return []
        return backtrack(i-1, back[i][tag]) + [tag]

    words = [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    best[0][startsym] = 1
    back = defaultdict(dict)


    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])
    for i, word in enumerate(words[1:], 1):
        for tag in dictionary[word]:
            for prev in best[i-1]:
                score = best[i-1][prev] + model[prev, tag] + model[tag, word]
                if score > best[i][tag]:
                    best[i][tag] = score
                    back[i][tag] = prev
        #print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, stopsym)[:-1]
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags

def test(filename, dictionary, model):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary ,model)
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot

def predict(filename, outfile,  dictionary, model):
    with open(outfile) as out:
        for words in [ x.split() for x in open(filename) ]:
            mytags = decode(words, dictionary, model)
            line = ''.join( word + '/' + tag for word, tag in zip(words, mytags) )
            print(line, file=out)

if __name__ == "__main__":
    trainfile, devfile = sys.argv[1:3]

    
    dictionary, model = mle(trainfile)

    perceptron(trainfile, devfile)

    #print("train_err {0:.2%}".format(test(trainfile, dictionary, model)))
    #print("dev_err {0:.2%}".format(test(devfile, dictionary, model)))

                                                                                                        
