#!/usr/bin/env python


#from collections import defaultdict
import matplotlib.pyplot as plt
from modDict import myclass as defaultdict
import sys, time
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



def templateToString(template, tags, words, idx):#use this to generate string for a given template
    tupList = []
    strList = []
    for t in template.split('_'):
        tupList.append((t[0], int(t[1:])))
    tupList.sort()
    for elem in tupList:
        curr = idx + elem[1]
        if curr< 0 or curr >=len(words):
            strList.append("OOR")#out of range
            continue
        strList.append( str(elem) + (tags[curr] if elem[0] == "t" else words[curr]) )

    return ''.join(strList)


def update(model, tags, mytag, words, template):
    
    for i, (x, y, z) in enumerate(list(zip(words, tags, mytag))[1:], 1):
        for guide in template:
            active = [ int(tag[1:]) for tag in filter(lambda x: x[0] =='t', guide.split('_'))]
            for idx in active:
                val = i+idx
                if val < 0 or val >= len(words):
                    continue
                if tags[val] != mytag[val]:
                    bad = templateToString(guide, mytag, words, i)
                    good = templateToString(guide, tags, words, i)
                    model[bad]  -= 1
                    model[good] += 1
                    break
'''
        if y!= z:
            model[y, x] += 1
            model[z, x] -= 1
        if y != z or tags[i-1] != mytag[i-1]:
            model[tags[i-1], y] += 1
            model[mytag[i-1], z] -=1

     for template
        for tag in that template
            if tag was predicted wrong and tag exists
                
                model[ bad] -=1 
                model[good] +=1
'''

def perceptron(train, testF, template, AVG = True, epoch=10):
    dictionary, _ = mle(train)
    train_dat = list(readfile(train))
    updates = errTag = 0
    model = defaultdict(float)
    avgModel = defaultdict(float)
    accSpread = [ [] for _ in range(4) ]
    c = 0
    for ep in range(epoch):
        updates = 0
        for words, tags in train_dat:
            mytag = decode(words, dictionary, model, template)
            if mytag != tags:
                updates += 1
                words   = pad(words)
                tags    = pad(tags)
                mytag   = pad(mytag)
                #for i, (x, y, z) in enumerate(list(zip(words, tags, mytag))[1:], 1):
                update(model, tags, mytag, words, template)
                '''
                if y != z:
                    model[y, x] += 1
                    model[z, x] -= 1
                if y != z or tags[i-1] != mytag[i-1]:
                    model[tags[i-1], y] += 1
                    model[mytag[i-1], z] -=1
                '''

                if AVG:
                    avgModel += model
                    c+=1
        
        trainErr    = test(train, dictionary, model, template)
        devErr      = test(testF, dictionary, model, template)
        accSpread[0].append(trainErr)
        accSpread[1].append(devErr)
        if AVG:
            avgTrain    = test(train, dictionary, avgModel, template)
            avgDev      = test(testF, dictionary, avgModel, template)
            averageString = " Average Train: {:0.2%} Average Dev: {:0.2%}".format(avgTrain, avgDev)
            accSpread[2].append(avgTrain)
            accSpread[3].append(avgDev)

        weightSize = sum( x!=0 for x in model.values() )

        #print(trainErr, devErr)
        print("Epoch: {} Updates: {} |W| = {} TrainErr: {:0.2%} DevErr: {:0.2%}"\
                .format(ep, updates, weightSize, trainErr, devErr), end='' ) 

        print( ((AVG and averageString)  or ''))

    return model, avgModel, accSpread




def decode(words, dictionary, model, template):
    def backtrack(i, tag):
        if i == 0:
            return []
        return backtrack(i-1, back[i][tag]) + [tag]

    def templateScore(model, template, words, prev, curr, i ):
        
        tags = [startsym] + backtrack(i-1, prev) + [tag]
        #print(tags)
        cumulative = best[i-1][prev] 
        for guide in template:
            cumulative += model[templateToString(guide, tags, words, i)]
        return cumulative

    words = [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    best[0][startsym] = 1
    back = defaultdict(dict)


    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])
    for i, word in enumerate(words[1:], 1):
        for tag in dictionary[word]:
            for prev in best[i-1]:
                #score = best[i-1][prev] + model[prev, tag] + model[tag, word]
                score = templateScore(model, template, words, prev, tag, i)
                if score > best[i][tag]:
                    best[i][tag] = score
                    back[i][tag] = prev
        #print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, stopsym)[:-1]
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags

def test(filename, dictionary, model, template):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary ,model, template)
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot

def predict(filename, outfile,  dictionary, model):
    with open(outfile) as out:
        for words in [ x.split() for x in open(filename) ]:
            mytags = decode(words, dictionary, model)
            line = ''.join( word + '/' + tag for word, tag in zip(words, mytags) )
            print(line, file=out)

def errPlot(spread):
    plt.figure(figsize=(12,6))
    plotting = [ "r--", "b--", "rs", "bs"]
    labels = ["Basic Train","Basic Dev","Averaged Train","Averaged Dev"]
    hands = []
    dim = list(range(len(spread[0])))
    
    for i, dat in enumerate(spread):
        handle, = plt.plot( dim, dat, plotting[i], label=labels[i])
        hands.append(handle)
    #handle, = plt.plot( dim, spread[0], plotting[3], label=labels[0])
    #hands.append(handle)
    plt.legend(handles=hands)
    plt.xlabel("Epoch")
    plt.xticks(dim)
    plt.ylabel("Error Rate")
    plt.title("Perceptron Error vs. Epoch")
    plt.savefig("Avg_and_basic_POS.png")
    plt.show()
    return

if __name__ == "__main__":
    trainfile, devfile = sys.argv[1:3]

    
    dictionary, model = mle(trainfile)

    template = ["t0_t-1", "t0_w0"]

    currTime = time.time()
    model, avgModel, acc = perceptron(trainfile, devfile, template, AVG = True)
    
    #timetaken = time.time()-currTime
    #print("Averaged run: {}".format(timetaken))
    #model, avgModel, _ = perceptron(trainfile, devfile, template, AVG = False)
    #print("Not averaged run: {}".format( time.time() - currTime -timetaken))

    #errPlot(acc)


                                                                                                        
