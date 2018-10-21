"""Docstring"""
import os
import sys
import csv
import pickle
from scipy import spatial
import numpy as np
import rouge
import time
from tree import *
from pulp import *

Tsim = .6

def evaluate(summerie, referenceDir):
    rog = Rouge()
    dic = 0
    for summerie in os.listdir(referenceDir):
        summ = referenceDir + '/' + summerie
        dic = max(rog.get_scores(summerie, summ)[0]["rouge-1"]['r'], dic)
    # return dic["rouge-1"], dic["rogue-2"]
    return dic

def compare(tree1, tree2):
    vec1 = []
    for word in tree1.getTerminals():
        vec1.append(word.feature[0])
        vec1.append(word.feature[1])

    vec2 = []
    for word in tree2.getTerminals():
        vec2.append(word.feature[0])
        vec2.append(word.feature[1])

    diff = len(vec1) - len(vec2)

    if diff < 0:
        for _ in range(0, -diff):
            vec1.append(0)
    else:
        for _ in range(0, diff):
            vec2.append(0)

    return 1 - spatial.distance.cosine(vec1, vec2)


def greedySelection(pickleF, maxCount):
    treesList = pickle.load( open(pickleF, 'rb') )
    trees = []
    for treeList in treesList:
        for tree in treeList:
            trees.append(tree)
            
    treesSorted = sorted(trees, key=lambda t: t.root.salience)

    wordCount = 0
    summerie = []
    while wordCount < maxCount:
        sentence = treesSorted.pop()
        similarity = 0
        for s in summerie:
            similarity = max(similarity, compare(s, sentence))
        if similarity < Tsim:
            summerie.append(sentence)
            wordCount += len(sentence.wordlist)

    text = ""
    wL = []
    for tree in summerie:
        for word in tree.wordlist:
            text += word + ' '
            wL += word
    return text, wL

def ILP(pickleF, maxCount):
    help_dic = {}

    treesList = pickle.load( open(pickleF, 'rb') )
    trees = []
    for treeList in treesList:
        for tree in treeList:
            trees.append(tree)


    wordList = []
    for tree in trees:
        for word in tree.getTerminals():
            #print(word.label)
            wordList.append(word.label)
            if word.parent.salience == -1:
                help_dic[word.label] = 0
            else:
                help_dic[word.label] = word.parent.salience
            #print(word.salience)

    wordList = list(set(wordList))
    Ns = len(trees)
    Nw = len(wordList)
    Occ = np.zeros((Ns, Nw))
    l = [len(tree.wordlist) for tree in trees]

    Ss = [t.root.salience for t in trees]
    Sw = [help_dic[word] for word in wordList]

    for idx, tree in enumerate(trees):
        for word in tree.wordlist:
            Occ[idx][wordList.index(word)] = 1

    p = LpProblem("Selection problem", LpMaximize)

    x = [LpVariable("x"+str(i), 0, 1, cat='Integer') for i in range(Ns)]
    c = [LpVariable("c"+str(i), 0, 1, cat='Integer') for i in range(Nw)]

    p += LpAffineExpression([(a, b) for a, b in zip(x, l)]) <= maxCount

    for i in range(Ns):
        for j in range(Nw):
            p += x[i]*Occ[i][j] <= c[j]
    
    for j in range(Ns):
        p += LpAffineExpression([(a, b) for a, b in zip(x, Occ[:][j])]) >= c[j]

    Z = .5 * lpSum([x[i]*l[i]*Ss[i] for i in range(Ns)]) + .5 * lpSum(
            [c[j]*Sw[j] for j in range(Nw)])

    p += Z
    p.solve()

    summerie = []
    for i in range(Ns):
        if value(x[i]):
            summerie.append(trees[i])

    text = ""
    wL = []
    for tree in summerie:
        for word in tree.wordlist:
            text += word + ' '
            wL += word
    return text, wL

if __name__ == '__main__':
    word_count = 100
    pickle_dir = sys.argv[1]
    summeries_dir = sys.argv[2]
    text_dump = "text_dump"
    score_file = "text_dump"
    for idx, pickle_file in enumerate(sorted(os.listdir(pickle_dir))):
        name = pickle_file.split('.')[0]
        pickle_path = pickle_dir + '/' + pickle_file
        summeries_path = summeries_dir + '/' + name
        print("Working on claster: {} ({}/{})".format(name, idx, len(os.listdir(pickle_dir))))
        text, word_count = greedySelection(pickle_path, word_count)
        print("Evluating . . .")
        score = evaluate(text, summeries_path)
        print("Score for {} is: {}".format(name, score))
        with open(score_file, 'a') as f:
            dumper = csv.writer(f)
            dumper.writerow([name, score])
        with open(text_dump, 'a') as f:
            f.writelines('=' * 10 + name + '=' * 10)
            f.writelines(text)

