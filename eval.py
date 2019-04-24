import json
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

total = 307

def eval(threshold):
    with open('LDA/rank1.json') as f:
        wordsdict = json.load(f)

    true = [1] * threshold
    pred = []
    cutwordlist = wordsdict[:total]
    predictedstop=[item[0] for item in cutwordlist]
    # print(predictedstop)

    corpuswide = []
    fr = open('corpuswide-stop.txt', 'r')
    for line in fr:
        corpuswide.append(line.strip('\r\n'))
    fr.close()

    # print(corpuswide)

    i = 0
    for word in corpuswide:
        if i > (threshold-1):
            break
        if word in predictedstop:
            pred.append(1)
            i += 1
        else:
            pred.append(0)
            i += 1

    # print(pred)

    # print(true)
    # print(pred)
    # print(f1_score(true, pred[:threshold], average='macro'))
    return f1_score(true, pred[:threshold], average='macro')


if __name__ == '__main__':
    threshold = []
    f1scores = []
    for i in range(10,40):
        threshold.append(i)
        f1scores.append(eval(i))

    plt.scatter(threshold, f1scores)
    plt.title('F1 scores based on threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.show()
