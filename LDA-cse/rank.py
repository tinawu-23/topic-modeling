import json

def rank1(outputfile, inputfile):
    wordscores = {}
    fr = open(inputfile, 'r')
    for line in fr:
        i = 0
        arr = line.strip('\r\n').split(',')
        for item in arr:
            if item.startswith('topic '): continue
            i += 1
            word = item.split(':')[0]
            prob = item.split(':')[1]
            wordscores[word] = wordscores.get(word, 0) + float(prob) * float(1/i) * 1000

    wordscores = sorted(wordscores.items(), key=lambda x: x[1], reverse=True)
    with open(outputfile, 'w') as fp:
        json.dump(wordscores, fp, indent=4)

if __name__ == '__main__':
    rank1('rank5.json', 'topicwords5.csv')