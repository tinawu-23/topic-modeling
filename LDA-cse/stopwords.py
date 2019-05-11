import json

stopWordFiles = ['rank1.json', 'rank2.json', 'rank3.json', 'rank4.json', 'rank5.json', 'rank6.json', 'rank7.json', 'rank8.json']

rankedStop = {}

for rankfile in stopWordFiles:
  with open(rankfile) as f:
    worddict = json.load(f)
    for word in worddict:
      rankedStop[word[0]] = rankedStop.get(word[0], 0) + word[1]

rankedStop = sorted(rankedStop.items(), key=lambda x: x[1], reverse=True)
for word in rankedStop[:15]:
  print(word[0])