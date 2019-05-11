def cleantext(text,IF_STOPWORD):
    if text == '': return ''
    from nltk.tokenize import sent_tokenize,word_tokenize
    from gensim.parsing.preprocessing import STOPWORDS    
    ret = ''
    for sentences in sent_tokenize(text):
        for word in word_tokenize(sentences):
            word = word.lower()
            if word.isalpha():
                if IF_STOPWORD:
                    if len(word) > 2 and not word in STOPWORDS:
                        ret += ' '+word
                else:
                    ret += ' '+word
    return ret[1:]

def preprocessing(fileoutput,fileinput,IF_STOPWORD=True):
    from tqdm import tqdm
    fw = open(fileoutput,'w')
    fr = open(fileinput, 'r', encoding="utf8", errors='ignore')
    fr.readline()
    for line in tqdm(fr):
        arr = line.strip('\r\n').split('\t')
        title = arr[4]
        abstract = arr[5]
        if abstract == "none":
            abstract = ""
        text = title+' '+abstract
        if len(text.split()) < 10: continue
        text = cleantext(text,IF_STOPWORD)
        if text == '': continue
        fw.write(text+'\n')
    fr.close()
    fw.close()

def preprocesstop(fileoutput,fileinput):
    from tqdm import tqdm
    from gensim.parsing.preprocessing import remove_stopwords
    fw = open(fileoutput,'w')
    fr = open(fileinput, 'r', encoding="utf8", errors='ignore')
    for line in tqdm(fr):
        text = remove_stopwords(line.strip())
        fw.write(text+'\n')
    fr.close()
    fw.close()

def runlda(filetopicwords,fileinput,NUMTOPICS=30,NUMPASSES=10,NUMITERATIONS=10):
    print('runlda...')
    from gensim.corpora import Dictionary
    from gensim.models.ldamodel import LdaModel
    import numpy as np
    docs,word2freqtopics = [],{}
    fr = open(fileinput,'r')
    for line in fr:
        words = line.strip('\r\n').split(' ')
        docs.append(words)
        for word in words:
            if not word in word2freqtopics:
                word2freqtopics[word] = [0,[0. for i in range(NUMTOPICS)]]
            word2freqtopics[word][0] += 1
    fr.close()
    V = len(word2freqtopics)
    dct = Dictionary(docs)
    model = LdaModel(corpus=[dct.doc2bow(doc) for doc in docs],id2word=dct, \
            num_topics=NUMTOPICS,passes=NUMPASSES,iterations=NUMITERATIONS) 
    fw = open(filetopicwords,'w')
    for topicid in range(NUMTOPICS):
        s = 'topic '+str(topicid)
        wordscores = []
        for (wordid,score) in model.get_topic_terms(topicid,topn=V):
            if score < 1e-6: break
            wordscores.append([dct[wordid],score])
        scoresum = sum([x[1] for x in wordscores])
        for [word,score] in wordscores:
            s += ','+word+':'+str(np.round(score/scoresum,6))
            word2freqtopics[word][1][topicid] = score
        fw.write(s+'\n')
    fw.close()
    '''
    fw = open(filewordtopics,'w')
    for [word,[freq,scores]] in sorted(word2freqtopics.items(),key=lambda x:-x[1][0]):
        s = word+','+str(freq)
        scoresum = sum(scores)
        if scoresum == 0.: continue
        for score in scores:
            s += ','+str(np.round(score/scoresum,6))
        fw.write(s+'\n')
    fw.close()
    fw = open(filedoctopics,'w')
    for doc in docs:
        s = ''        
        topicid2score = {}
        scoresum = 0.
        for (topicid,score) in model.get_document_topics([dct.doc2bow(doc)])[0]:
            topicid2score[topicid] = score
            scoresum += score
        for [topicid,score] in sorted(topicid2score.items(),key=lambda x:x[0]):
            s += ','+str(topicid)+':'+str(np.round(score/scoresum,6))
        fw.write(s[1:]+'\n')
    fw.close()
    '''

def flagstopword(fileoutput,fileinput,filecorpuswide):
    from gensim.parsing.preprocessing import STOPWORDS
    corpuswide = set()
    fr = open(filecorpuswide,'r')
    for line in fr:
        corpuswide.add(line.strip('\r\n'))
    fr.close()
    fw = open(fileoutput,'w')
    fr = open(fileinput,'r')
    for line in fr:
        arr = line.strip('\r\n').split(',')
        n = len(arr)
        s = arr[0]
        for i in range(1,n):
            word,score = arr[i].split(':')
            if word in STOPWORDS:
                word = '['+word+']'
            elif word in corpuswide:
                word = '('+word+')'
            s += ','+word+':'+score
        fw.write(s+'\n')
    fr.close()
    fw.close()

def preprocessing_corpuswide(fileoutput,fileinput,filecorpuswide):
    corpuswide = set()
    fr = open(filecorpuswide,'r')
    for line in fr:
        corpuswide.add(line.strip('\r\n'))
    fr.close()
    fw = open(fileoutput,'w')
    fr = open(fileinput,'r')
    for line in fr:
        s = ''
        for word in line.strip('\r\n').split(' '):
            if word in corpuswide: continue
            s += ' '+word
        fw.write(s[1:]+'\n')
    fr.close()
    fw.close()

if __name__ == '__main__':

    ''' run LDA while removing regular stopwords '''
    # preprocesstop('LDA-all/nsf-awards-stop.txt','LDA-all/nsf-awards.txt')

    # runlda('LDA-all/lda-award-5.csv', 'LDA-all/nsf-awards-stop.txt', 30, 20, 20)

    ''' run LDA while NOT removing regular stopwords '''
    # preprocessing('LDA-all/nsf-awards.txt','nsf-awards-org.txt',False)
    # runlda('LDA/topicwords-stop.csv','LDA/wordtopics-stop.csv', \
    #         'LDA/doctopics-stop.csv','LDA/documents-stop.txt',30,100,100)


    ''' flag/format stopwords in the csv files '''
    # flagstopword('LDA/flag-topicwords.csv','LDA/topicwords.csv', \
    #         'LDA/corpuswide-stop.txt')
    # flagstopword('LDA/flag-topicwords-stop.csv','LDA/topicwords-stop.csv', \
    #         'LDA/corpuswide-stop.txt')


    ''' run LDA on original data but with corpus wide stopwords removed '''
    preprocessing_corpuswide('LDA-all/nsf-awards-allcpstop.txt',
                             'LDA-all/nsf-awards-stop.txt', 'corpuswide-stop.txt')
    # runlda('LDA/topicwords-corpuswide.csv','LDA/wordtopics-corpuswide.csv', \
    #         'LDA/doctopics-corpuswide.csv','LDA/documents-corpuswide.txt',30,100,100)


