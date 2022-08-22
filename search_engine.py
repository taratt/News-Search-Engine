from parsivar import Tokenizer, Normalizer, FindStems
from hazm import utils
from copy import deepcopy
import json
import re
import csv
import matplotlib.pylab as plt
import math
import numpy as np

def restore_from_json(path):
    file = open(path)
    dictionary = json.load(file)
    file.close()
    return dictionary

def get_stemmed(tokens, mode):
    stemmed_tokens = []
    stemmer = FindStems()
    for token in tokens:
        stemmed = stemmer.convert_to_stem(token[0]).split("&")
        if mode == 1:
            for root in stemmed:
                stemmed_tokens.append([root, token[1], token[2]])
        elif mode == 2:
            for root in stemmed:
                stemmed_tokens.append([root, token[1]])
    return stemmed_tokens

def delete_stopwords(tokens):
    stopwords_list = utils.stopwords_list()
    for token in deepcopy(tokens):
        if token[0] in stopwords_list:
            tokens.remove(token)

    return tokens

def position_tokenizing(tokens, docID, mode):
    position = 1
    positioned = []
    for token in tokens:
        if mode == 1:
            positioned.append([token, docID, position])
        elif mode == 2:
            positioned.append([token, position])
        position += 1
    return positioned

def store_info(tokens, path):
    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerows(tokens)

def preprocess(dataset):
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    termID_docID = []
    title_urls = {}
    for doc_id in dataset:
        if int(doc_id) % 100 == 0:
            print("processing " + doc_id)
        title_urls[doc_id] = [dataset[doc_id]["title"], dataset[doc_id]["url"]]
        content = dataset[doc_id]["content"]
        no_punc = re.sub(r'[^\w\s]', '', content)
        normalized_content = normalizer.normalize(no_punc)
        tokens = position_tokenizing(tokenizer.tokenize_words(normalized_content), doc_id, 1)
        # no_stopwords = [token for token in tokens if token[0] not in utils.stopwords_list()]
        no_stopwords = delete_stopwords(tokens)
        stemmed_list = get_stemmed(no_stopwords, 1)
        termID_docID.extend(stemmed_list)
    return termID_docID, title_urls


def preprocess_with_stopword(dataset):
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    termID_docID = []
    title_urls = {}
    for doc_id in dataset:
        if int(doc_id) % 100 == 0:
            print("processing " + doc_id)
        title_urls[doc_id] = [dataset[doc_id]["title"], dataset[doc_id]["url"]]
        content = dataset[doc_id]["content"]
        no_punc = re.sub(r'[^\w\s]', '', content)
        normalized_content = normalizer.normalize(no_punc)
        tokens = position_tokenizing(tokenizer.tokenize_words(normalized_content), doc_id, 1)
        stemmed_list = get_stemmed(tokens, 1)
        termID_docID.extend(stemmed_list)
    return termID_docID, title_urls

def preprocess_no_stem(dataset):
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    termID_docID = []
    title_urls = {}
    for doc_id in dataset:
        if int(doc_id) % 100 == 0:
            print("processing " + doc_id)
        title_urls[doc_id] = [dataset[doc_id]["title"], dataset[doc_id]["url"]]
        content = dataset[doc_id]["content"]
        no_punc = re.sub(r'[^\w\s]', '', content)
        normalized_content = normalizer.normalize(no_punc)
        tokens = position_tokenizing(tokenizer.tokenize_words(normalized_content), doc_id, 1)
        # no_stopwords = [token for token in tokens if token[0] not in utils.stopwords_list()]
        no_stopwords = delete_stopwords(tokens)
        termID_docID.extend(no_stopwords)
    return termID_docID, title_urls

def retrieve_tokens(path):
    tokens = []
    with open(path) as file:
        reader = csv.reader(file)
        for token in reader:
            tokens.append(token)

    return tokens

def store_json(json_dic, path):
    with open(path, 'w') as file:
        json.dump(json_dic, file, indent=4)

def build_inverted_index(tokens):
    inverted_index = {}
    for token in tokens:
        if token[0] in inverted_index:
            if token[1] in inverted_index[token[0]]:
                inverted_index[token[0]][token[1]].append(token[2])
                # inverted_index[token[0]][0] += 1
                # inverted_index[token[0]][token[1]][0]+=1
                # inverted_index[token[0]][token[1]][1].append(token[2])

            else:
                inverted_index[token[0]][token[1]] = [token[2]]
                # inverted_index[token[0]][0] += 1
                # inverted_index[token[0]][1][token[1]] = [1,[token[2]]]
        else:
            inverted_index[token[0]] = {token[1]: [token[2]]}
            # inverted_index[token[0]] = [1, {token[1]:[1, [token[2]]]}]
    return inverted_index

def query_parsing(query):
    phrasal = re.findall('"([^"]*)"', query)
    remainder = query
    for phrase in phrasal:
        remainder = remainder.replace("\"" + phrase + "\"", "")
    not_q = []
    remainder_copy = remainder
    if remainder != "":
        while "!" in deepcopy(remainder_copy):
            not_q.append(remainder_copy.split('!', maxsplit=1)[-1] \
                             .split(maxsplit=1)[0])
            if len(remainder_copy.split('!', maxsplit=1)[-1] \
                    .split(maxsplit=1)) !=1:
                remainder_copy = remainder_copy.split('!', maxsplit=1)[-1] \
                    .split(maxsplit=1)[1]
            else:
                remainder_copy=""
        for n_query in not_q:
            remainder = remainder.replace("! "+n_query,"")

    # print(remainder)
    # print(phrasal)
    # print(not_q)

    return remainder, phrasal, not_q

def query_preprocessing(query):
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    no_punc = re.sub(r'[^\w\s]', '', query)
    normalized_query = normalizer.normalize(no_punc)
    tokens = position_tokenizing(tokenizer.tokenize_words(normalized_query), [], 2)
    no_stopwords = delete_stopwords(tokens)
    stemmed_list = get_stemmed(no_stopwords, 2)
    return stemmed_list

def phrase_query(tokens, inverted_index):
    try:
        result = {}
        docs = inverted_index[tokens[0][0]].keys()
        #documenthaii k hameye kalamato daran
        for token in tokens:
            docs &= inverted_index[token[0]].keys()
        for doc in docs:
            for pos1 in inverted_index[tokens[0][0]][doc]:
                pos1 = int(pos1)
                if all(inverted_index[token[0]][doc].__contains__(str(pos1 + token[1] - 1)) for token in tokens):
                    if doc in result:
                        result[doc] += 1
                    else:
                        result[doc] = 1
        #documentha ba tedad tekrare phrase
        return {k: v for k, v in sorted(result.items(), reverse=True, key=lambda item: item[1])}
    except:
        return None

def normal_retrieval(remainders, inverted_index):
    result = []
    for remainder in remainders:
        try:
            remainder = remainder[0]
            retrieved = inverted_index[remainder]
            remainder_retrieved = {}
            for key in retrieved:
                remainder_retrieved[key] = len(retrieved[key])
            result.append(remainder_retrieved)
        except:
             continue
    #result mishe ye list az dictionaryha k har dictionary keysh mishe shomareye document va valuesh mishe frequency too oon document
    return result

def not_query(not_q, results, inverted_index, titles_urls):
    # return (np.setdiff1d(result, list(inverted_index[not_q].keys())))
    # try:
        if results == []:
            #return set(range(0, len(titles_urls))) - set(inverted_index[not_q].keys())
            return []
        else:
            not_q_list = inverted_index[not_q].keys()
            for result in results:
                for doc in deepcopy(result):
                    if doc in not_q_list:
                        result.pop(doc,None)
    # except:
    #     pass
        return results

def ranking(results):
    keys = []
   # print(results)
    for result in results:
        keys.extend(list(result.keys()))
    frequencies = {}
    for key in keys:
        if key in frequencies:
            frequencies[key]+=1
        else:
            frequencies[key] = 1
    repeats = {}
    for key in keys:
        for result in results:
            if key in result:
                if key in repeats:
                    repeats[key] += result[key]
                else:
                    repeats[key] = result[key]
    ranked =list({k: v for k, v in sorted(frequencies.items(), reverse=True, key=lambda item: item[1])}.keys())

    ranked = sorted(ranked,reverse=True, key= lambda x: (frequencies[x],repeats[x]))
    return ranked


def query_processing(remainder, phrasal, not_q, inverted_index, titles_urls):
    phrasal_retrievals = []
    if not phrasal is None:
        for phrase in phrasal:
            #print(query_preprocessing(phrase))
            phrasal_retrievals.append(phrase_query(query_preprocessing(phrase), inverted_index))
    #phrasal_retrievals ye liste k toosh dictionaryhaii k harkodoom az phraseha ro daran az doc va tedade tekrar
    retrievals = []
    if remainder != "":
        retrievals = normal_retrieval(query_preprocessing(remainder), inverted_index)

    retrievals.extend(phrasal_retrievals)
    for q_not in not_q:
        retrievals = not_query(q_not, retrievals, inverted_index, titles_urls)
    results = result_format(ranking(retrievals),titles_urls)
    for result in results:
             print(result)


def result_format(results, titles_urls):
    final_format = []
    for doc in results:
            final_format.append(doc + ":\n    title: " + titles_urls[doc][0] + "\n    URL: " + titles_urls[doc][1])
    return final_format

def count_frequencies(inverted_index):
    frequencies = {}
    for word in inverted_index:
        frequency = 0
        for doc in inverted_index[word]:
            frequency += len(inverted_index[word][doc])
        frequencies[word] = frequency
    return {k: v for k, v in sorted(frequencies.items(), reverse=True, key=lambda item: item[1])}
def zipf(inverted_index):
    zipf_no_stop = count_frequencies(inverted_index)

    # termID_docID, urls_titles =preprocess_with_stopword(restore_from_json('IR_data_news_12k.json'))
    # store_info(termID_docID,"termID_docID_with_stopword.csv")
    # store_json(build_inverted_index(retrieve_tokens('termID_docID_with_stopword.csv')), "inverted_index_with_stop.json")
    inverted_with_stop = restore_from_json("inverted_index_with_stop.json")
    zipf_with_stop = count_frequencies(inverted_with_stop)
    print("done")

    x_stop = [math.log10(y) for y in list(range(1,len(zipf_with_stop)+1))]
    y_stop = [math.log10(y) for y in list(zipf_with_stop.values())]
    y_ideal = [math.log10(list(zipf_with_stop.values())[0]) - x for x in x_stop]
    x_no_stop = [math.log10(y) for y in list(range(1, len(zipf_no_stop) + 1))]
    y_no_stop = [math.log10(y) for y in list(zipf_no_stop.values())]
    y_ideal_no = [math.log10(list(zipf_no_stop.values())[0]) - x for x in x_no_stop]
    plt.subplot(1, 2, 1)
    plt.plot(x_stop, y_stop)
    plt.plot(x_stop, y_ideal)
    plt.xlabel("log10 rank")
    plt.ylabel("log10 cf")
    plt.title("Including Stopwords")

    plt.subplot(1, 2, 2)
    plt.plot(x_no_stop, y_no_stop)
    plt.plot(x_no_stop, y_ideal_no)
    plt.xlabel("log10 rank")
    plt.ylabel("log10 cf")
    plt.title("No Stopwords")

    plt.show()

def calculate_dic_len(inverted_index, heaps_dic):
    for instance in heaps_dic:
        for word in inverted_index:
            for doc in inverted_index[word]:
                if int(doc)<= instance:
                    heaps_dic[instance][0]+=1
                    break
    return heaps_dic

def calculate_tokens_len(tokens, heaps_dic):
    for token in tokens:
        for doc_num in heaps_dic:
            if int(token[1])<= doc_num:
                heaps_dic[doc_num][1]+=1
    return heaps_dic

def heap(inverted_index, tokens):
    all_docs = len(restore_from_json('IR_data_news_12k.json'))

    # termID_docID, urls_titles =preprocess_no_stem(restore_from_json('IR_data_news_12k.json'))
    # store_info(termID_docID,"termID_docID_no_stem.csv")
    # store_json(build_inverted_index(retrieve_tokens('termID_docID_no_stem.csv')), "inverted_index_no_stem.json")
    termID_docID = retrieve_tokens("termID_docID_no_stem.csv")
    inverted_no_stem= restore_from_json("inverted_index_no_stem.json")
    tokens_num = len(tokens)
    tokens_no_stem_num = len(termID_docID)
    dic_num = len(inverted_index)
    dic_no_stem_num = len(inverted_no_stem)

    #print("done")
    heaps_dic= {500:[0,0], 1000:[0,0], 1500:[0,0], 2000:[0,0]}
    heaps_dic1 = {500: [0, 0], 1000: [0, 0], 1500: [0, 0], 2000: [0, 0]}
    stemmed = calculate_dic_len(inverted_index,heaps_dic)
    not_stemmed = calculate_dic_len(inverted_no_stem, heaps_dic1)

    stemmed = calculate_tokens_len(tokens,stemmed)
    not_stemmed = calculate_tokens_len(termID_docID, not_stemmed)
    print(stemmed)
    print(not_stemmed)
    x = np.array([math.log10(t[1]) for t in list(stemmed.values())])
    y = [math.log10(t[0]) for t in list(stemmed.values())]
    m, b = np.polyfit(x, y, 1)
    x_no_stem = np.array([math.log10(t[1]) for t in list(not_stemmed.values())])
    y_no_stem = [math.log10(t[0]) for t in list(not_stemmed.values())]
    mn, bn = np.polyfit(x_no_stem, y_no_stem, 1)
    plt.subplot(1, 2, 1)
    plt.scatter(x, y,color = 'pink')
    plt.plot(x, m*x + b)
    plt.xlabel("log10 T")
    plt.ylabel("log10 M")
    plt.title("With stemming")

    plt.subplot(1, 2, 2)
    plt.scatter(x_no_stem, y_no_stem,color = 'pink')
    plt.plot(x_no_stem, mn*x_no_stem+bn)
    plt.xlabel("log10 T")
    plt.ylabel("log10 M")
    plt.title("No Stemming")

    plt.show()
    print("Size of the Vocabulary when stemming is done: "+ str(dic_num))
    print("k= "+ str(10**b)+ "  b=" + str(round(m,3)))
    print("Heap's law prediction: "+ str(round(((10**b)*(tokens_num**m)),0)))
    print("Size of the Vocabulary when stemming is skipped: " + str(dic_no_stem_num))
    print("k= " + str(10 ** bn) + "  b=" + str(round(mn, 3)))
    print("Heap's law prediction: "+ str(round(((10**bn)*(tokens_no_stem_num**mn)),0)))

# termID_docID, urls_titles =preprocess(restore_from_json('IR_data_news_12k.json'))
# store_info(termID_docID,"termID_docID.csv")
# store_json(urls_titles,"urls_titles.json")
# store_json(build_inverted_index(retrieve_tokens('termID_docID.csv')), "inverted_index.json")
inverted_index = restore_from_json("inverted_index.json")
tokens = retrieve_tokens("termID_docID.csv")
#zipf(inverted_index)
#heap(inverted_index,tokens)
titles_urls = restore_from_json("urls_titles.json")
#print("built")
t = input("query?")
r, p, n = query_parsing(t)
query_processing(r, p,n, inverted_index, titles_urls)
