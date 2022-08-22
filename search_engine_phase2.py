from parsivar import Tokenizer, Normalizer, FindStems
from hazm import utils
import re
from copy import deepcopy
import json
import csv
import math
import numpy as np

def restore_from_json(path):
    file = open(path)
    dictionary = json.load(file)
    file.close()
    return dictionary

def store_info(tokens, path):
    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerows(tokens)


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
            if token[1] in inverted_index[token[0]][1]:
                inverted_index[token[0]][1][token[1]][0]+=1
                inverted_index[token[0]][1][token[1]][1].append(token[2])

            else:

                inverted_index[token[0]][0] += 1
                inverted_index[token[0]][1][token[1]] = [1,[token[2]]]
        else:
            inverted_index[token[0]] = [1, {token[1]:[1, [token[2]]]}]
    return inverted_index

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

def query_preprocessing(query):
    tokenizer = Tokenizer()
    normalizer = Normalizer()
    no_punc = re.sub(r'[^\w\s]', '', query)
    normalized_query = normalizer.normalize(no_punc)
    tokens = position_tokenizing(tokenizer.tokenize_words(normalized_query), [], 2)
    no_stopwords = delete_stopwords(tokens)
    stemmed_list = get_stemmed(no_stopwords, 2)
    res = []
    for term in stemmed_list:
        res.append(term[0])
    return res
def get_lengths (inverted_index, number_of_docs,mode):
    lengths = np.zeros(number_of_docs)
    for term in inverted_index:
        for doc in inverted_index[term][1]:
            if mode == 0:
                lengths[int(doc)]+= ((1 + math.log10(inverted_index[term][1][doc][0]))*math.log10(number_of_docs/inverted_index[term][0]))**2
            else:
                lengths[int(doc)] += ((1 + math.log10(inverted_index[term][1][doc])) * math.log10(
                    number_of_docs / inverted_index[term][0])) ** 2
    lengths = lengths** 0.5
    return lengths

def search(query, inverted_index, number_of_docs, lengths,mode):
    scores = np.zeros(number_of_docs)
    preprocessed_query = query_preprocessing(query)
    query_dic = {}
    for term in preprocessed_query:
        if term in query_dic:
            query_dic[term]+=1
        else:
            query_dic[term] = 1
    for term in query_dic:
        try:
            postings = inverted_index[term]
            weight_term_query = (1+ math.log10(query_dic[term]))
            for doc in postings[1]:
                if mode == 0:
                    weight_term_doc = (1 + math.log10(postings[1][doc][0]))* math.log10(number_of_docs/postings[0])
                else:
                    weight_term_doc = (1 + math.log10(postings[1][doc])) * math.log10(number_of_docs / postings[0])
                scores[int(doc)] += weight_term_query * weight_term_doc
        except:
            pass
    scores_dic = {}
    for i in range(number_of_docs):
        scores_dic[str(i)] = scores[i] / lengths[i]

    results = {k: v for k, v in sorted(scores_dic.items(), key=lambda item: item[1], reverse=True)}
    k_tops_scores = list(results.items())[:10]
    k_tops = []
    for k in k_tops_scores:
        k_tops.append(k[0])
    return k_tops

def build_champion_lists(inverted_index):
    champion_lists = {}
    for term in inverted_index:
        postings = inverted_index[term][1]
        sorted_docs = {k: v for k, v in sorted(postings.items(), key=lambda item: item[1][0], reverse=True)}
        list_champ = list(sorted_docs.items())[:500]
        champion_list = {}
        for champ in list_champ:
            champion_list[champ[0]] = champ[1][0]
        champion_lists[term] = [inverted_index[term][0], champion_list]
    return champion_lists

def result_format(results, titles_urls):
    final_format = []
    for doc in results:
            final_format.append(doc + ":\n    title: " + titles_urls[doc][0] + "\n    URL: " + titles_urls[doc][1])
    return final_format



if __name__ == '__main__':
    #store_json(build_inverted_index(retrieve_tokens('termID_docID.csv')), "inverted_index_tf_idf.json")
    inverted_index = restore_from_json("inverted_index_tf_idf.json")
    #store_json(build_champion_lists(inverted_index), "champion_lists.json")
    champion_lists = restore_from_json("champion_lists.json")
    titles_urls = restore_from_json("urls_titles.json")
    number_of_docs = len(list(titles_urls.keys()))
    lengths = get_lengths(inverted_index,number_of_docs,0)
    champ_lengths = get_lengths(champion_lists,number_of_docs,1)
    print("built")
    query = input("query?")
    while query!= "t":
        print("results:")
        results = search(query, inverted_index, number_of_docs, lengths,0)
        final_format = result_format(results, titles_urls)
        for result in final_format:
            print(result)
        print("champion results:")
        results = search(query, champion_lists, number_of_docs, champ_lengths,1)
        final_format = result_format(results, titles_urls)
        for result in final_format:
            print(result)
        query = input("query?")

