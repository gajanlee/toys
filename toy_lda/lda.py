#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lee.py
@Time    :   2019/09/07 21:20:53
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   Simple LDA implementation
'''

import numpy as np

from bisect import bisect_left
from functools import reduce
from operator import add
from prettytable import PrettyTable
from random import randint

# formula 60
# p(w_m | alpha, beta)
def log_likelihood():
    pass


class Vocabulary:
    
    def __init__(self):
        self.UNKNOWN = 'UNK'
        self.token_to_id = {self.UNKNOWN: 0}
        self.id_to_token = [self.UNKNOWN]

    def update_docs(self, docs):
        list(map(self.update_single, docs))
        return self
    
    def update_single(self, doc):
        for token in doc:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)
        return self
    
    def convert(self, docs):
        return np.array(list(map(self.convert_single, docs)), dtype=np.int32)
    
    def convert_single(self, doc):
        X = np.zeros(len(self.id_to_token), dtype=np.int32)
        for token in filter(lambda token: token in self.token_to_id, doc):
            X[self.get_id_by_token(token)] += 1
        return X
        
    def get_token_by_id(self, token_id):
        token = self.UNKNOWN if token_id >= len(self.id_to_token) or token_id < 0 else self.id_to_token[token_id]
        return token

    def get_id_by_token(self, token):
        token = self.UNKNOWN if token not in self.token_to_id else token
        return self.token_to_id[token]

class LDA:
    
    def __init__(self, n_topics=20, n_iters=5, alpha=0.1, beta=0.01, random_seed=1234):
        """Init the model with hyper-parameters.

        Parameters
        ----------
        alpha: double
            Lower alpha will lead to sparse document-topic distribution,
            means that the model prefers to characterise documents by few topics.
        beta: double
            Lower beta will lead to sparse topic-term distribution,
            means that the model prefers to assign few terms to each topic.
        """
        self.n_topics = n_topics
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.random_seed = random_seed

    def fit_documents(self, documents):
        """Convert the documents to words id and construct vocabulary,
        fit the model with data.

        Parameters
        ----------
        documents: 2-dimension list, documents with words.
        """
        self.vocab = Vocabulary().update_docs(documents)
        X = self.vocab.convert(documents)
        self.fit(X)
        return self
    
    def fit(self, X):
        self.initialize(X)

        for i in range(1, self.n_iters+1):
            if i % 1 == 0: print(f"iter: {i}")
            self.gibbs_sample()
        self.construct_distribution()

        return self

    def inference(self, docs, max_iter=20, tolerance=0.01, topk=10):
        """
        Wallach, Hanna M., Iain Murray, Ruslan Salakhutdinov, and David Mimno. 2009.
        “Evaluation Methods for Topic Models.” In Proceedings of the 26th Annual
        International Conference on Machine Learning, 1105–1112. ICML ’09. New York,
        NY, USA: ACM. https://doi.org/10.1145/1553374.1553515.

        "iterated pseudo-counts" in Section 4.1.

        Also, we could use gibbs sample to caculate once, the model as prior.
        """
        topic_distributions = np.empty((len(docs), self.n_topics))
        docs = self.vocab.convert(docs)
        for i, doc in enumerate(docs):
            terms = np.nonzero(doc)[0]
            doc = np.repeat(terms, doc[terms])

            topic_distributions[i] = self._inference_single(doc, max_iter, tolerance)

        # sort topic to topK
        topic_table = PrettyTable(["doc_id"] + [f"top{k+1}" for k in range(topk)])

        for i, topic_distribution in enumerate(topic_distributions):
            topic_table.add_row([f"doc{i}"] + 
                    [f"t{topic_ind}/{topic_distribution[topic_ind]:.2}"
                    for topic_ind in np.argsort(-topic_distribution)[:topk]])
        print(topic_table)

        return topic_distributions

    def display_distribution(self, topk=10):
        """Display topk terms in every topic.
        """
        topic_term_table = PrettyTable(["topic_id"] + [f"top{k+1}" for k in range(topk)])

        for i, dist in enumerate(self.topic_term_distribution):
            topic_term_table.add_row([f"topic{i}"] + 
                [f"{self.vocab.get_token_by_id(term_id)}" for term_id in np.argsort(-dist)[:topk]])
        print(topic_term_table)

    def _inference_single(self, doc, max_iter, tolerance):
        # formula 85
        topic_distribution = np.zeros((len(doc), self.n_topics), dtype=np.float)
        for iteration in range(max_iter + 1):
            # 获得n_m_k，m只有一个
            topic_distribution_new = self.topic_term_distribution[:, doc].T

            topic_distribution_new *= topic_distribution.sum(axis=0) - topic_distribution + self.alpha
            topic_distribution_new /= topic_distribution_new.sum(axis=1)[:, np.newaxis]
            delta_naive = np.abs(topic_distribution_new - topic_distribution).sum()

            topic_distribution = topic_distribution_new
            
            if delta_naive < tolerance:
                break
        theta_doc = topic_distribution.sum(axis=0) / topic_distribution.sum()

        return theta_doc

    def initialize(self, X):
        self.n_documents, self.n_terms = X.shape

        self.document_topic_count = np.zeros((self.n_documents, self.n_topics), dtype=np.intc)
        self.topic_term_count = np.zeros((self.n_topics, self.n_terms), dtype=np.intc)
        # record every topic has how many terms
        self.topic_count = np.zeros(self.n_topics, dtype=np.intc)

        doc_i, term_j = np.nonzero(X)
        term_count = X[doc_i, term_j]

        self.Doc_is = np.repeat(doc_i, term_count).astype(np.intc)
        self.Term_js = np.repeat(term_j, term_count).astype(np.intc)
        # every term has its own topic index.
        self.term_topic = np.empty_like(self.Doc_is, dtype=np.intc)

        self.docs_term_count = len(self.Doc_is)
        
        for i in range(self.docs_term_count):
            d, t = self.Doc_is[i], self.Term_js[i]

            # random topic assignment
            topic_sample = i % self.n_topics
            self.term_topic[i] = topic_sample

            self.document_topic_count[d, topic_sample] += 1
            self.topic_term_count[topic_sample, t] += 1
            self.topic_count[topic_sample] += 1

        self.loglikelihoods = []
    

    def gibbs_sample(self):

        def topic_prob_func(document_id, topic_id, term_id):
                # forumula 79
                # 求出每个topic的概率 p(z_i=k | z_noti, w)
                #
                # Assume: every term/topic 's beta/alpha is the same value.
                # beta size: self.n_terms
                # beta_sum : beta * n_terms
                topic_prob = ((self.topic_term_count[topic_id, term_id] + self.beta) / 
                    (self.topic_count[topic_id] + self.beta * self.n_terms) * 
                    (self.document_topic_count[document_id, topic_id] + self.alpha))
                return topic_prob

        for i in range(self.docs_term_count):
            d, t = self.Doc_is[i], self.Term_js[i]
            old_topic = self.term_topic[i]

            self.document_topic_count[d, old_topic] -= 1
            self.topic_term_count[old_topic, t] -= 1
            self.topic_count[old_topic] -= 1

            topic_props_inc = np.zeros(self.n_topics, dtype=np.float)
            for topic in range(self.n_topics):
                topic_props_inc[topic] = topic_prob_func(d, topic, t) + (0 if topic == 0 else topic_props_inc[topic-1])

            #topic_props = [topic_prob_func(d, topic, t) for topic in range(self.n_topics)]
            #topic_props_inc = reduce(add, topic_props)
            
            # random a prob and find its corresponding topic_id
            random_prop = randint(0, 1000-1) / 1000 * topic_props_inc[-1]
            sample_topic = bisect_left(topic_props_inc, random_prop)

            self.document_topic_count[d, sample_topic] += 1
            self.topic_term_count[sample_topic, t] += 1
            self.topic_count[sample_topic] += 1
            self.term_topic[i] = sample_topic

    def construct_distribution(self):
        # formula 82
        # Topic distribution in documents

        topic_term_distribution = (self.topic_term_count + self.beta)
        topic_term_distribution /= np.sum(topic_term_distribution, axis=1)[:, np.newaxis]

        document_topic_distribution = (self.document_topic_count + self.alpha)
        document_topic_distribution /= np.sum(document_topic_distribution, axis=1)[:, np.newaxis]

        self.topic_term_distribution = topic_term_distribution
        self.document_topic_distribution = document_topic_distribution


    def loglikelihood(self):
        # formula 96
        # 困惑度越高，代表越难预测
        # 概率相乘取log

        document_props = np.zeros(self.n_documents)
        document_lengths = np.zeros(self.n_documents)
        for i in range(self.docs_term_count):
            d, t = self.Doc_is[i], self.Term_js[i]

            document_props[d] += np.log(
                sum(self.topic_word_distribution[topic, t] * self.document_topic_distribution[d, topic]
                for topic in range(self.n_topics)))

            document_lengths[d] += 1

        return pow(np.e, -sum(document_props) / sum(document_lengths))            



if __name__ == "__main__":
    import datasets
    vocab = datasets.load_reuters_vocab()
    X = datasets.load_reuters()

    docs = [[] for _ in range(X.shape[0])]
    for i, doc in enumerate(X):
        term_ids = np.nonzero(doc)[0]
        for term_id in term_ids:
            docs[i] += [vocab[term_id]] * doc[term_id]
        
    model = LDA()
    model.fit_documents(docs)
    model.inference(docs)
    model.display_distribution()