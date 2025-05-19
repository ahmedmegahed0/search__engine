import math
from collections import defaultdict, Counter
from preprocessing import preprocess_text

def document_term_incidence(docs, query):
    query_terms = preprocess_text(query)
    results = []
    for doc_id, tokens in docs.items():
        if any(term in tokens for term in query_terms):
            results.append(doc_id)
    return results

def inverted_index_search(docs, query):
    inverted_index = defaultdict(list)
    for doc_id, tokens in docs.items():
        for token in set(tokens):
            inverted_index[token].append(doc_id)
    query_terms = preprocess_text(query)
    result_docs = set()
    for term in query_terms:
        result_docs.update(inverted_index.get(term, []))
    return sorted(result_docs)

def tfidf_search(docs, query):
    query_terms = preprocess_text(query)
    all_terms = list(set([term for tokens in docs.values() for term in tokens]))
    tfidf_scores = {}

    def tf(term, tokens):
        return tokens.count(term) / len(tokens) if tokens else 0

    def idf(term):
        doc_count = sum(1 for tokens in docs.values() if term in tokens)
        return math.log(1 + (len(docs) / (1 + doc_count)))

    for doc_id, tokens in docs.items():
        score = 0
        for term in query_terms:
            score += tf(term, tokens) * idf(term)
        tfidf_scores[doc_id] = score

    norm = math.sqrt(sum(v**2 for v in tfidf_scores.values()))
    if norm == 0:
        return {}

    for doc_id in tfidf_scores:
        tfidf_scores[doc_id] /= norm

    return dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))


def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return numerator / denominator

def cosine_similarity_search(docs, query):
    query_terms = preprocess_text(query)
    query_vec = Counter(query_terms)

    scores = {}
    for doc_id, tokens in docs.items():
        doc_vec = Counter(tokens)
        score = cosine_similarity(query_vec, doc_vec)
        scores[doc_id] = score

    # ترتيب النتائج تنازليًا حسب التشابه
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
