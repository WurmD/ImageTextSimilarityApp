
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    try:
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
    except LookupError as e:  # Resource [93mpunkt[0m not found.
        print(e)
        print('Downloading punkt')
        nltk.download('punkt')
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    # requires   python -m spacy download en_core_web_lg   for slightly better similarity values,
    # or simply   python -m spacy download en   for smaller instalation download (12MB vs 791MB)
    nlp = spacy.load('en')
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    # print('Spacy metric:', similarity)
    return similarity

    # tfidf = vectorizer.fit_transform([text1, text2])
    # return ((tfidf * tfidf.T).A)[0, 1]

