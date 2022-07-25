import pandas as pd
import nltk
from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
from gensim.models import Word2Vec


def make_dataset(bodies_path, stances_path, word2vec_save_path) -> int:
    bodies = pd.read_csv(bodies_path).set_index("Body ID")
    stances = pd.read_csv(stances_path)
    dataset_size = len(stances)

    texts = list(bodies['articleBody'])
    nltk.download('stopwords', quiet=True)
    texts.extend(list(stances['Headline']))
    tokenized_texts = [tokenize(remove_stopwords(text, stopwords=stopwords.words('english')), to_lower=True) for text in
                       texts]
    
    stemmer = PorterStemmer()
    processed_texts = list(map(lambda text: [stemmer.stem(token) for token in text], tokenized_texts))
    model = Word2Vec(sentences=processed_texts, vector_size=50, window=10, min_count=1, workers=0, sg=True, epochs=3)
    model.build_vocab(processed_texts)
    model.train(processed_texts, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(word2vec_save_path)

    return dataset_size

