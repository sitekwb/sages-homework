from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import pandas as pd
from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
from gensim.models import Word2Vec


class FakeNewsDataset(Dataset):
    def __init__(self, bodies_path: str, stances_path: str, w2v_model_path: str, text_input_len: int):
        super().__init__()
        self.bodies_path = bodies_path
        self.stances_path = stances_path
        self.bodies = None
        self.stances = None
        self.label_dictionary = None
        self.w2v_model: Optional[Word2Vec] = None
        self.w2v_model_path = w2v_model_path
        self.text_input_len = text_input_len
        self.setup()

    def setup(self):
        self.bodies = pd.read_csv(self.bodies_path)
        self.stances = pd.read_csv(self.stances_path)
        self.label_dictionary = {text: i for i, text in enumerate(self.stances['Stance'].unique())}
        self.w2v_model = Word2Vec.load(self.w2v_model_path)

    def preprocess_text(self, text):
        tokenized_text = [_ for _ in tokenize(remove_stopwords(text, stopwords=stopwords.words('english')), to_lower=True)]
        stemmer = PorterStemmer()
        vectors = [self.w2v_model.wv[stemmer.stem(token)] for token in tokenized_text[:self.text_input_len]]
        vectors.extend([np.zeros((self.w2v_model.vector_size,))] * (max(self.text_input_len - len(vectors), 0)))
        return vectors

    def __getitem__(self, index: int) -> T_co:
        stances_element = self.stances.iloc[index]
        stances_dict = dict(stances_element)
        body_id = stances_dict['Body ID']
        headline_text = stances_dict['Headline']
        stance = self.label_dictionary[stances_dict['Stance']]
        bodies_element = self.bodies.loc[self.bodies["Body ID"] == body_id]
        bodies_dict = dict(bodies_element)
        body_text = bodies_dict['articleBody'].tolist()[0]
        headline = self.preprocess_text(headline_text)
        body = self.preprocess_text(body_text)
        headline.extend(body)
        x_tensor = torch.tensor(headline, dtype=torch.double)
        y_tensor = torch.tensor(stance, dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.stances)