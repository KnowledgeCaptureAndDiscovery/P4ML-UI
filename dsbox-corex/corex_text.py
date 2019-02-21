from sklearn import preprocessing
#import primitive

# Import corex_topic
import os
import sys
corex_path = os.path.dirname(
    os.path.abspath(__file__)) + os.sep + "corex_topic"
sys.path.append(corex_path)
from corex_topic import Corex

from collections import defaultdict
from scipy import sparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TFIDF:  # (Primitive):
    def __init__(self, replace_df=True, raw_data_path=None, **kwargs):
        '''TO DO:  Add functionality to analyze multiple columns together'''
        self.cls = 'corex_primitives.tfidf'
        #super().__init__(name = 'TFIDF', cls = 'corex_primitives.tfidf')

        self.bow = TfidfVectorizer(decode_error='ignore', **kwargs)

        self.replace_df = True  # necessary?
        self.raw_data_path = raw_data_path

    def fit(self, X, y=None):
        '''X = df[col] for single text column'''
        self.columns = list(X)
        if len(self.columns) > 1:
            self.columns = self.columns[0] if self.columns[0] != X.index.name else self.columns[1]
            print('WARNING: Only first column being analyzed', self.columns)
            # TO DO: handle multiple? naming in transform... modify read_text

        if self.raw_data_path is not None:
            # for col in self.columns: #multiple columns?
            self.idf_ = self.bow.fit_transform(
                self._read_text(X, self.columns, self.raw_data_path))
        else:  # data frame contains raw text
                        # for col in self.columns:
            self.idf_ = self.bow.fit_transform(X[col].values)
        return self

    def transform(self, X, y=None):
        # just let X be new dataframe of single column
        self.idf_df = pd.DataFrame(columns=str(
            self.columns + '_tfidf'), index=X.index)
        self.idf_df.fillna()
        self.idf_df = self.idf_df.astype('object')

        for i, _ in self.idf_df.iterrows():
            # for col in self.columns:
            self.idf_df.loc[i, self.columns] = self.idf_[i, :]

        # returns dataframe in order to put array rows in each index
        return self.idf_df

        # returns matrix (n_samples, n_vocab) instead of a single pandas df
        # return self.idf_

    def fit_transform(self, X, y=None):
        # only set up for one column... last column will be stored / returned if multiple given
        self.fit(X, y)
        return self.transform(X, y)

    def _read_text(self, df, columns, data_path):
        column = columns  # still takes column just in case want to switch back
        df = pd.DataFrame(df)
        # for col in columns:
        for i, _ in df.iterrows():
            file = df.loc[i, column]
            # legacy check for faulty one-hot encoding
            file = file if not isinstance(file, int) else str(file) + '.txt'
            with open(os.path.join(data_path, file), 'r') as myfile:
                yield myfile.read()

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'TFIDF'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = 'NA'
        self._annotation.ml_algorithm = 'NA'
        self._annotation.tags = ['feature_extraction', 'text']
        return self._annotation

    def _get_feature_names(self):
        return ['tfidf']  # col + '_' + 'tfidf' in L2 execution


class CorexText:  # (Primitive):

    def __init__(self, n_hidden=None, **kwargs):
        self.n_hidden = 10 if n_hidden is None else n_hidden  # DEFAULT = 10 topics
        # no real need to pass extra Corex parameters.  Can be done if TFIDF primitive split
        self.model = Corex(n_hidden=self.n_hidden)  # , **kwargs)

        self.bow = TfidfVectorizer(decode_error='ignore', **kwargs)
        # if max_factors not None and n_hidden is None:
       # 	self.n_hidden = int(max_factors/len(self.columns))
        # else:

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X, y=None):  # TAKES IN DF with index column
        #self.columns = list(X)
        # X_ = X[self.columns].values # useless if only desired columns are passed
        bow = self.bow.transform(X.values.ravel())
        factors = self.model.transform(bow)
        return factors

    def fit_transform(self, X, y=None):  # TAKES IN DF with index column
        #self.columns = list(X)
        # X_ = X[self.columns].values # useless if only desired columns are passed

        bow = self.bow.fit_transform(X.values.ravel())
        factors = self.model.fit_transform(bow)
        return factors

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'CorexText'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = 'UnsupervisedLearning'
        self._annotation.ml_algorithm = ['Dimension Reduction']
        self._annotation.tags = ['feature_extraction', 'text']
        return self._annotation

    def get_feature_names(self):
        return ['CorexText_' + str(i) for i in range(self.n_hidden)]
