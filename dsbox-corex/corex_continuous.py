from sklearn import preprocessing
#import primitive
import sys
import os
import LinearCorex.linearcorex.linearcorex as corex_cont
#import LinearCorex.linearcorex as corex_cont
from collections import defaultdict
from scipy import sparse
import pandas as pd

class CorexContinuous:  #(Primitive):

    def __init__(self, n_hidden = None, latent_pct = None, **kwargs):
    	'''TO DO: Prune/add initialization arguments'''
        # super().__init__(name = 'CorexContinuous', cls = 'corex_primitives.CorexContinuous') # inherit from primitive?
        #self.columns = columns if not columns else columns 
        #default for no columns fed?  do nothing rather than take all

        self.latent_pct = latent_pct if (latent_pct and not n_hidden) else None
 
        # if latent_pct is not None and n_hidden is None:
        # 	self.n_hidden = int(len(self.columns)*latent_pct)
        # else:
        # 	self.n_hidden = 2 if n_hidden is None else n_hidden #DEFAULT = 2 
        if n_hidden:
            self.n_hidden = n_hidden
            self.model = corex_cont.Corex(n_hidden= self.n_hidden, **kwargs)
        else:
        	if latent_pct is None:
        		self.n_hidden = 2
        		self.model = corex_cont.Corex(n_hidden= self.n_hidden, **kwargs)
        	else:
        		self.kwargs = kwargs
        		self.model = None

    def fit(self, X):
        
        self.fit_transform(X)
        return self

    def transform(self, X, y = None): 
       
        self.columns = list(X)
        X_ = X[self.columns].values 

        if self.model is None and self.latent_pct:
        	self.model = corex_cont.Corex(n_hidden= int(self.latent_pct*len(self.columns)), **self.kwargs)
    	
    	factors = self.model.transform(X_)
        return factors

    def fit_transform(self, X, y = None):
        
        self.columns = list(X)
     	X_ = X[self.columns].values
        
        if self.model is None and self.latent_pct:
        	self.model = corex_cont.Corex(n_hidden= int(self.latent_pct*len(self.columns)), **self.kwargs)

    	factors = self.model.fit_transform(X_)
        return factors

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'CorexContinuous'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = 'UnsupervisedLearning'
        self._annotation.ml_algorithm = ['Dimension Reduction']
        self._annotation.tags = ['feature_extraction'] #'continuous'
        return self._annotation

    def get_feature_names(self):
    	return ['CorexContinuous_'+ str(i) for i in range(self.n_hidden)]

