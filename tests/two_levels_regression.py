import unittest
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.datasets import load_boston

from os import listdir
from os.path import isfile, join

from ensembler import ensemble


class TwoLevelsTest(unittest.TestCase):
	def multilevel_ensembling(self, train, y):

	    level1 = {'Lasso':LassoCV(), 'RidgeCV':RidgeCV()}
	    level2 = {'Linear Regression':LinearRegression()}

	    model = ensemble.Ensemble(levels = [level1, level2], store_oof=True)
	    model.fit_predict(train, y)

	def test_classification(self):
		(train, y) = load_boston(return_X_y = True)
		train = pd.DataFrame(train)
		self.multilevel_ensembling(train, y)

if __name__ == '__main__':
	unittest.main()