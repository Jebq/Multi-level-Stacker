import unittest
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score

from os import listdir
from os.path import isfile, join

from ensembler import ensemble


class TwoLevelsTest(unittest.TestCase):
	def multilevel_ensembling(self, train, y, test):

	    level1 = {'Lasso':LassoCV(), 'LogisticRegression':LogisticRegressionCV()}
	    level2 = {'Ridge':RidgeCV()}

	    model = ensemble.Ensemble(levels = [level1, level2], binary_scale=True, stratified_k_fold = True)
	    model.fit_predict(train, y, test)

	def test_classification(self):
		print('Reading files...')
		path = 'tests/data/'
		train_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('train_oof.csv')]
		test_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('test_oof.csv')]

		train = []
		test = []
		for i, file in enumerate(train_files):
		    train.append(pd.read_csv(path+file)['toxic'].values)

		for i, file in enumerate(test_files):
		    test.append(pd.read_csv(path+file)['toxic'].values)

		train = pd.DataFrame(np.array(train).transpose())
		test = pd.DataFrame(np.array(test).transpose())

		y = pd.read_csv(path +'labels.csv')['toxic']
		self.multilevel_ensembling(train, y, test)

if __name__ == '__main__':
	unittest.main()