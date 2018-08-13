""" Multi-Level Ensemble.

This module allows to automatize a multi-level blending using strategy A, as mentionned in
https://www.kaggle.com/general/18793

Todo:
	* Improve documentation
	* Predict Test set - DONE 16/3
	* Improve classification and regression choice
	* Add a way to handle xgboost, lgbm...
	* Fix CV
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import rankdata

import os

class Ensemble:
	"""Ensemble is a pipeline for fitting every stacking level you want to add in your final architecture.
	"""
	def __init__(self, levels = [], metric = mean_squared_error, cv = None, correlation_threshold=None,
				binary_scale = False, stratified_k_fold = False, store_oof = False):
		"""__init__ creates the Level objects. There could be as many as level as you want.
		
		Args:
			levels (list of dict): levels is a list of dictionnaries whose attributes are the models you
				want to incorporate at a given level.
			metric (:obj:`sklearn.metrics`, optional): Metric used for assessing your models' performance.
			cv (int): CV object
			correlation_threshold (float): Between 0 and 1 - Before training models - check the correlation between features and drop one if 
				it has a pearson's correlation with another feature > threshold.
			binary_scale (bool, optional): Whether the models' outputs should be scaled to [0,1].
			store_oof (bool, optional): Whether the intermediate predictions should be saved or not.
		"""
		Level.n_level = 0
		self.levels=[]
		self.metric = metric
		self.store_oof = store_oof

		if cv == None:
			self.cv = KFold(n_splits = 5, shuffle = True, random_seed = 1994)
		else:
			self.cv = cv

		for i, models in enumerate(levels):
			self.levels.append(Level(models = models, cv = self.cv, corr_thresh = correlation_threshold,
							binary_scale = binary_scale, stratified_k_fold = stratified_k_fold))

	def save_predictions(self, predict_train, predict_test, i):
		cwd = os.getcwd()

		filename = os.path.join(cwd,'log','level_'+str(i)+'_OOF.pkl')
		predict_train.to_pickle(filename)

		filename = os.path.join(cwd,'log','level_'+str(i)+'_prediction.pkl')
		predict_test.to_pickle(filename)


	def fit_predict(self, df, y, test = pd.DataFrame()):
		"""Iteratively fits and predicts the models in each levels.
		For the first level - the fitting is done using df. For the following levels, the out of fold predictions from
		the previous level are used. By default - a simple KFold() method is used.
		
		Args:
			df (:obj:`pd.DataFrame`): Training set for the first level.
			y (:obj:`pd.DataFrame` or :obj:`np.array`): Ground truth values of the target.
			test (:obj:`pd.DataFrame`, optional): Testing set - Predicition will be made on this set using models fitted with df and y
		"""
		if self.store_oof:
			cwd = os.getcwd()
			dir_name = os.path.join(cwd,'log')
			if not os.path.exists(dir_name):
				os.mkdir(dir_name)

		for i, level in enumerate(self.levels):
			print('Fitting Level {}/{}...'.format(i+1, Level.n_level))
			# Case with a single level - Only predict the prediction for train and test using the training set without cross validation.
			if Level.n_level == 1:
					predict_train, predict_test = level.fit_predict(df, y, test, metric=self.metric)
			# If there are more than one level - Train the first level with the initial training set and get the OOF prediction.
			elif i == 0:
				predict_train, predict_test = level.fit_predict_cv(df, y, test, metric=self.metric)
			# Last level - doesn't use cross validation
			elif i == Level.n_level-1:
				predict_train, predict_test = level.fit_predict(predict_train, y, predict_test, metric=self.metric)
			# Predict the intermediate levels with the previous OOF prediction as training set
			else:
				predict_train, predict_test = level.fit_predict_cv(predict_train, y, predict_test, metric=self.metric)
			
			if self.store_oof:
				self.save_predictions(predict_train, predict_test, i)


		return predict_train, predict_test


class Level():
	""" Level contains every models in a given level.
	Args:
		n_level (int): Total number of levels in Ensemble.
	"""
	n_level = 0
	def __init__(self, models, cv, corr_thresh, binary_scale, stratified_k_fold):
		"""__init__ adds the models to the level.
			
		Args:
			models (dict): Contains the models - keys are the models' name and attributes the models' object.
			cv (int): Number of folds for the cross validation
			corr_thresh (float): Threshold for removing multicollinearity between dataset or models (in case of several level)
			binary_scale (bool): Whether the models' outputs should be scaled to [0,1]
		"""
		Level.n_level += 1
		self.cv = cv
		self.corr_thresh = corr_thresh
		self.binary_scale = binary_scale
		self.model_names = list(models.keys())
		self.stratified_k_fold = stratified_k_fold

		print('Level {} - {} Models.'.format(Level.n_level, len(models)))
		for model in models:
			print(model)
		self.models = models

	def fit(self, df, y):
		"""Calls iteratively the fit function of every models.
		
		Args:
			df (:obj:`pd.DataFrame`): Training set for the first level.
			y (:obj:`pd.DataFrame` or :obj:`np.array`): Ground truth values of the target.
		"""
		for i, model_name in enumerate(self.model_names):
			#print(' {}...'.format(model_name))
			self.models[model_name].fit(df, y)

	def predict(self, df):
		"""Calls iteratively the predict function of every models.

		Args:
			df (:obj:`pd.DataFrame`): Training set for the first level.
		"""
		predictions = np.zeros((len(df), len(self.models)))
		for i, model_name in enumerate(self.model_names):
			#print(' {}...'.format(model_name))
			try:
				predictions[:, i] = self.models[model_name].predict_proba(df)[:,1]
				print('{} - Classifier detected...probability of 1 predicted'.format(model_name))
			except:
				predictions[:, i] = self.models[model_name].predict(df).reshape(1,-1)[0]
				if self.binary_scale:
					predictions[:, i] = rankdata(predictions[:, i])/len(predictions[:, i])

		return predictions

	def fit_predict_cv(self, df, y, test, metric):
		"""Uses a cross validation scheme to fit the model and retrieve out of folds prediction.
		Predictions on test are done at each fold and averaged.
		
		Args:
			df (:obj:`pd.DataFrame`): Training set for the first level.
			y (:obj:`pd.DataFrame` or :obj:`np.array`): Ground truth values of the target.
			test (:obj:`pd.DataFrame`, optional): Testing set - Predicition will be made on this set using models fitted with df and y
			metric (:obj:`sklearn.metrics`): Metric used for assessing models` performance
			kf (:obj:`sklearn.KFold` or :obj:`sklearn.StratifiedKFold`): Cross validation method.
		"""
		#n_folds = self.cv
		#if self.stratified_k_fold:
		#	kf = StratifiedKFold(n_splits = n_folds, random_state = 55, shuffle= True)
		#else:
		#	kf = KFold(n_splits = n_folds, random_state = 55, shuffle= True)
		n_folds = self.cv.get_n_splits()

		if self.corr_thresh:
			to_drop = check_multicollinearity(df)
			print('{} will be dropped - Highly correlated to another feature'.format(to_drop))
			df.drop(to_drop, axis = 1, inplace = True)

		OOF_prediction = np.zeros((len(df), len(self.models)))
		test_prediction = np.zeros((len(test), len(self.models)))

		score_CV = np.zeros((n_folds, len(self.models)))
		for i, (train_index, valid_index) in enumerate(self.cv.split(df, y)):
			df_train = df.loc[train_index]
			df_valid = df.loc[valid_index]
			y_train = y[train_index]
			y_valid = y[valid_index]

			print('CV - Fold {}'.format(i+1))

			self.fit(df_train, y_train)
			OOF_prediction[valid_index, :] = self.predict(df_valid)
			if len(test) > 0:
				test_prediction += self.predict(test)/n_folds

			for k in np.arange(0, OOF_prediction.shape[1]):
				score_CV[i, k] = metric(y_valid, OOF_prediction[valid_index, k])
				print('{0} : {1:.5f}'.format(self.model_names[k], score_CV[i, k]))

		# Print scores
		self.print_score(score_CV)

		OOF_prediction = pd.DataFrame(OOF_prediction, columns=self.models.keys())
		test_prediction = pd.DataFrame(test_prediction, columns=self.models.keys())

		return OOF_prediction, test_prediction

	def fit_predict(self, df, y, test, metric):
		"""Perform fit and predict - with no cross validation.
		
		Args:
			df (:obj:`pd.DataFrame`): Training set for the first level.
			y (:obj:`pd.DataFrame` or :obj:`np.array`): Ground truth values of the target.
			test (:obj:`pd.DataFrame`, optional): Testing set - Predicition will be made on this set using models fitted with df and y
			metric (:obj:`sklearn.metrics`): Metric used for assessing models` performance
		"""
		if self.corr_thresh:
			to_drop = check_multicollinearity(df)
			print('{} will be dropped - Highly correlated to another feature'.format(to_drop))
			df.drop(to_drop, axis = 1, inplace = True)

		self.fit(df, y)
		prediction = self.predict(df)
		test_prediction = np.zeros((len(test), len(self.models)))

		if len(test) > 0:
			test_prediction = self.predict(test)

		for k in np.arange(0, prediction.shape[1]):
			print(metric(y, prediction[:,k]))
			
		prediction = pd.DataFrame(prediction, columns=self.models.keys())
		test_prediction = pd.DataFrame(test_prediction, columns=self.models.keys())

		return prediction, test_prediction 
		
	def check_multicollinearity(self, df):
		"""Compute correlation between features - Return features to drop (if correlation > self.corr_thresh)
		Args:
			df (:obj:`pd.DataFrame`): Dataset containing features.
		"""
		
		corr = df.corr().abs()
		# Select the upper triangle
		upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
		to_drop = [col for col in upper.columns if any(upper[col] > self.corr_thresh)]
		
		return to_drop

	def print_score(self, score):
		for i, model in enumerate(self.model_names):
			score_mean = score[:, i].mean()
			score_std = score[:, i].std()
			print('{0} Score : {1:.6f} ({2:.6f})'.format(model, score_mean, score_std))
