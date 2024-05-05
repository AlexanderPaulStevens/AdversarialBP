# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:59:47 2023

@author: u0138175
"""
import sklearn
import os
import wandb
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)
from util.settings import global_setting, training_setting
clip = training_setting["clip"]
from sklearn.metrics import roc_auc_score
from util.load_model import load_trained_model, save_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

seed = global_setting['seed']

class MLModel:      
    def __init__(self, cls_method, dataset_name, dataset_manager, **kwargs):
        
        # Common parameters for all algorithms
        common_training_params = {"lr": None, "subsample": None, "max_depth": None, "colsample_bytree": None, "min_child_weight": None, "max_features": None, "C": None}
        # Update common parameters with specific parameters
        common_training_params.update(**kwargs)
        self.cls_method = cls_method
        self.dataset_name = dataset_name
        self.path = global_setting['models']
        self.training_params = common_training_params
        self.dataset_manager = dataset_manager

    def train_model(self, x_train, x_test, y_train, y_test):

        """
        Parameters
        ----------
        x_train: pd.DataFrame
            training features
        y_train: pd.DataFrame
            training labels
        x_test: pd.DataFrame
            test features
        y_test: pd.DataFrame
            test labels
        """

        print(f"balance on train set {y_train.mean()}, balance on test set {y_test.mean()}")

        if self.cls_method == 'LR':
            model = LogisticRegression(C=2**self.training_params["C"], solver='saga', penalty="l1", n_jobs=-1, random_state=seed, max_iter=10000)
        elif self.cls_method =="XGB":
            model = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate= self.training_params['learning_rate'],
                                                subsample=self.training_params['subsample'],
                                                max_depth=int(self.training_params['max_depth']),
                                                colsample_bytree=self.training_params['colsample_bytree'],
                                                min_child_weight=int(self.training_params['min_child_weight']),
                                                n_jobs=-1,
                                                seed=seed)

        elif self.cls_method == "RF":
            model = RandomForestClassifier(n_estimators=500,
                                          max_features=self.training_params['max_features'],
                                          n_jobs=-1,
                                          random_state=seed)

        preds_all = []
        x_train = self.dataset_manager.transform_data(x_train, traintest = 'train')
        x_test = self.dataset_manager.transform_data(x_test, traintest = 'test')
        model.fit(x_train, y_train)
        preds_pos_label_idx = np.where(model.classes_ == 1)[0][0]
        pred = model.predict_proba(x_test)[:, preds_pos_label_idx]
        preds_all.extend(pred)
        print('roc auc score', roc_auc_score(y_test, preds_all))

        return model

    "This is the first function that is called in the file 'experiment.py'"
    def train(
        self,
        x_train, x_test, y_train, y_test,
        force_train=False,
        attack= None):
        """

        Parameters
        ----------
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        batch_size: int
            Number of samples in each batch
        force_train: bool
            Force training, even if model already exists in cache.
        hidden_sizes: list[int]
            hidden_sizes[i] contains the embedding size
            hidden_sizes[-1] contains the number of lstm layers
        n_estimators: int
            Number of estimators in forest.
        max_depth: int
            Max depth of trees in the forest.

        Returns
        -------

        """
        
        x_train, _ = self.dataset_manager.ohe_cases(x_train)
        x_test, _ = self.dataset_manager.ohe_cases(x_test)
        
        x_train = self.dataset_manager.transform_data(x_train, traintest = 'train')
        x_test = self.dataset_manager.transform_data(x_test, traintest = 'test')
        
        filtered_params = {key: value for key, value in self.training_params.items() if value is not None}
        list(filtered_params.keys())
        save_string_list = list(filtered_params.values()) + [attack]
        save_name = f"{self.cls_method}_layers_{save_string_list}" # the name we use to save the model 

        # try to load the model from disk, if that fails train the model instead.
        self._model = None
        if not force_train:
            self._model = load_trained_model(
                save_name=save_name, data_name=self.dataset_name
            )

            # sanity check to see if loaded model accuracy makes sense
            if self._model is not None:
                self._test_accuracy(self._model, x_test, y_test)

        # if model loading failed or force_train flag set to true.
        if self._model is None or force_train:
            # get preprocessed data

            self._model = self.train_model(x_train, x_test, y_train, y_test)
            
            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.dataset_name)
        return self._model

    def _test_accuracy(self, model, x_test, y_test):
        # get preprocessed data
        preds_pos_label_idx = np.where(model.classes_ == 1)[0][0]
        pred = model.predict_proba(x_test)[:, preds_pos_label_idx]
        print('roc auc score', roc_auc_score(y_test, pred))
        print('check probability', pred)
        prediction = [0 if prob < 0.5 else 1 for prob in pred]

        print('set of prediction and test set values', set(prediction), set(y_test))
        print(f"test AUC for model: {sklearn.metrics.roc_auc_score(y_test, pred)}")
            
        print(f"test accuracy for model: {sklearn.metrics.roc_auc_score(y_test, prediction)}")