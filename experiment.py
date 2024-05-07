
import sys
sys.path.append("G:\My Drive\CurrentWork\Robust PPM") # Add the path
import torch
from adversarial_generation.adversarial_example_generation import AdversarialAttacks_manifold
from util.settings import global_setting, Training_Parameters

from util.DatasetManager import DatasetManager

import numpy as np
import pandas as pd
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(22)
pd.set_option('display.max_rows', 100)
  
# the event logs, classifiers, attack types and attack strategies
dataset_names = ['production', 
                 'bpic2012_accepted', 'bpic2012_cancelled', 'bpic2012_refused', 
                 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2',
                 'sepsis_cases_2']
cls_methods = ['LR', 'RF', 'XGB', # ML models
               'LSTM', 'GAN', 'BNN'] # DL models

attack_types = ['last_event', 'all event'] 

attack_cols = ['Activity']
attack_strategies = ['adversarial_attack', 'manifold_projection', 'latent_sampling']

######################################################################################################
############################ The experiment starts here ##############################################
######################################################################################################
force_attack = False
force_train = False
dataset_name = 'sepsis_cases_2'
cls_method = 'LSTM'
attack_type = 'last_event'

results_dir = global_setting['results_dir'] + '/' + dataset_name+'/'+ cls_method

# START
print('the dataset is:', dataset_name)
print('the classifier is:', cls_method)
print('the attack columns are:', attack_cols)

dataset_manager = DatasetManager(dataset_name, cls_method)
dt_train, dt_test, train_y, test_y, train_cases, test_cases, vocab_size, max_prefix_length = dataset_manager.create_data(cat_cols = attack_cols, prefix_generation=False)

# define parameters
training_params = Training_Parameters(dataset_name, dataset_manager, cls_method, vocab_size, force_train)
vae_params = training_params.get_vae_params()
model = training_params.get_model(cls_method, dt_train, dt_test, train_y, test_y)
attack_params = training_params.get_attack_params(vocab_size, max_prefix_length)

# the adversarial attacks   
attack_manifold = AdversarialAttacks_manifold(dataset_manager, max_prefix_length, 
                                              dt_train = dt_train, dt_test = dt_test, train_y = train_y, test_y = test_y, train_cases = train_cases,  test_cases = test_cases, 
                                              model = model,
                                              training_params = training_params, vae_params = vae_params)

attack_manifold.regular_adversarial_examples(attack_type, attack_cols[0], cls_method, force_attack, force_train)
#attack_manifold.option_2(cls_method, force_attack, force_train, 100)
#attack_manifold.option_3(cls_method, force_attack, force_train, 100)
#attack_manifold.option_4(vocab_size, max_prefix_length)