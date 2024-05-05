from operator import itemgetter
#from attacks import AdversarialAttacks
import os
os.chdir('G:\My Drive\CurrentWork\Robust PPM')
import torch
import random
import numpy as np
import pandas as pd
from util.settings import global_setting
from util.load_dataset import load_trained_dataset, save_dataset
from adversarialBP.autoencoder.VAE import LSTM_VAE
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adversarialBP.generation_methods.AdversarialREVISE import AdversarialExamples
from scipy.spatial.distance import euclidean

#to evaluate the adversarial examples
from adversarialBP.evaluation.benchmark import Benchmark
import adversarialBP.evaluation.catalog as evaluation_catalog

from sklearn.metrics import roc_auc_score, accuracy_score
g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(32)
seed = global_setting['seed']
random.seed(seed)

class Manifold:
    """class to define the manifold and to project prefixes on the manifold.
    The code is written to handle both ML and DL models
    The functions are:
        - project_on_manifold: project the adversarial prefixes on the manifold
        - create_manifold_dataset: create the manifold dataset
    """
    def __init__(self, dataset_name, dataset_manager, max_prefix_length, train, test, train_y, test_y, training_params, vae_params):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.seed = global_setting['seed']
        self.path = global_setting['models']
        self.train_ratio = global_setting['train_ratio']
        self.params_dir = global_setting['params_dir_DL']
        self.best_manifold_path = global_setting['best_manifolds']
        self.results_dir = global_setting['results_dir'] + '/' + self.dataset_name
        self.max_prefix_length = max_prefix_length
        self.activity_col = dataset_manager.activity_col
        self.resource_col = dataset_manager.resource_col
        self.cat_cols = dataset_manager.cat_cols
        self.cols = dataset_manager.cols
        self.vocab_size = len(list(self.dataset_manager.ce.get_maps().values())[0].keys()) + 2 # padding token and EoS token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train = train
        self.test = test
        self.training_params = training_params
        self.threshold = 0.5 
        

        self.vae_regular = LSTM_VAE(
            dataset_name=self.dataset_name, dataset_manager=self.dataset_manager, label="regular", layers=vae_params["layers"], vocab_size=self.vocab_size, max_prefix_length=self.max_prefix_length, batch_size=vae_params["batch_size"], epochs=vae_params["epochs"])
  
        self.vae_deviant = LSTM_VAE(
            dataset_name=self.dataset_name, dataset_manager=self.dataset_manager, label="deviant", layers=vae_params["layers"], vocab_size=self.vocab_size, max_prefix_length=self.max_prefix_length, batch_size=vae_params["batch_size"], epochs=vae_params["epochs"])
  
        if vae_params["force_train"]:
            
            ans_regular, ans_deviant, label_regular, label_deviant, _, _ = self.dataset_manager.groupby_caseID(self.train)
            ###### CAT COL################
            _, ohe_regular = self.dataset_manager.ohe_cases(ans_regular)
            _, ohe_deviant = self.dataset_manager.ohe_cases(ans_deviant)

            self.vae_regular.fit(
                x_train=ohe_regular, y_train=label_regular,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
                label="regular"
            )
            self.vae_deviant.fit(
                x_train=ohe_deviant, y_train=label_deviant,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
                label="deviant"
            )
        
        else:
            try:
                self.vae_regular.load("regular")
                self.vae_deviant.load("deviant")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def project_on_latent_space(self, adversarial_prefixes, label):
        """Project onto the latent space."""
        if label ==0:
            vae_model = self.vae_regular
        elif label == 1:
            vae_model = self.vae_deviant
        vae_model.eval()
        with torch.no_grad():
            mu, log_var = vae_model.encode(adversarial_prefixes)
        return mu, log_var
     
    def on_manifold_projected(self, adversarial_prefixes, label):
        """Projected adversarial examples"""

        if label =='regular':
            model = self.vae_regular
        elif label == 'deviant':
            model = self.vae_deviant

        adversarial_prefixes_label = adversarial_prefixes.copy()
        activity, _, _ = self.dataset_manager.groupby_pad(adversarial_prefixes_label)
        activity = activity.to(self.device)
        model.eval()
        # project the adversarial prefixes on the manifold
        x_hat_param_act1, x_hat_param_res1, _, _= model(activity)
        ans,_,_ = self.dataset_manager.groupby_caseID(adversarial_prefixes_label, self.activity_col)
        manifold_activities = self.dataset_manager.prefix_lengths_adversarial(ans, x_hat_param_act1)
        activities = np.concatenate([x.ravel() for x in manifold_activities])
        #manifold_resources = self.dataset_manager.prefix_lengths_adversarial(ans, x_hat_param_res1)
        #resources = np.concatenate([x.ravel() for x in manifold_resources])
        on_manifold_adversarial = adversarial_prefixes_label.copy()
        on_manifold_adversarial['Case ID_label'] = on_manifold_adversarial['Case ID'] + label
        on_manifold_adversarial['Activity'] = activities
        #on_manifold_adversarial['Resource'] = resources
        #on_manifold_adversarial = self.ce.inverse_transform(on_manifold_adversarial)
        return on_manifold_adversarial
    
    def get_latent_representations(self, adversarial_prefixes, label):
        """
        This function is used to get the latent representations of the adversarial prefixes.
        """
        if label =='regular':
            model = self.vae_regular
        elif label == 'deviant':
            model = self.vae_deviant
        model.eval()
        adversarial_prefixes_label = adversarial_prefixes.copy()

        activity, _, _ = self.dataset_manager.groupby_pad(adversarial_prefixes_label)
        activity = activity.to(self.device)

        # project the adversarial prefixes to the latent space
        input_tensor, _, mu, logvar = model(activity)

        return input_tensor, mu, logvar

    def create_manifold_dataset(self, adversarial_prefixes_train):
        """Manifold experiment DL."""
        # on-manifold training data
        on_manifold_adversarial_regular = pd.DataFrame()
        on_manifold_adversarial_deviant = pd.DataFrame()
        adversarial_prefixes_train_regular = adversarial_prefixes_train[adversarial_prefixes_train.label == 0]
        adversarial_prefixes_train_deviant = adversarial_prefixes_train[adversarial_prefixes_train.label == 1]
        if len(adversarial_prefixes_train_regular) > 0:
            on_manifold_adversarial_regular = self.project_on_manifold(adversarial_prefixes_train_regular, 'regular')
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_adversarial_deviant = self.project_on_manifold(adversarial_prefixes_train_deviant, 'deviant')
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])
        else:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        train_y_manifold = self.dataset_manager.get_label_numeric(on_manifold_prefixes_train_total)
        return on_manifold_prefixes_train_total, train_y_manifold
    
class Attack:
    """class to define the regular attacks.
    The functions are:
        - permute_last_event: permute the last event
        - permute_all_event: permute all the events
        - check_adversarial_cases_DL: check for adversarial cases for DL
        - check_adversarial_cases_ML: check for adversarial cases for ML
    """
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
  
    def permute_last_event(self, trace):
        """Permute the last event of each of the prefixes with a random value from this list.
        """
        possible_values = self.activity_values.copy()
        possible_values.remove(trace[-1])
        result = [[*trace[:-1], value] for value in possible_values] # Create a list of lists where each inner list has the last value replaced by each value from activity_values      
        # Print the result
        return result
    
    def permute_all_event(self, prefixes, j, num_permutations):
        """Permute all the events."""

        for key, value in self.ce.get_maps().items():
            values = []
            if j in key:
                values.append(value)

        prefixes[j] = random.choices(values, k=len(prefixes))
        return prefixes

    def permute_random_event(self, prefixes, j, num_permutations):
        """to do"""
        return prefixes
    
    def check_adversarial_cases_DL(self, data_adv):
        """Check for adversarial cases."""
        #dt_prefixes2 = dt_prefixes_original.copy()
        #dt_prefixes2 = dt_prefixes2[self.cols]
        activity, y_A, cases_adv = self.dataset_manager.groupby_pad(data_adv)
        activity = activity.to(self.device)

        pred = self.model.predict_proba([activity]).cpu().detach().numpy()
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))
        indices = self.dataset_manager.return_indices_adversarial_guess(y_A, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases

        elif len(indices) == 1:
            adversarial_cases = [cases_adv[int(indices[0])]]
        else:
            adversarial_cases = list(itemgetter(*indices)(cases_adv))

        dt_prefixes2 = data_adv[data_adv['Case ID'].isin(adversarial_cases)]
        activity, y_A, cases_adv = self.dataset_manager.groupby_pad(dt_prefixes2)
        activity = activity.to(self.device)

        pred = self.model.predict_proba([activity]).cpu().detach().numpy()
        pred = self.model.predict_proba([activity]).cpu().detach().numpy()
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))
        print('adversarial predicted', accuracy_score(y_A , pred)) 

        return adversarial_cases
    
    def check_adversarial_cases_ML(self, data, y2, caseIDs):
        """Check for adversarial cases."""
        dt_named = self.dataset_manager.transform_data_test
        y2 = self.dataset_manager.get_label_numeric(data)
        pred = self.model.predict_proba(dt_named)[:,-1]
        pred = (np.where(np.array(pred) > self.threshold , 1, 0))
        indices = self.dataset_manager.return_indices_adversarial_guess(y2, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases
        adversarial_cases = list(itemgetter(*indices)(caseIDs))
        return adversarial_cases
    
class AdversarialAttacks_manifold(Attack, Manifold):
    """class to define on-manifold adversarial attacks.
    The functions in this class are based on the design choices we made and explained in the paper.

    The functions are:
        - correctly_predicted_prefixes: check which prefixes are correctly predicted
        - create_adversarial_data: create the adversarial data
        - create_adversarial_prefixes: create the adversarial prefixes
        - perform_attack: perform the attack
    """
    def __init__(self, dataset_manager, max_prefix_length, dt_train, dt_test, train_y, test_y, train_cases, test_cases, model, training_params, vae_params):
        self.dataset_manager = dataset_manager
        self.dataset_name = self.dataset_manager.dataset_name

        Attack.__init__(self, dataset_manager)
        Manifold.__init__(self, dataset_manager.dataset_name, dataset_manager, max_prefix_length, dt_train, dt_test, train_y, test_y, training_params, vae_params)
        self.cls_method = self.dataset_manager.cls_method
        self.max_prefix_length = max_prefix_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train = dt_train
        self.test = dt_test
        self.train_y = train_y
        self.test_y = test_y
        self.train_cases = train_cases
        self.test_cases = test_cases
        self.model = model
        self.cat_cols = dataset_manager.cat_cols
        self.cols = dataset_manager.cols
        self.activity_col = dataset_manager.activity_col
        self.resource_col = dataset_manager.resource_col
        self.ce = dataset_manager.ce
        self.activity_values = list(list(self.ce.get_maps().values())[0].values())
        
        if self.cls_method in ['LR', 'RF', 'XGB']:
            self.algorithm = 'ML'
        elif self.cls_method in ['LSTM', 'GAN', 'BNN']:
            self.algorithm = 'DL'
            
    def correctly_predicted_prefixes(self, data, y, cases):
        """Check which prefixes are correctly predicted based on the specified algorithm (ML or DL)."""
        
        print('amount of original cases', len(cases))
        pred = self.predict_proba(data)    
        pred = np.where(pred > self.threshold, 1, 0)  
        indices = self.dataset_manager.return_indices_correlated_guess(y, pred)
        correct_cases = list(itemgetter(*indices)(cases))
        print('amount of correctly predicted cases', len(correct_cases))
        correct_y = list(itemgetter(*indices)(y))
        if self.algorithm == 'ML':
            correct_data = list(itemgetter(*indices)(data))
        elif self.algorithm == 'DL':
            correct_data = list(itemgetter(*indices)(data)) # Convert data to tensor
        
        # just to double check whether they where all correctly predicted
        correct_pred = self.predict_proba(correct_data)
        correct_pred = np.where(correct_pred > self.threshold, 1, 0)  
        assert accuracy_score(correct_pred, correct_y) == 1.0
        
        return correct_data, correct_y, correct_cases, cases
  
    def create_adversarial_data(self, attack_type, attack_col, attack_strategy, classifier_name, force_attack, traintest):
        """
        This function creates adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        but they should have a different case ID.
        We save the adversarial dataset so we do not have to rerun it each time
        """
        save_string_list = [str(attack_type), str(attack_col), str(attack_strategy)]
        save_name = f"{classifier_name}_train_{save_string_list}" # the name we use to save the model 

        print('save dataset name', save_name)
        # try to load the adversarial dataset from disk, if that fails perform the attacks instead.
        
        adversarial_prefixes = None
        if not force_attack:
            adversarial_prefixes = load_trained_dataset(
                save_name=save_name, data_name=self.dataset_name
            )

        
        # if model loading failed or force_train flag set to true.
        if adversarial_prefixes is None or force_attack:
            # get preprocessed data
            if self.algorithm == 'ML':
                dt_prefixes_correct,_ , caseIDs = self.correctly_predicted_prefixes(self.train)

            elif self.algorithm == 'DL':
                dt_prefixes_correct, y_correct, caseIDs, cases = self.correctly_predicted_prefixes(self.train_prefixes)

            adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, attack_strategy, caseIDs, self.train_prefixes, dt_prefixes_correct, traintest, self.algorithm)
       
            if traintest == 'test':
                caseIDs_incorrect = list(set(caseIDs)-set(caseIDs))
                incorrect_dt_prefixes = self.test_prefixes[self.test_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
                adversarial_prefixes = pd.concat([incorrect_dt_prefixes, adversarial_prefixes])

            save_dataset(
                    dataset= adversarial_prefixes,
                    save_name=save_name,
                    data_name=self.dataset_name
                )
        return adversarial_prefixes
        
    def adversarial_data_and_training(self, attack_type, attack_col, attack_strategy, model_type, cls_method, force_attack, force_train):
        """This function performs the attack. It first creates the adversarial data and then trains the model on this data."""
        print('attack', attack_type, attack_col, attack_strategy)
        # Adversarial - Training Data
        if attack_strategy == 'adversarial':
            adversarial_prefixes_train = self.create_adversarial_data(attack_type, attack_col, attack_strategy, cls_method, force_attack, traintest='train', model_type=model_type)

            if adversarial_prefixes_train.shape[0] != 0:
            # Here, you need to save the adversarially created prefixes to a file and you need to check whether this file exists before you start your attacking loop
                adversarial_train_prefixes = pd.concat([self.train_prefixes, adversarial_prefixes_train])
            
            else:
                adversarial_train_prefixes = self.train_prefixes

            self.model.train(
            adversarial_train_prefixes, 
            self.test_prefixes, 
            attack = [attack_type, attack_col, attack_strategy],
            force_train=force_train)
            
            # Adversarial - Testing Data
            adversarial_prefixes_test = self.create_adversarial_data(attack_type, attack_col, attack_strategy, cls_method, force_attack, traintest='test', model_type=model_type)

            #test_y = self.dataset_manager.get_label_numeric(adversarial_prefixes_test)
            #dt_test_named = self.dataset_manager.transform_data_test(self.feature_combiner, self.scaler, adversarial_prefixes_test)

        # Save Adversarial Results to Dictionary
        #results_cls['adv_cls_' + attack_type + '_' + attack_col] = cls_adv
        #results_test['adv_test_' + attack_type + '_' + attack_col] = (dt_test_named, test_y)
        
        if attack_strategy == 'manifold':
            # On-Manifold - Training Data
            manifold_prefixes_train = self.create_adversarial_data(attack_type, attack_col, attack_strategy, cls_method, force_attack, traintest='train', model_type=model_type)

            if manifold_prefixes_train.shape[0] != 0:
            # Here, you need to save the adversarially created prefixes to a file and you need to check whether this file exists before you start your attacking loop
                manifold_train_prefixes = pd.concat([self.train_prefixes, manifold_prefixes_train])
            
            else:
                manifold_train_prefixes = self.train_prefixes

            self.model.train(
            manifold_train_prefixes, 
            self.test_prefixes,
            attack = [attack_type, attack_col, attack_strategy],
            force_train=force_train)


            # On-Manifold - Testing Data
            manifold_prefixes_test = self.create_adversarial_data(attack_type, attack_col, attack_strategy, cls_method, force_attack, traintest='test', model_type=model_type)

            #test_y_manifold = self.dataset_manager.get_label_numeric(manifold_prefixes_test)
            #dt_test_named_manifold = self.dataset_manager.transform_data_test(self.feature_combiner, self.scaler, manifold_prefixes_test)

        # Save On-Manifold Results to Dictionary
        #results_cls['manifold_cls_' + attack_type + '_' + attack_col] = cls_manifold
        #results_test['manifold_test_' + attack_type + '_' + attack_col] = (dt_test_named_manifold, test_y_manifold)
        print('saved')
        
    def regular_adversarial_examples(self, attack_type, attack_col, cls_method, force_attack, force_train):
        
        attack_strategy = 'regular'
        
        # first find the correctly predicted train data
        dt_correct, y_correct, correct_cases, cases = self.correctly_predicted_prefixes(self.train, self.train_y, self.train_cases)
        # then, we generate the adversarial examples. We generate 100 adversarial examples for each correctly predicted trace
        # then we need to calculate whcih one is the closest to the original trace in latent space
        
        # Find the index where the labels change from 0 to 1
        index_of_first_deviant_trace = y_correct.index(1)

        # Split the traces based on the index
        traces_regular = dt_correct[:index_of_first_deviant_trace]
        traces_deviant = dt_correct[index_of_first_deviant_trace:]
            
        # Generate adversarial prefixes for regular traces
        total_adversarial_prefixes_regular = []
        for trace in traces_regular:
            trace = trace[:-1]  # Remove EoS token
            # Perform the desired adversarial attack on the trace
            if attack_type == 'last_event':
                adversarial_prefixes = self.permute_last_event(trace)
            elif attack_type == 'all_event':
                adversarial_prefixes = self.permute_all_event(trace, 10)
            elif attack_type == 'random_event':
                adversarial_prefixes = self.permute_random_event(trace, 10)
                
            prefixes = [trace] + adversarial_prefixes 
            _, prefixes = self.dataset_manager.ohe_cases(prefixes)
            mu_regular, log_var = self.project_on_latent_space(prefixes, label=0)
            
            # Extract the first latent sample
            first_latent_sample = mu_regular[0]

            # Calculate pairwise distances with the first latent sample
            pairwise_distances = [euclidean(first_latent_sample.flatten(), sample.flatten()) for sample in mu_regular[1:]]

            # Find the index of the lowest value
            min_index = pairwise_distances.index(min(pairwise_distances))

            adversarial_prefix = adversarial_prefixes[min_index] # The first element is the original trace
            adversarial_prefix += [self.vocab_size-1] # Add the EoS token
            total_adversarial_prefixes_regular.append(adversarial_prefix)
            

        # Generate adversarial prefixes for deviant traces
        total_adversarial_prefixes_deviant = []
        for trace in traces_deviant:
            trace = trace[:-1]  # Remove EoS token
            # Perform the desired adversarial attack on the trace
            if attack_type == 'last_event':
                adversarial_prefixes = self.permute_last_event(trace)
            elif attack_type == 'all_event':
                adversarial_prefixes = self.permute_all_event(trace, 10)
            elif attack_type == 'random_event':
                adversarial_prefixes = self.permute_random_event(trace, 10)
            
            prefixes = [trace] + adversarial_prefixes 
            _, prefixes = self.dataset_manager.ohe_cases(prefixes)
            mu_deviant, _ = self.project_on_latent_space(prefixes, label=0)
            
            # Extract the first latent sample
            first_latent_sample = mu_deviant[0]

            # Calculate pairwise distances with the first latent sample
            pairwise_distances = [euclidean(first_latent_sample.flatten(), sample.flatten()) for sample in mu_deviant[1:]]

            # Find the index of the lowest value
            min_index = pairwise_distances.index(min(pairwise_distances))

            adversarial_prefix = adversarial_prefixes[min_index] # The first element is the original trace
            adversarial_prefix += [self.vocab_size-1] # Add the EoS token
            total_adversarial_prefixes_deviant.append(adversarial_prefix)

        # Combine the adversarial prefixes
        total_adversarial_prefixes = total_adversarial_prefixes_regular + total_adversarial_prefixes_deviant
        print('original cases', len(dt_correct))
        print('adversarial generated cases', len(total_adversarial_prefixes))
        return total_adversarial_prefixes

    def option_2(self,attack_type, attack_col, attack_strategy, model_type, cls_method, force_attack, force_train):
        # first find the correctly predicted prefixes
        dt_prefixes_correct, y_correct, caseIDs, cases = self.correctly_predicted_prefixes(self.train_prefixes)

        dt_prefixes_correct_regular = dt_prefixes_correct[dt_prefixes_correct.label == 1]
        dt_prefixes_correct_deviant = dt_prefixes_correct[dt_prefixes_correct.label == 0]
        correct_list = [dt_prefixes_correct_regular]
        # Dictionary to store the minimal adversarial distance for each trace
        trace_distances = {}

       
        if attack_type == 'all_event':
            dt_prefixes_adv = self.permute_all_event(dt_prefixes_correct_deviant, attack_col, 10)
        elif attack_type == 'last_event':
            dt_prefixes_adv = self.permute_last_event(dt_prefixes_correct_deviant, attack_col, 10)

        on_manifold_adversarial_deviant = self.project_on_manifold(dt_prefixes_adv, 'deviant')
        print('case original', dt_prefixes_correct_deviant['Case ID'].nunique())
        print('cases on manifold', on_manifold_adversarial_deviant['Case ID'].nunique())
        
        original_cases = dt_prefixes_correct_deviant['original_case_id'].unique()
        for case in original_cases:
            on_manifold_adversarial_deviant_case = on_manifold_adversarial_deviant[on_manifold_adversarial_deviant['original_case_id'] == case]
            
        # now we have a dataset that is 100x the size of the original dataset
    
        # for each correctly predicted trace
         # create 100 adversarial prefixes
            # for each adversarial prefix
                # project on the manifold
                # calculate the latent space representation
                # calculate the euclidean distance between the original and adversarial latent space representation
                # keep the attack with the minimal adversarial distance

    def is_tensor_in_list(self, target_tensor, list_of_tensors):
        """
        Check if a target tensor is in a list of tensors.

        Args:
        - target_tensor (torch.Tensor): The tensor to search for.
        - list_of_tensors (list): List of tensors to search in.

        Returns:
        - bool: True if the target tensor is found in the list, False otherwise.
        """
        return any(target_tensor.equal(t) for t in list_of_tensors)
    
    def visualize_tsne(self, means):
        # Perform dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_space_tsne = tsne.fit_transform(means)

        # Perform dimensionality reduction using PCA
        pca = PCA(n_components=2, random_state=42)
        latent_space_pca = pca.fit_transform(means)

        # Visualize the latent space representations using t-SNE
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(latent_space_tsne[:, 0], latent_space_tsne[:, 1], alpha=0.5)
        plt.title('Latent Space Representations (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)

        # Visualize the latent space representations using PCA
        plt.subplot(1, 2, 2)
        plt.scatter(latent_space_pca[:, 0], latent_space_pca[:, 1], alpha=0.5)
        plt.title('Latent Space Representations (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def option_3(self, cls_method, force_attack, force_train, samples_per_trace):
        """
        This function is used to generate adversarial examples for the correct traces.
        The function generates 100 adversarial examples per trace and then projects these examples to the latent space.
        """

        dt_prefixes_correct, y_correct, caseIDs, cases = self.correctly_predicted_prefixes(self.train_prefixes) # first find the correctly predicted prefixes
        
        # take the prefixes per label
        #dt_prefixes_correct_regular = dt_prefixes_correct[dt_prefixes_correct.label == 0]
        dt_prefixes_correct_regular = dt_prefixes_correct[dt_prefixes_correct.label == 0]
        
        original_samples = [] # Initialize a list to store the original samples
        with torch.no_grad():
            input_tensor, mu, logvar = self.get_latent_representations(dt_prefixes_correct_regular, "regular") # the latent representations for the correctly predicted traces
            original_samples.append(torch.argmax(input_tensor, dim=2)) # Append the trace to the list of original_samples samples
        # mu has shape (batch_size, sequence_length, latent_dim). This is the mean of the latent space
        # logvar has shape (batch_size, sequence_length, latent_dim). This is the log variance of the latent space

        means_2d = mu.reshape(mu.shape[0], -1)

        self.visualize_tsne(means_2d)
        
        df_samples = pd.DataFrame(columns=['Input', 'Generated']) # Initialize an empty DataFrame to store the generated samples

        # Iterate over each trace in the input tensor
        for trace_nr, input_trace in enumerate(input_tensor):

            unique_samples  = [] # Initialize a list to store unique generated samples for this trace   
            # Unsqueezing the tensor to add a batch dimension
            input_trace = input_trace.unsqueeze(0)
            #print('the original trace shape is', torch.argmax(input_trace, dim=2))
           
            #print('original input trace', self.model.predict_proba(input_trace, 'inference'))
            for i in range(samples_per_trace):
                mu_tensor = mu[trace_nr,:,:]
                logvar_tensor = logvar[trace_nr,:,:]
                std_tensor = torch.exp(0.5 * logvar_tensor).to(self.device)
                epsilon = torch.randn_like(std_tensor)  # we are going to sample from a normal distribution with mean 0 and variance 1
                z = mu_tensor + epsilon * std_tensor # this is the reparametrization trick, where we multiply this noise with the standard deviation and add the mean
                
                with torch.no_grad():
                    self.vae_regular.eval()
                    output_act = self.vae_regular.decode(z) # Generate output sample from the latent sample using the VAE decoder. THe shape is [timesteps, vocab_size]
                
                output_act = output_act.unsqueeze(0)
                output_act = self.vae_regular.mask_out_tensor(output_act)
                output_act_argmaxed = torch.argmax(output_act, dim=2) # we argmax the output to get the most likely event per timestep

                if not self.is_tensor_in_list(output_act_argmaxed, unique_samples) and not self.is_tensor_in_list(output_act_argmaxed, original_samples): # If the generated sample is unique
                    print('new sample', self.model.predict_proba(output_act, 'inference'))
                    unique_samples.append(output_act_argmaxed)  # Append the unique generated sample to the list of unique samples
                    print('new sample probability', output_act_argmaxed)
                    
                # Append the unique generated samples along with the corresponding input tensor to the DataFrame
                for sample in unique_samples:
                    df_samples = df_samples._append({'Input': torch.argmax(input_trace, dim=2).detach().numpy(), 'Generated': sample.detach().numpy()}, ignore_index=True)

        # Print the DataFrame containing the input and generated samples
        print(df_samples)
        df_samples.to_csv('G:\My Drive\CurrentWork\Robust PPM\generated_samples.csv')

        # next we are going to save all the generated adversarial examples and then check whether they are succesful adversarial examples or not

    def option_4(self, hyperparams):
        """
        This function is the gradient-based approach to generate adversarial examples for the correct traces.
        """
        
        # hyperparameters for the gradient-based adversarial attacks
        """
        * "lambda": float, default: 0.5
                Decides how similar the adv is to the factual
        * "optimizer": {"adam", "rmsprop"}
                Optimizer for generation of adversarial examples.
        * "lr": float, default: 0.1
                Learning rate for Revise.
        * "max_iter": int, default: 1000
                Number of iterations for Revise optimization.
        * "target_class": int, default: [1]
                List of one-hot-encoded target class.
        * "vocab_size": int
                Size of the vocabulary.
        * "max_prefix_length": int
                Maximum prefix length.
        * "threshold": float, default: 0.5
                Threshold for stopping criterion.
        """

        dt_prefixes_correct, y_correct, caseIDs, cases = self.correctly_predicted_prefixes(self.train_prefixes) # first find the correctly predicted prefixes
        ans_regular, ans_deviant, label_regular, label_deviant, cases_regular, cases_deviant  = self.dataset_manager.groupby_caseID(dt_prefixes_correct)

        padded_activity, one_hot_matrix = self.dataset_manager.ohe_cases(ans_deviant)

        adversarial_example_alg = AdversarialExamples(self.model, one_hot_matrix, self.vae_regular, hyperparams)

        target = [0] 
        #df_cfs = adversarial_example_alg.get_adversarial_examples(one_hot_matrix, target)

        # Create an empty DataFrame to store the results
        all_results = pd.DataFrame()
        
        # now run all implemented measurements and concatenate the results
        print('The results:')
        for selected_batch in one_hot_matrix:
            selected_batch = selected_batch.unsqueeze(0)
            benchmark = Benchmark(self.dataset_manager, self.model, adversarial_example_alg, selected_batch, target)
            # now you can decide if you want to run all measurements
            # or just specific ones.
            evaluation_measures = [
                    evaluation_catalog.YNN(self.dataset_manager, benchmark.mlmodel, one_hot_matrix, {"y": 5, "cf_label": target[0], "NN": 5}),
                    evaluation_catalog.Distance(self.dataset_manager),
                ]

            # now run all implemented measurements and concatenate the results
            print('The results:')
            results = benchmark.run_benchmark(evaluation_measures)
            all_results = pd.concat([all_results, results], ignore_index=True)

        outfile = "results/"
        outfile += "results_" + self.dataset_name + "_" + str(hyperparams["max_iter"]) + "_" + "regular.csv"
        all_results.to_csv(outfile, index=False)
        
    def predict_proba(self, data):
        if self.algorithm == 'ML':
            data, _ = self.dataset_manager.ohe_cases(data)    
            data = self.dataset_manager.transform_data(data, traintest='test')
            pred = self.model.predict_proba(data)[:, -1]

        elif self.algorithm == 'DL':
            _, data = self.dataset_manager.ohe_cases(data)
            pred = self.model.predict_proba(data, 'inference').cpu().detach().numpy()

        return pred
     
    def create_adversarial_prefixes(self, attack_type, attack_col, attack_strategy, caseIDs, dt_prefixes, dt_prefixes_correct, traintest, model_type):
        
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()

        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        else:
            total_caseIDs = caseIDs
        print('amount of total case IDs', len(total_caseIDs))
        
        while len(total_adversarial_cases) < len(total_caseIDs):
            adversarial_cases = 0
            loop_count += 1
            print('loop count', loop_count)
            if loop_count < 25:
                dt_prefixes2 = dt_prefixes_correct.copy()
                # These are the adversarially attacked examples
                if attack_type == 'all_event':
                    dt_prefixes_adv = self.permute_all_event(dt_prefixes2, attack_col, num_permutations=1)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.permute_last_event(dt_prefixes2, attack_col, num_permutations=1)

                if attack_strategy == 'manifold':
                    dt_prefixes_adv, y2 = self.create_manifold_dataset(dt_prefixes_adv)

                if model_type == 'ML':
                        adversarial_cases = self.check_adversarial_cases_ML(dt_prefixes_adv, y2, caseIDs)

                elif model_type == 'DL':
                        adversarial_cases = self.check_adversarial_cases_DL(dt_prefixes_adv)
                
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    if len(total_adversarial_cases) > len(total_caseIDs):
                        total_adversarial_cases = total_adversarial_cases[0:len(total_caseIDs)]
                    created_adversarial_df = dt_prefixes_adv[dt_prefixes_adv['Case ID'].isin(adversarial_cases)].copy()
                    created_adversarial_df.loc[:, 'Case ID'] = created_adversarial_df.loc[:, 'Case ID'] + '_adv' + str(loop_count)
                    adversarial_prefixes = pd.concat([adversarial_prefixes, created_adversarial_df])
                    print('total adversarial cases after loop', loop_count, 'is:', len(total_adversarial_cases))
                else:
                    print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                    print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
            else:

                print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
                return adversarial_prefixes
        print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
        print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
        print('percentage of adversarial cases:', len(total_adversarial_cases)/len(total_caseIDs))
        return adversarial_prefixes
     
