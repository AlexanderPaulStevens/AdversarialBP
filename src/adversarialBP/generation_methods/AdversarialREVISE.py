from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from carla.data.api import Data
#from carla.models.api import MLModel
#from carla.recourse_methods.api import RecourseMethod
#from carla.recourse_methods.autoencoder import VariationalAutoencoder
#from carla.recourse_methods.processing.counterfactuals import (merge_default_parameters)

class AdversarialExamples():
    """
    Parameters
    ----------
    mlmodel : 
        Black-Box-Model
    dataset: 
        Dataset to perform on
    vae:
        Variational Autoencoder
    hyperparams : 
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_adversarial_examples:
        Generate adversarial examples for given factuals.
    _adversarial_example_optimization:
        Optimize the generation of adversarial examples.
    compute_loss:
        Compute the loss for the optimization process.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the method to initialize.
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

    def __init__(self, mlmodel, dataset, vae, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.vae = vae
        self.hinge_loss_weight = self.hyperparams["lambdas"][0] # the weight for the hinge loss 
        self.proximity_weight = self.hyperparams["lambdas"][1] # the weight for the proximity loss decides how close the adversarial example is to the original instance
        self._optimizer = self.hyperparams["optimizer"] # the optimizer used for the optimization process
        self._lr = self.hyperparams["lr"] # the learning rate for the optimization process
        self._max_iter = self.hyperparams["max_iter"] # the maximum number of iterations
        self._mlmodel = mlmodel # the black-box model   
        self.vocab_size = self.hyperparams['vocab_size'] # the size of the vocabulary
        self._max_prefix_length = self.hyperparams['max_prefix_length'] # the maximum prefix length
        self.stopping_threshold = self.hyperparams["threshold"] # the threshold for the stopping criterion
        self.beta = 0.1 # the weight for the l1 loss
        self.verbose = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

    def get_adversarial_examples(self, factuals, target):
        """
        This function is responsible for generating adversarial examples for the given factuals.
        """
        self._target_class = target
        list_cfs = self._adversarial_example_optimization(factuals) # This stores the generated adversarial examples
        #cf_df = check_counterfactuals(self._mlmodel, list_cfs, negative_label=0) # this checks whether it actually is an adversarial examples. The assumumption is that this is the case when the probability for the other label is higher than probability for the current label.
        return list_cfs
    
    def are_tensors_identical(self, cf, tensor_list):
        """
        This helper function checks whether a tensor is already in a list of tensors.
        """
        reference_tensor =cf
        for tensor in tensor_list:
            if torch.equal(reference_tensor, tensor):
                return False
        return True
    
    def add_candidate_adv_example(self, adv):
        """
        This function is responsible for adding the adversarial examples to the list of adversarial examples.
        """
        adv = adv.clone().detach()
        argmax = torch.argmax(adv.detach(), dim=2)
        argmax = argmax[0].numpy()
        softmax_adv = F.softmax(adv, dim=2)        
        candidate_adv = softmax_adv.cpu().detach().numpy()
        #one_hot_encoded = np.zeros_like(candidate_counterfactual) # Create one-hot encoded values
        # Fill the one-hot encoding matrix with the label values
        #for i in range(len(argmax)):
        #    one_hot_encoded[0, i, argmax[i]] = 1

        # Print the resulting one-hot encoding matrix
        if not any(np.array_equal(argmax, array) for array in self.candidate_advs_argmax):
                self.candidate_advs_argmax.append(argmax) #this is to check whether the argmaxed version is not already used
                
                self.candidate_advs.append(candidate_adv)
                if self.verbose:
                    print('new adv added')

    def _adversarial_example_optimization(self, torch_fact):
        """
        This function is responsible for the optimization process to generate adversarial examples.
        """
       
        test_loader = DataLoader(torch_fact, batch_size=1, shuffle=False)  # Dataloader to prepare data for optimization steps. We take a batch size of 1 because we generate adversarial examples for each trace seperately.

        list_advs = [] #this stores the adversarial examples
        
        for query_instance in test_loader:
            self.query_instance = query_instance
            print('original query:', np.argmax(self.query_instance.clone().detach().numpy(), axis=2))
            target = torch.FloatTensor(self._target_class).to(self.device) #ensure both target and input are on the same device and both are a floattensor
            z = self.vae.encode(query_instance.float())[0]  # encode the query instance
            z = z.clone().detach().requires_grad_(True)  #t we start from the mean of the latent variable

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            self.candidate_advs = []  # all possible adversarial examples
            self.candidate_advs_argmax = []
            advs_list_check = []

            for idx in range(self._max_iter):
                adv = self.vae.decode(z) 
                adv = self.vae.mask_out_tensor(adv)
                advs_list_check.append(adv[0])
                output_orig = self._mlmodel.predict_proba(query_instance, 'inference')[0]
                output = self._mlmodel.predict_proba(adv, 'inference')[0]
                z.requires_grad = True
                if self.verbose:
                    print('the current adv:', np.argmax(adv.clone().detach().numpy(), axis=2))
                loss = self.compute_loss(adv, query_instance, target)
                loss = loss.to(self.device)
                # Set the VAE model to evaluation mode before backward pass
                self.vae.eval()
                if ((self._target_class[0] == 0 and output.item() < self.stopping_threshold) or
                (self._target_class[0] == 1 and output.item() > self.stopping_threshold)):
                    self.add_candidate_adv_example(adv)  #we add the adversarial example if it actually is an adversarial example

                loss.backward() # After loss.backward(), check the gradients of z
                
                if self.verbose:
                    print('are the adv unique?', self.are_tensors_identical(adv[0],advs_list_check))
                    print('original', torch.argmax(query_instance, dim=2))
                    print('the probability for the original', output_orig)
                    print('the predicted prob adv', output.item())
                    print("Gradients of z:")
                    print(z.grad is not None)
                
                optim.step()
                optim.zero_grad()  # Clear gradients for the next iteration
                adv.detach_()

            # print out the adversarial examples
            if len(self.candidate_advs):
                print("Adversarial example found!")
                for i in self.candidate_advs:
                    ax_indices = np.argmax(i, axis=2)
                    list_advs.append(i)
            else:
                print("No Adversarial example found")
                list_advs.append(query_instance.cpu().detach().numpy())
        tensor_cfs = np.concatenate(np.array(list_advs),axis=0)
        return tensor_cfs
    
    def compute_loss(self, cf_initialize, query_instance, target):
        query_instance = query_instance.cpu()
        # Computes the first component of the loss function (a differentiable hinge loss function)
        epsilon = 1e-8  # Small epsilon to avoid division by zero
        probability = self._mlmodel.predict_proba(cf_initialize, 'inference')[0]
        #print('probability', probability)
        loss_function = nn.BCELoss()
        # classification loss
        loss1 = loss_function(probability, target)
      
        # Compute weighted distance between two vectors.
        softmax_cf = F.softmax(cf_initialize, dim=2).cpu()
        delta = abs(softmax_cf - query_instance)
        l1_loss =  torch.sum(torch.abs(delta))
        l2_loss = torch.sqrt(torch.sum(delta ** 2))
        loss2 = self.beta*l1_loss + l2_loss
        
        if self.verbose:
            print('loss1', self.hinge_loss_weight * loss1)
            print('loss2', self.proximity_weight * loss2)
            #print('loss3', self.constraint_penalty * violations)
        
        total_loss =  self.hinge_loss_weight*loss1 + self.proximity_weight * loss2

        if total_loss <0:
             print('negative loss so you should change the weights')
            
        return total_loss
        