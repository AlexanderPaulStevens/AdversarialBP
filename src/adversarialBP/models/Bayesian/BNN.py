# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:59:47 2023

@author: u0138175
"""
import sklearn
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)

from typing import Tuple

from util.load_model import load_trained_model, save_model
from util.BNN_uncertainty import aleatoric_uncertainty, epistemic_uncertainty, uncertainty_avg, score_avg
from util.BNN_uncertainty import MCSingleMixin
from util.settings import global_setting, training_setting

import matplotlib.pyplot as plt

class Model(nn.Module, MCSingleMixin):
    def __init__(self, training_params):
        super(Model, self).__init__()

        """This class is used to define the architecture of the LSTM model.

        Parameters
        ----------
        dropout : float
            Dropout rate for the LSTM layer.

        embed_size : int
            Size of the embedding layer.

        lstm_size : int 
            Number of layers in the LSTM layer.

        Returns
        -------
        probability : float
        """
        self.training_params = training_params
        self.optimizer_name = self.training_params["optimizer_name"]
        self.learning_rate = self.training_params["learning_rate"]
        self.weight_decay = self.training_params["weight_decay"]
        self.epochs = self.training_params["epochs"]
        self.vocab_size = self.training_params["vocab_size"]
        self.embed_size = self.training_params["hidden_sizes"][0]
        self.hidden_size = self.training_params["hidden_sizes"][1]
        self.nr_lstm_layers = self.training_params["hidden_sizes"][2]
        self.weight_regularizer = self.training_params["weight_regularizer"]
        self.dropout_regularizer = self.training_params["dropout_regularizer"]
        self.hs = self.training_params["hs"]
        self.dropout = self.training_params["dropout"]
        self.concrete = self.training_params["concrete"]
        self.p_fix = self.training_params["p_fix"]
        self.Bayes = self.training_params["Bayes"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # the stochastic LSTM model wants a list of tuples containing the vocab size and the embedding size
        self.lstm_bnn = StochasticLSTM([(self.vocab_size[0], self.embed_size)], self.hidden_size, self.weight_regularizer, self.dropout_regularizer,
                                        self.hs, self.dropout, self.concrete, self.p_fix, Bayes = self.Bayes, nr_lstm_layers=self.nr_lstm_layers)
        self.final_output = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def regularizer(self):        
        # Weight and bias regularizer
        weight_sum, bias_sum, dropout_reg = self.lstm_bnn.regularizer()
        
        return weight_sum + bias_sum + dropout_reg
    
    def get_output_shape(self, x: torch.Tensor):
        B = x.shape[0]
        return ([B,1])

    def forward(self, x_act):
        l1_out, (h,c) = self.lstm_bnn(x_act)
        l1_out = l1_out.transpose(0, 1)
        output = l1_out[:,-1,:]
        output = self.final_output(output)
        output = torch.sigmoid(output)
        return output

    """In this class, we make the attributes fixed:
        - embed size
        - optimizer name
        - batch size
        - learning rate
        - dropout
        - vocab size list
        - lstm size 
        - max prefix length

    Because we want to be able to run the function with these parameters for multiple runs/datasets.
    """

class StochasticLSTMCell(nn.Module):
    """
    This class defines the individual LSTM cell with stochastic behavior (the stochastic LSTM cell).
    It includes parameters and methods for dropout handling, regularization, weight initialization, and forward pass computation.
    
    
    """
    def __init__(self, input_size: int, hidden_size: int, p_fix=0.01, concrete=True,
                 weight_regularizer=.1, dropout_regularizer=.1, Bayes=True):

        """
        This class defines a stochastic LSTM cell with concrete dropout."""
        """
        Parameters
        ----------
        input_size: int
            number of features (after embedding layer)
        hidden_size: int
            number of nodes in LSTM layers
        p_fix: float
            dropout parameter used in case of not self.concrete
        concrete: Boolean
            dropout parameter is fixed when "False". If "True", then concrete dropout
        weight_regularizer: float
            parameter for weight regularization in reformulated ELBO
        dropout_regularizer: float
            parameter for dropout regularization in reformulated ELBO
        Bayes: Boolean
            BNN if "True", deterministic model if "False" (only sampled once for inference)
        Returns
        -------
        hn: tensor of hidden states h_t. Dimension (sequence_length x batch_size x hidden size)
        h_t: hidden states at time t. Dimension (batch size x hidden size (number of nodes in LSTM layer))
        c_t: cell states. Dimension (batch size x hidden size (number of nodes in LSTM layer))
        '''
        """

        super(StochasticLSTMCell, self).__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concrete = concrete
        self.wr = weight_regularizer
        self.dr = dropout_regularizer
        self.Bayes = Bayes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if concrete:
            self.p_logit = nn.Parameter(torch.empty(1).normal_()) # logit of dropout rate
        else:
            if np.isnan(p_fix):
                p_fix = .5
            self.p_logit = torch.full([1], p_fix)

        self.Wi = nn.Linear(self.input_size, self.hidden_size) # input gate
        self.Wf = nn.Linear(self.input_size, self.hidden_size) # forget gate
        self.Wo = nn.Linear(self.input_size, self.hidden_size) # output gate
        self.Wg = nn.Linear(self.input_size, self.hidden_size) # cell state

        self.Ui = nn.Linear(self.hidden_size, self.hidden_size) # input gate
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size) # forget gate
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size) # output gate
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size) # cell state

        self.init_weights() # initialize weights

    def init_weights(self):
        """
        Xavier initialization for weights and uniform initialization for biases
        """
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt() # Xavier initialization

        self.Wi.weight.data.uniform_(-k, k).to(self.device) 
        self.Wi.bias.data.uniform_(-k, k).to(self.device)

        self.Wf.weight.data.uniform_(-k, k).to(self.device)
        self.Wf.bias.data.uniform_(-k, k).to(self.device)

        self.Wo.weight.data.uniform_(-k, k).to(self.device)
        self.Wo.bias.data.uniform_(-k, k).to(self.device)

        self.Wg.weight.data.uniform_(-k, k).to(self.device)
        self.Wg.bias.data.uniform_(-k, k).to(self.device)

        self.Ui.weight.data.uniform_(-k, k).to(self.device)
        self.Ui.bias.data.uniform_(-k, k).to(self.device)

        self.Uf.weight.data.uniform_(-k, k).to(self.device)
        self.Uf.bias.data.uniform_(-k, k).to(self.device)

        self.Uo.weight.data.uniform_(-k, k).to(self.device)
        self.Uo.bias.data.uniform_(-k, k).to(self.device)

        self.Ug.weight.data.uniform_(-k, k).to(self.device)
        self.Ug.bias.data.uniform_(-k, k).to(self.device)

    def _sample_mask(self, batch_size, stop_dropout):
        """
        This function generates dropout masks for inputs and hidden states.
        
        ARGUMENTS:
        batch_size: batch size
        stop_dropout: if "True" prevents dropout during inference for deterministic models

        OUTPUTS:
        zx: dropout masks for inputs. Tensor (GATES x batch_size x input size (after embedding))
        zh: dropout masks for hiddens states. Tensor (GATES x batch_size x number hidden states)
        """

        if not self.concrete:
            p = self.p_logit.to(self.device)
        else:
            p = torch.sigmoid(self.p_logit).to(self.device)
        GATES = 4
        eps = torch.tensor(1e-7)
        t = 1e-1

        if not stop_dropout:
            ux = torch.rand(GATES, batch_size, self.input_size).to(self.device)
            uh = torch.rand(GATES, batch_size, self.hidden_size).to(self.device)

            if self.input_size == 1:
                zx = (1 - torch.sigmoid((torch.log(eps) - torch.log(1 + eps)
                                         + torch.log(ux + eps) - torch.log(1 - ux + eps))
                                        / t))
            else:
                zx = (1 - torch.sigmoid((torch.log(p + eps) - torch.log(1 - p + eps)
                                         + torch.log(ux + eps) - torch.log(1 - ux + eps))
                                        / t)) / (1 - p)
            zh = (1 - torch.sigmoid((torch.log(p + eps) - torch.log(1 - p + eps)
                                     + torch.log(uh + eps) - torch.log(1 - uh + eps))
                                    / t)) / (1 - p)
        else:
            zx = torch.ones(GATES, batch_size, self.input_size).to(self.device)
            zh = torch.ones(GATES, batch_size, self.input_size).to(self.device)

        return zx, zh

    
    def regularizer(self):
        """
        This method computes regularization terms for weight and bias parameters and dropout regularization in reformulated ELBO.
        
        
        OUTPUTS:
        self.wr * weight_sum: weight regularization in reformulated ELBO
        self.wr * bias_sum: bias regularization in reformulated ELBO
        self.dr * dropout_reg: dropout regularization in reformulated ELBO
        """

        if not self.concrete:
            p = self.p_logit.to(self.device)
        else:
            p = torch.sigmoid(self.p_logit)

        if self.Bayes:
            weight_sum = torch.tensor([
                torch.sum(params ** 2) for name, params in self.named_parameters() if name.endswith("weight")
            ]).sum() / (1. - p)

            bias_sum = torch.tensor([
                torch.sum(params ** 2) for name, params in self.named_parameters() if name.endswith("bias")
            ]).sum()

            if not self.concrete:
                dropout_reg = torch.zeros(1)
            else:
                dropout_reg = self.input_size * (p * torch.log(p) + (1 - p) * torch.log(1 - p))
            return self.wr * weight_sum, self.wr * bias_sum, self.dr * dropout_reg
        else:
            print('are we here?')
            return torch.zeros(1)


    def forward(self, input: Tensor, stop_dropout=False) -> Tuple[
        Tensor, Tuple[Tensor, Tensor]]:
        """
        This method computes the forward pass of the LSTM cell with stochastic behavior. 
        Dropout is only applied during training.

        PARAMETERS:
        input: Tensor (sequence length x batch size x input size(after embedding) )
        stop_dropout: if "True" prevents dropout during inference for deterministic models

        OUTPUTS:
        hn: tensor of hidden states h_t. Dimension (sequence_length x batch_size x hidden size)
        h_t: hidden states at time t. Dimension (batch size x hidden size (number of nodes in LSTM layer)
        c_t: cell states. Dimension (batch size x hidden size (number of nodes in LSTM layer)
        """

        seq_len, batch_size = input.shape[0:2]

        h_t = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype).to(self.device)

        hn = torch.empty(seq_len, batch_size, self.hidden_size, dtype=input.dtype)

        zx, zh = self._sample_mask(batch_size, stop_dropout)

        for t in range(seq_len):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t
            hn = hn.to(self.device)

        return hn, (h_t, c_t)

class StochasticLSTM(nn.Module):
    """This class defines the stacked LSTM layers with dropout and optionally heteroscedastic output uncertainty.
    It includes parameters and methods for regularization, weight initialization, and forward pass computation.
    """

    def __init__(self, embed_size, hidden_size, weight_regularizer, dropout_regularizer, 
                 hs, dropout, concrete, p_fix, Bayes, nr_lstm_layers):
        '''
        ARGUMENTS:
        emb_dims: list of tuples (a, b) for the categorical variables 'activity'
                  with a: number of levels, and b: embedding dimension
        hidden_size: number of nodes in LSTM layers (the output size of the LSTM)
        weight_regularizer: parameter for weight regularization in reformulated ELBO
        dropout_regularizer: parameter for dropout regularization in reformulated ELBO
        hs: "True" if heteroscedastic, "False" if homoscedastic
        dropout: in case of deterministic model, apply dropout if "True", otherwise no dropout
        concrete: dropout parameter is fixed when "False". If "True", then concrete dropout
        p_fix: dropout parameter in case "concrete"="False"
        Bayes: BNN if "True", deterministic model if "False" (only sampled once for inference)
        nr_lstm_layers: number of LSTM layers
        '''

        super(StochasticLSTM, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.weight_regularizer =  weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.heteroscedastic = hs
        self.dropout = dropout
        self.concrete = concrete
        self.p_fix = p_fix
        self.Bayes = Bayes
        self.nr_layers = nr_lstm_layers

        self.no_of_embs = 0
        self.input_size = 0 # the number of range variables

        if self.embed_size:

            self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                             for x, y in self.embed_size])
            self.no_of_embs = sum([y for x, y in self.embed_size])

        self.input_size += self.no_of_embs

        self.first_layer = StochasticLSTMCell(self.input_size, self.hidden_size, p_fix=self.p_fix, concrete=self.concrete,
                                              weight_regularizer=self.weight_regularizer,
                                              dropout_regularizer=self.dropout_regularizer, Bayes=self.Bayes)
        self.hidden_layers = nn.ModuleList(
            [StochasticLSTMCell(self.hidden_size, self.hidden_size, self.p_fix, concrete=self.concrete,
                                weight_regularizer=self.weight_regularizer,
                                dropout_regularizer=self.dropout_regularizer,
                                Bayes=self.Bayes) for i in range(self.nr_layers - 1)])
        
        """
        self.linear1 = nn.Linear(self.hidden_size, 5)
        self.linear2_mu = nn.Linear(5, 1)

        
        if self.heteroscedastic:
            self.linear2_logvar = nn.Linear(5, 1)

        
        self.conc_drop1 = ConcreteDropout(dropout=self.dropout, concrete=self.concrete, p_fix=self.p_fix,
                                          weight_regularizer=self.weight_regularizer,
                                          dropout_regularizer=self.dropout_regularizer, conv="lin", Bayes=self.Bayes)
        self.conc_drop2_mu = ConcreteDropout(dropout=self.dropout, concrete=self.concrete, p_fix=self.p_fix,
                                            weight_regularizer=self.weight_regularizer,
                                            dropout_regularizer=self.dropout_regularizer, conv="lin", Bayes=self.Bayes)
        
        if self.heteroscedastic:
            self.conc_drop2_logvar = ConcreteDropout(dropout=self.dropout, concrete=self.concrete, p_fix=self.p_fix,
                                                weight_regularizer=self.weight_regularizer,
                                                dropout_regularizer=self.dropout_regularizer, conv="lin", Bayes=self.Bayes)
        """                                
        self.relu = nn.ReLU()


    def regularizer(self):
        total_weight_reg, total_bias_reg, total_dropout_reg = self.first_layer.regularizer()
        for l in self.hidden_layers:
            weight, bias, dropout = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
            total_dropout_reg += dropout
        return total_weight_reg, total_bias_reg, total_dropout_reg


    def forward(self, x_cat):
        """
        The forward pass is used to compute the outputs (point estimates) and the log of the uncertainty estimates.

        PARAMETERS:
        x_cat: categorical variables. Torch tensor (batch size x sequence length x number of variables)
        stop_dropout: if "True" prevents dropout during inference for deterministic models

        OUTPUTS:
        mean: outputs (point estimates). Torch tensor (batch size x number of outputs)
        log_var: log of uncertainty estimates. Torch tensor (batch size x number of outputs)
        regularization.sum(): sum of KL regularizers over all model layers
        """
        
        #regularization = torch.empty(4, device=x_range.device)

        if self.no_of_embs != 0: # if there are categorical variables
            x = [emb_layer(x_cat[:, :, i]) # for each categorical variable
                 for i, emb_layer in enumerate(self.emb_layers)] # for each embedding layer
            x = torch.cat(x, -1) # concatenate the embeddings
        x = x.transpose(0, 1) # now the tensors have shape [seq_length, batch_size, features]
        batch_size = x.shape[1]

        h_n = torch.zeros(self.nr_layers, batch_size, self.first_layer.hidden_size)
        c_n = torch.zeros(self.nr_layers, batch_size, self.first_layer.hidden_size)

        outputs, (h, c) = self.first_layer(x)
        h_n[0] = h
        c_n[0] = c

        for i, layer in enumerate(self.hidden_layers):
            outputs, (h, c) = layer(outputs, (h, c))
            h_n[i+1] = h
            c_n[i+1] = c

        return outputs, (h_n, c_n)

####################################################################################################

def _training_torch_BNN(
        model,
        train_loader,
        test_loader,
        optimizer_name,
        learning_rate,
        weight_decay,
        epochs):
        
        loaders = {"train": train_loader, "test": test_loader}

        # Use GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # define the loss
        criterion = nn.BCELoss()

        # declaring optimizer
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999), amsgrad=False)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, nesterov=False)
        elif optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        # training
        for e in range(epochs):
            print("Epoch {}/{}".format(e, epochs - 1))
            print("-" * 10)
            
            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:

                running_loss = 0.0
                predictions_all = []
                pred_probs_all = []
                labels_all = []
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                for i, (inputs, labels) in enumerate(loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device).type(torch.int64)
                    #labels = torch.nn.functional.one_hot(labels, num_classes=2)
                    labels = labels.unsqueeze(-1) #you need this if your LSTM model only predicts 1 probability
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        inputs = inputs.long()
                        inputs = inputs.unsqueeze(-1)
                        outputs = model(inputs).float()
                        loss = criterion(outputs, labels.float())
                        loss_ = loss + model.regularizer() / len(loaders[phase].dataset)
                        output_detached = outputs.clone().detach().numpy()
                        pred_probs_all.extend(output_detached)
                        labels_all.extend(labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss_.backward()
                            optimizer.step()

                    # statistics
                    # loss.item() extrats the scalar value of the loss function at a particular step. 
                    # inputs.size(0) retrieves the size of the input batch
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(loaders[phase].dataset)
                epoch_auc = sklearn.metrics.roc_auc_score(labels_all, pred_probs_all)
                print("{} Running loss: {:.4f}  epoch AUC: {:.4f}".format(phase, epoch_loss, epoch_auc))

