
# Define models with the use of minibatch
from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=1000)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pprint
from torch.nn.utils.rnn import pad_sequence
import pathlib
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
device=torch.device('cuda:0')
plt.style.use('ggplot')
from util.load_model import load_trained_model, save_model
import sklearn

# Define an RNN model (The generator)
class LSTMGenerator(nn.Module):
    def __init__(self, training_params):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim1 = training_params['hidden_sizes'][0]
        self.hidden_dim2 = training_params['hidden_sizes'][1]
        self.vocab_size = training_params['vocab_size'][0]
        self.GAN_dropout = training_params['dropout']
        self.lstm_size= training_params['hidden_sizes'][2]
        self.lstm1 = nn.LSTM(input_size = self.vocab_size, hidden_size = self.hidden_dim1, dropout = self.GAN_dropout, num_layers=self.lstm_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2*self.hidden_dim1, self.hidden_dim2)
        self.fc2= nn.Linear(self.hidden_dim2, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        self.to(self.device)

    def forward(self, x_act, mode):
        # x_act = x_act.to(self.device) # when running on a GPU
        x_act = x_act.float()

        if mode == 'train' or mode=='val':
            sequence_length = (x_act[:, :, 0] != 1).sum(dim=1).cpu()[0].item() # Find the indices with the first occurrence of 1 in the first column (i.e. the sequence length)
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, sequence_length-1, :] # take the last hidden state of a non-padding token

        elif mode=='inference':
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, -1, :] # take the last hidden state

        out = self.fc1(final_hidden_state)
        output = torch.sigmoid(self.fc2(out))
        return output

    
    """In this class, we make the attributes fixed:
        - embed size
        - optimizer name
        - batch size
        - learning rate
        - dropout
        - lstm size 
        - max prefix length

    Because we want to be able to run the function with these parameters for multiple runs/datasets.
    """

# Define an RNN model (The discriminator)
class LSTMDiscriminator(nn.Module):
    def __init__(self, training_params):
        super(LSTMDiscriminator, self).__init__()
        
        self.hidden_dim1 = training_params['hidden_sizes'][0]
        self.hidden_dim2 = training_params['hidden_sizes'][1]
        self.vocab_size = training_params['vocab_size'][0]
        self.GAN_dropout = training_params['dropout']
        self.lstm_size= training_params['hidden_sizes'][2]
        self.lstm1 = nn.LSTM(input_size = self.vocab_size, hidden_size = self.hidden_dim1, dropout = self.GAN_dropout, num_layers=self.lstm_size, bidirectional=True, batch_first=True)
        self.final_output = nn.Linear(2*self.hidden_dim1 + 1, 1) # 1 for the label, 2*self.hidden_dim1 for the hidden state because of bidirectional LSTM
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_act, labels, mode):
        # x_act = x_act.to(self.device) # when running on a GPU
        x_act = x_act.float()

        if mode == 'train' or mode=='val':
            sequence_length = (x_act[:, :, 0] != 1).sum(dim=1).cpu()[0].item() # Find the indices with the first occurrence of 1 in the first column (i.e. the sequence length)
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, sequence_length-1, :] # take the last hidden state of a non-padding token

        elif mode=='inference':
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, -1, :] # take the last hidden state
            
        output = torch.cat([final_hidden_state, labels], dim=1)
        output = self.final_output(output)
        output = torch.sigmoid(output)
        return output

####################################################################################################
def _training_torch_GAN(
        rnnG, rnnD,
        train_loader,
        test_loader,
        epochs):
        '''
        @param rnnG: Generator neural network
        @param rnnD: Discriminator neural network
        @param train_loader: Training data
        @param test_loader: Testing data
        @param epoch:    The number of epochs
        '''

        
        loaders = {"train": train_loader, "inference": test_loader}

      
        # Use GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rnnG = rnnG.to(device)
        rnnD = rnnD.to(device)

        # define the loss
        criterion = nn.BCELoss()

        # declaring optimizer
        optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))

        disc_loss_tot, gen_loss_tot, gen_loss_pred = [], [], []

        # training
        for e in range(epochs): # run over the epochs
            print("Epoch {}/{}".format(e, epochs - 1))
            print("-" * 10)
            
            # Each epoch has a training and validation phase
            for phase in ["train", "inference"]: # run over the phases train and test
            
                pred_probs_all = []
                labels_all = []
                
                if phase == "train": # Set model to training mode
                    rnnD.train()
                    rnnG.train()  
                else: # Set model to evaluation mode
                    rnnD.eval()
                    rnnG.eval() 

                for i, (inputs, labels) in enumerate(train_loader): # run over the batches in the data
                    
                    inputs = inputs.to(device)
                    inputs = inputs.long()
                    labels = labels.to(device).type(torch.int64) 
                    labels = labels.unsqueeze(-1) #you need this if your LSTM model only predicts 1 probability
                       
                    # TRAINING THE DISCRIMINATOR
                    optimizerD.zero_grad() # zero the parameter gradients

                    with torch.set_grad_enabled(phase == "train"): # set the gradients to true if the phase is train
                        # Executing the Generator LSTM
                        y_pred = rnnG(inputs, mode=phase) # this is the probability of the outcome. 

                        output_detached = y_pred.clone().detach().numpy() # we need to detach the output from the computational graph and convert it to numpy
                        pred_probs_all.extend(output_detached)
                        labels_all.extend(labels)
                        # We could see this as the next event
                        # We should process the data through an LSTM and then add the predicted outcome to the prefix
                        # labels contains the y_truth, y_pred is the synthetic generated label

                        if phase == "train":  
                            discriminator_realistic_pred = rnnD(inputs, labels, mode=phase) # Training Discriminator on realistic dataset
                            disc_loss_realistic = F.binary_cross_entropy(F.sigmoid(discriminator_realistic_pred), torch.ones((inputs.shape[0], 1)), reduction='sum')
                            disc_loss_realistic.backward(retain_graph=True)
                                    
                            disriminator_synthetic_pred = rnnD(inputs, y_pred, mode=phase) # Training Discriminator on synthetic dataset
                            disc_loss_synthetic = F.binary_cross_entropy(F.sigmoid(disriminator_synthetic_pred), torch.zeros((inputs.shape[0], 1)), reduction='sum')
                            disc_loss_synthetic.backward(retain_graph=True)

                            disc_loss_tot.append(disc_loss_realistic.detach() + disc_loss_synthetic.detach())

                            optimizerD.step()

                            #if len(disc_loss_tot) % 10 == 0:
                            #    print("iter =------------------------------ i :", i, len(disc_loss_tot), " the Disc error is:",", the avg is:", np.mean(disc_loss_tot))
                            #    print('y_pred', y_pred)
                            #    print('discriminator_realistic_pred', discriminator_realistic_pred)
                            #    print('disriminator_synthetic_pred', disriminator_synthetic_pred)

                        # TRAINING THE GENERATOR
                        optimizerG.zero_grad() # zero the parameter gradients

                        # Computing the loss of prediction
                        lstm_loss_pred = F.binary_cross_entropy(y_pred.float(), labels.float(), reduction='sum')
                        gen_loss_pred.append(lstm_loss_pred.detach())
                        
                        if phase == "train": # for the discriminator
                            lstm_loss_pred.backward(retain_graph=True)

                            # Fooling the discriminator by presenting the synthetic dataset and considering the labels as the real ones
                            discriminator_synthetic_pred = rnnD(inputs, y_pred, mode=phase)
                            
                            gen_fool_dic_loss = F.binary_cross_entropy(F.sigmoid(discriminator_synthetic_pred), torch.ones((inputs.shape[0], 1)),reduction='sum')
                            gen_fool_dic_loss.backward(retain_graph=True)

                            gen_loss_tot.append(lstm_loss_pred.detach() + gen_fool_dic_loss.detach())

                            optimizerG.step()

                epoch_auc = sklearn.metrics.roc_auc_score(labels_all, pred_probs_all)
                print("epoch auc",  epoch_auc)