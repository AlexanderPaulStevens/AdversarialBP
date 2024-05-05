import sklearn
import torch
import torch.nn as nn
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)
from util.settings import training_setting
clip = training_setting["clip"]

import torch

"""
The training parameters for the Long Short-Term Memory (LSTM) neural network
The first two values of the hidden_sizes are the hidden dimensions, the last value is the lstm size. 
The hyperparameters used in Camargo et al.:
    - dropout : 0.2
    - batch size: 32  # Usually 32/64/128/256
    - epochs: 200
    - hidden_sizes: 50/100
    - optimizer Adam (lr=0.001), SGD (lr=0.01), Adagrad (lr=0.01)
"""
class LSTM(nn.Module):
    """
    This class is used to define the architecture of the LSTM model.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.

        dropout : float 
            Dropout rate for the LSTM layer. 

        hidden_sizes : list
            List of hidden sizes for the LSTM layers.
    Returns
    -------
    probability : float
    """
    def __init__(self, training_params):
        super(LSTM, self).__init__()
        
        self.hidden_dim1 = training_params['hidden_sizes'][0]
        self.hidden_dim2 = training_params['hidden_sizes'][1]
        self.vocab_size = training_params['vocab_size'][0]
        self.lstm_dropout = training_params['dropout']
        self.lstm_size= training_params['hidden_sizes'][2]
        self.lstm1 = nn.LSTM(input_size = self.vocab_size, hidden_size = self.hidden_dim1, dropout = self.lstm_dropout, num_layers=self.lstm_size, bidirectional=True, batch_first=True)
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

def _training_torch(
        model,
        train_loader,
        test_loader,
        optimizer_name,
        learning_rate,
        weight_decay,
        epochs):
        
        loaders = {"train": train_loader, "inference": test_loader}

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
            for phase in ["train", "inference"]:

                running_loss = 0.0
                predictions_all = []
                pred_probs_all = []
                labels_all = []
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                 # Train for all the batches
                for i, (inputs, labels) in enumerate(train_loader):
                    loss = 0 # Reset recon_loss to zero at the beginning of each batch
                    labels = labels.to(device).type(torch.int64)
                    #labels = torch.nn.functional.one_hot(labels, num_classes=2)
                    labels = labels.unsqueeze(-1) #you need this if your LSTM model only predicts 1 probability
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):     
                        outputs = model(inputs, mode = phase)
                        loss = criterion(outputs.float(), labels.float())
                        output_detached = outputs.clone().detach().numpy()
                        pred_probs_all.extend(output_detached)
                        labels_all.extend(labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    # loss.item() extrats the scalar value of the loss function at a particular step. 
                    # inputs.size(0) retrieves the size of the input batch
                    running_loss += loss.item() * inputs[0].size(0)
                epoch_loss = running_loss / len(loaders[phase].dataset)
                epoch_auc = sklearn.metrics.roc_auc_score(labels_all, pred_probs_all)
                print("{} Running loss: {:.4f}  epoch AUC: {:.4f}".format(phase, epoch_loss, epoch_auc))