import os
import torch
import wandb
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
logging.getLogger().setLevel(logging.INFO)
from util.load_model import get_home
from util.DatasetManager import SequenceDataset

"""
The parameters used for the encoder-decoder of Taymouri et al.
    - hidden size: 200
    - num_layers= 5
    - num_directions:1
    - dropout=0.3
    - epochs: 500
    - lr: 5e-5
"""

class LSTM_VAE(torch.nn.Module):
    def __init__(self, dataset_name, dataset_manager, label, layers, vocab_size, max_prefix_length, batch_size, epochs):
            
        """
        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            layers[0] contains the hidden size
            layers[1] contains the latent size
            layers[2] contains the number of lstm layers
        vocab_size:
            The vocabulary size
        max_prefix_length:
            The maximum length of the prefix sequences
        """   
        super(LSTM_VAE, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.embed_size = layers[0] # embedding dimension
        self.hidden_dim = layers[1] # hidden dimension
        self.latent_size = layers[2] # latent dimension
        self.lstm_factor = layers[3] # number of layers in the LSTM
        self.vocab_size = vocab_size
        self.max_prefix_length = max_prefix_length
        self.epochs = epochs
        self.label = label
        self.batch_size = batch_size

        # Encoder LSTM
        self.encoder = nn.LSTM(
                input_size=self.vocab_size,
                hidden_size= self.hidden_dim,  # Adjust this as needed
                num_layers=self.lstm_factor,  # You can increase layers if needed
                batch_first=True,
                bidirectional=False  # Set to True for bidirectional LSTM
            )

        # the ReLu layer 
        # Latent variable layers
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_size)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_size)
        # Decoder LSTM
        self.decoder = nn.LSTM(
                input_size=self.latent_size,
                hidden_size=self.hidden_dim,  # Should match encoder hidden size
                num_layers=self.lstm_factor,  # You can increase layers if needed
                batch_first=True,
                bidirectional=False  # Set to True for bidirectional LSTM
        )

        # Output layers
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
   
    def encode(self, input_x):
        input_x = input_x.to(self.device)
        input_x = input_x.float()  # [batch, sequence length, vocab]
        encoder_output, _ = self.encoder(input_x) # pass through lstm  
        mu = self.fc_mu(encoder_output) #previously encoder_output[:,-1,:]
        log_var = self.fc_var(encoder_output) # mu, logvar [batch, sequence length, latent_size]
        return mu, log_var

    def decode(self, z):
        z = z.to(self.device)
        decoder_output, _ = self.decoder(z) # z [batch, sequence length, latent_size]
        output = self.fc_out(decoder_output)
        #output = self.log_softmax(output)
        return output
  
    def _reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(self.device)
        epsilon = torch.randn_like(std).to(self.device)  # Sample from a Gaussian distribution with mean 0 and std 1
        return mu + std * epsilon
    
    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).to(self.device)
        return KLD
    
    def mask_out_tensor(self, tensor):
        # Find the index of the maximum value (EoS token) in each tensor
        tensor_max = tensor.clone().detach().to(self.device)
        _, index = torch.max(tensor_max, dim=2)
        index = index.to(self.device)
        index = index
        result_indexes = []
        for row in index:
            row = row.to(self.device)
            index2 = (row == self.vocab_size-1).nonzero(as_tuple=False).to(self.device)
            if len(index2) > 0:
                result_indexes.append(index2[0, 0].item())
            else:
                result_indexes.append(-1)   # result contains the indexes of where the value 
        one_hot_masked = tensor.clone()
        for idx in range(one_hot_masked.shape[0]):
            if result_indexes[idx] == -1:
                continue
            else:
                for j in range(result_indexes[idx]+1, one_hot_masked.shape[1]):
                    one_hot_masked[idx][j,:] = torch.tensor([0]*one_hot_masked.shape[2])

        return one_hot_masked
    
    def forward(self, input_x):
        input_x = input_x.to(self.device)
        mu, log_var = self.encode(input_x)
        z = self._reparametrization_trick(mu, log_var)
        output = self.decode(z)
        return input_x, output, mu, log_var
    
    def fit(
            self,
            x_train, y_train,
            kl_weight=0.3,
            lambda_reg=1e-6,
            epochs=5,
            lr=1e-3,
            batch_size=1,
            label=None
        ):
            """
            Fit the VAE model to the data.
            Parameters
            ----------
            xtrain: train prefixes
                The training data
            kl_weight: float
                The weight of the KL divergence in the loss
            lambda_reg: float 
                The weight of the regularization term in the loss
            epochs: int
                The number of epochs to train the model
            lr: float
                The learning rate of the optimizer
            batch_size: int                 
                The batch size of the data
            """
            train_dataset = SequenceDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=lambda_reg,
            )

            # Train the VAE with the new prior
            ELBO = np.zeros((epochs, 1))
            print("Start training of Variational Autoencoder...")
            for epoch in range(epochs):

                print('epoch', epoch)
                beta = epoch * kl_weight / epochs
                # Initialize the losses
                train_loss = 0
                train_loss_num = 0
                # Train for all the batches
                for i, (inputs, labels) in enumerate(train_loader):
    
                    recon_loss = 0 # Reset recon_loss to zero at the beginning of each batch
                    input, reconstruction, mu, log_var = self(inputs) # returns the reconstruction, mu and log_var. You can add as many inputs as you want in this forward pass
                    reconstruction  = self.mask_out_tensor(reconstruction)
                    class_indices = torch.argmax(input, dim=2).to(self.device)
                    class_indices = class_indices.view(-1).to(self.device)
                    reconstruction = reconstruction.view(-1, self.vocab_size).to(self.device)
                    nlloss = torch.nn.CrossEntropyLoss(ignore_index=0) # ignore padding index (0)
                    recon_loss = nlloss(reconstruction, class_indices).to(self.device)
                   
                    assert not torch.isnan(mu).any(), "NaN values detected in mu!"
                    assert not torch.isnan(log_var).any(), "NaN values detected in log_var!"
                    kld_loss = self.kld(mu, log_var).to(self.device)

                    loss = recon_loss + beta * kld_loss
    
                    optimizer.zero_grad() # Update the parameters
                    loss.backward() # Compute the loss

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0) #gradient clipping to avoid gradients from exploding

                    # Update the parameters
                    optimizer.step()

                    # Collect the ways
                    train_loss += loss.item()
                    train_loss_num += 1

                ELBO[epoch] = train_loss / train_loss_num
                if epoch % 10 == 0:
                    print(
                        "[Epoch: {}/{}] [objective: {:.3f}]".format(
                            epoch, epochs, ELBO[epoch, 0]
                        )
                    )

                ELBO_train = ELBO[epoch, 0].round(2)
                print("[ELBO train: " + str(ELBO_train) + "]")

            self.save(label)
            print("... finished training of Variational Autoencoder.")

            self.eval()

    def load(self, label):
            cache_path = get_home()
            cache_path = os.path.join(cache_path, label)
            load_path = os.path.join(
                cache_path,
                "{}_{}_{}_{}_{}.{}".format(self.dataset_name, self.hidden_dim, self.latent_size, self.lstm_factor, self.epochs, "pt"),
            )
            if torch.cuda.is_available() == False:
                self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(load_path))

            self.eval()

            return self

    def save(self, label):
            cache_path = get_home()
            cache_path = os.path.join(cache_path, label)
            if not os.path.exists(os.path.join(cache_path)):
                os.makedirs(os.path.join(cache_path))

            save_path = os.path.join(
                    cache_path,
                    "{}_{}_{}_{}_{}.{}".format(self.dataset_name, self.hidden_dim, self.latent_size, self.lstm_factor, self.epochs, "pt"),
                )
            torch.save(self.state_dict(), save_path)
    
class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val, learning_rate, latent_size, optimizer, batch_size):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ +'_'+ str(learning_rate) +'_'+ str(latent_size) +'_'+ str(optimizer) +'_'+ str(batch_size) + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt'+'_'+ str(learning_rate) +'_'+ str(latent_size) +'_'+ str(optimizer) +'_'+ str(batch_size) + f'-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]