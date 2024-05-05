import sys
sys.path.append("G:\My Drive\CurrentWork\Robust PPM") # Add the path to the adversarialBP package
global_setting = {
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
    "seed": 42,
    "params_dir_VAE": 'params_dir_VAE',
    "params_dir": 'params_dir',
    "params_dir_DL": 'params_dir_DL',
    "results_dir": 'results_dir',
    "manifolds": 'manifolds',
    "best_manifolds": 'best_manifolds',
    "best_LSTMs": 'best_LSTMs',
    "models": 'models',
    "n_splits": 3,
    "max_evals": 15,
    "home_directory": r'C:\Users\u0138175\OneDrive - KU Leuven\WorkDirectories\PackageGitHub\AdversialBP\src'
}

model_setting = {
    "lstm_layer": 1
}

training_setting = {
    "epochs": 100,
    "bptt": 60,
    "clip": 0.25,
    "train_losses": [],
    "test_losses": [],
    "patience": 10,
    "embed_size": 16
}

from adversarialBP.models.train_DL_model import Model
from adversarialBP.models.MLmodel import MLModel

class Training_Parameters:

    def __init__(self, dataset_name, dataset_manager, cls_method, vocab_size, force_train):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.cls_method = cls_method
        self.force_train = force_train
        self.vocab_size = vocab_size
    def get_vae_params(self):
        # define the LSTM VAE model
        # this is based on the insights obtained from WandB
        if self.dataset_name == 'sepsis_cases_2':
                VAE_embed_size = 32 # embedding dimension
                VAE_hidden_dim = 100 # hidden dimension
                VAE_latent_dim = 50 # latent dimension
                VAE_lstm_factor = 3 # number of layers in the LSTM
                VAE_lr = 0.001 # learning rate optimizer VAE
                batch_size = 1
                VAE_epochs = 100
        
        if self.dataset_name == 'production':
                VAE_embed_size = 32
                VAE_hidden_dim = 100
                VAE_latent_dim = 50
                VAE_lstm_factor = 1
                VAE_lr = 0.05
                batch_size = 1
                VAE_epochs = 10

        vae_params = {"lr": VAE_lr, "layers": [VAE_embed_size, VAE_hidden_dim, VAE_latent_dim, VAE_lstm_factor], "force_train": False , "lambda_reg": 1e-5, "batch_size": batch_size, "epochs": VAE_epochs}
        return vae_params
    
    
    def get_model(self, cls_method, dt_train, dt_test, train_y, test_y):
        if self.cls_method in ['LR', 'RF', 'XGB']:
                if self.cls_method == 'LR':
                        training_params = {"C": 5}
                elif self.cls_method == "XGB":
                        training_params = {"max_features": 1}
                elif self.cls_method == 'RF':
                        training_params = {"lr": 0.001, "subsample": 0.5, "max_depth": 4, "colsample_bytree": 0.5, "min_child_weight": 5}
                cls = MLModel(self.cls_method, self.dataset_name, self.dataset_manager, **training_params)
                model = cls.train(dt_train, dt_test, train_y, test_y, attack = ['no_attack'], force_train = self.force_train) # train the classifier (without adversarial attacks)      
        else:
                if self.cls_method == 'LSTM':
                        # we try to keep the parameters as generic as possible so we can use the same one for each model
                        LSTM_lr = 0.001
                        LSTM_hidden_dim1 = 50
                        LSTM_hidden_dim2 = 20
                        lSTM_factor = 2
                        LSTM_dropout = 0.3
                        LSTM_epoch = 100
                        training_params = {"optimizer_name": "Adam", "learning_rate": LSTM_lr, "weight_decay":1e-5, "epochs": LSTM_epoch, "vocab_size": self.vocab_size, "hidden_sizes": [LSTM_hidden_dim1, LSTM_hidden_dim2, lSTM_factor], "dropout": LSTM_dropout}
                        
                elif self.cls_method == 'GAN':
                        # we try to keep the parameters as generic as possible so we can use the same one for each model
                        GAN_lr = 0.001
                        GAN_hidden_dim1 = 50
                        GAN_hidden_dim2 = 20
                        GAN_factor = 2
                        GAN_dropout = 0.3
                        GAN_epoch = 100
                        training_params = {"lr": GAN_lr, "betas": (0.5, 0.999), "epochs": GAN_epoch, "vocab_size": self.vocab_size, "hidden_sizes": [GAN_hidden_dim1, GAN_hidden_dim2, GAN_factor], "dropout": GAN_dropout}

                elif self.cls_method == 'BNN':
                        # we try to keep the parameters as generic as possible so we can use the same one for each model
                        BNN_lr = 0.001
                        BNN_embed_size = 50
                        BNN_hidden_dim = 20
                        BNN_factor = 2
                        Bayes = True
                        training_params = {"optimizer_name": "Adam", "learning_rate": BNN_lr, "weight_decay":1e-5, "epochs": 10, "vocab_size": self.vocab_size, "hidden_sizes": [BNN_embed_size, BNN_hidden_dim, BNN_factor], 
                                        "dropout": True, "hs": False, "concrete": False, "p_fix": 0.01, "Bayes": Bayes, "weight_regularizer": 0.01, "dropout_regularizer": 0.01}
                
                        # For now, we set heteroscedasticity to False, concrete to False, because it is not implemented yet
                
                model = Model(dataset_name=self.dataset_name, dataset_manager=self.dataset_manager, cls_method= cls_method, **training_params)  
                _, x_train = self.dataset_manager.ohe_cases(dt_train)
                _, x_test = self.dataset_manager.ohe_cases(dt_test)
                model.train(x_train, x_test, train_y, test_y, attack = ['no_attack'], force_train = self.force_train) # train the classifier (without adversarial attacks)      
        return model
    
    def get_attack_params(self, vocab_size, max_prefix_length):
        threshold = 0.5
        hyperparams = {"lambdas": [2, 10], "optimizer": "Adam", "lr": 0.01, "max_iter": 100, "vocab_size": vocab_size, "max_prefix_length": max_prefix_length, "threshold": threshold}
        return hyperparams