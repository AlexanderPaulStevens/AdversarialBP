from util.DatasetManager import SequenceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util.load_model import load_trained_model, save_model
import sklearn
import torch
import torch.nn as nn 
from sklearn.metrics import roc_auc_score
from util.settings import global_setting
from adversarialBP.models.LSTM import LSTM as LSTM, _training_torch
from adversarialBP.models.GAN import LSTMGenerator, LSTMDiscriminator, _training_torch_GAN
from adversarialBP.models.Bayesian.BNN import Model as BNN
from adversarialBP.models.Bayesian.BNN import _training_torch_BNN

# Define an LSTM model
class Model(nn.Module):      
    def __init__(self, dataset_name, dataset_manager, cls_method, **kwargs):
        super(Model, self).__init__()
        # Common parameters for all algorithms
        common_training_params = {"optimizer_name": None, "learning_rate": None, "weight_decay": None, "epochs": None, "batch_size": None, "embed_size": None, "lstm_factor": None, "dropout": None}
        # Update common parameters with specific parameters
        common_training_params.update(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.training_params = common_training_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = global_setting['models']
        self.batch_size = 1
        self.cls_method = cls_method

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
        
        train_dataset = SequenceDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = SequenceDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.cls_method == 'LSTM':
            model = LSTM(training_params = self.training_params)

            _training_torch(
                    model,
                    train_loader,
                    test_loader,
                    self.training_params["optimizer_name"],
                    self.training_params["learning_rate"], self.training_params["weight_decay"], self.training_params["epochs"])
            
        elif self.cls_method == 'GAN':
             # Initializing a generator
            model = LSTMGenerator(training_params = self.training_params)

            # Initializing a discriminator
            rnnD = LSTMDiscriminator(training_params = self.training_params)
    
            _training_torch_GAN(
                    model, rnnD,
                    train_loader,
                    test_loader,
                    self.training_params["epochs"])
            
        elif self.cls_method == 'BNN':
            model = BNN(training_params = self.training_params)
                    
            _training_torch_BNN(
                    model,
                    train_loader,
                    test_loader,
                    self.training_params["optimizer_name"],
                    self.training_params["learning_rate"], self.training_params["weight_decay"], self.training_params["epochs"])
        
        return model

    "This is the first function that is called in the file 'experiment.py'"
    def train(self, x_train, x_test, y_train, y_test, force_train, attack):
        """
        Parameters
        ----------
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        force_train: bool
            Force training, even if model already exists in cache.
        Returns
        -------

        """
                
        filtered_params = {key: value for key, value in self.training_params.items() if value is not None}
        save_string_list = list(filtered_params.values()) + attack
        save_name = f"{self.cls_method}_layers_{save_string_list}" # the name we use to save the model 

        # try to load the model from disk, if that fails train the model instead.
        self._model = None
        if not force_train:
            self._model = load_trained_model(
                save_name=save_name, data_name=self.dataset_name
            )

            # sanity check to see if loaded model accuracy makes sense
            if self._model is not None:
                self._test_accuracy(x_test, y_test)

        # if model loading failed or force_train flag set to true.
        if self._model is None or force_train:
            # get preprocessed data

            self._model = self.train_model(x_train, x_test, y_train, y_test)
            
            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.dataset_name)

    def _test_accuracy(self, x_test, y_test):
        # get preprocessed data
        pred = self.predict_proba(x_test, 'inference')
        #print('check probability', pred)
        prediction = [0 if prob < 0.5 else 1 for prob in pred]

        print('set of prediction and test set values', set(prediction), set(y_test))
        print(f"test AUC for model: {sklearn.metrics.roc_auc_score(y_test, pred.detach().numpy())}")
            
        print(f"test accuracy for model: {sklearn.metrics.roc_auc_score(y_test, prediction)}")

    def predict_proba(self, x, mode):
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """
        
        self._model = self._model.to(self.device) # Keep model and input on the same device
        self._model.eval()
        if self.cls_method == 'BNN':
            output = self._model(x)
        else:
            output = self._model(x, mode =mode)
        return output      
    
####################################################################################################