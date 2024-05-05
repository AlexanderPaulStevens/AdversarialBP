import os
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlretrieve

import joblib
import torch
from util.settings import global_setting
PYTORCH_EXT = "pt"
TENSORFLOW_EXT = "h5"
SKLEARN_EXT = "skjoblib"
XGBOOST_EXT = "xgjoblib"

def load_trained_model(
    save_name: str,
    data_name: str,
    models_home: Optional[str] = None,
):
    """
    Try to load a trained model from disk, else return None.

    Parameters
    ----------
    save_name: str
        The filename which is used for the saved model.
    data_name: str
        The subfolder which the model is saved in, corresponding to the dataset.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.

    Returns
    -------
    None if not able to load model, else returns loaded model.
    """
    # set model extension

    ext = PYTORCH_EXT

    # save location
    cache_path = os.path.join(
        get_models_home(models_home), data_name, f"{save_name}.{ext}"
    )

    if os.path.exists(cache_path):
        # load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('path', cache_path)
        model = torch.load(cache_path, map_location=device)
        
        print(f"Loaded model from {cache_path}")
        return model


def save_model(
    model,
    save_name: str,
    data_name: str,
    models_home: Optional[str] = None,
):
    """
    Save a model to disk.

    Parameters
    ----------
    model: classifier model
        Model that we want to save to disk.
    save_name: str
        The filename which is used for the saved model.
    data_name: str
        The subfolder which the model is saved in, corresponding to the dataset.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.

    Returns
    -------
    None
    """
    # set model extension

    ext = PYTORCH_EXT

    # save location
    cache_path = os.path.join(
        get_models_home(models_home), data_name, f"{save_name}.{ext}"
    )
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # save the model
    torch.save(model, cache_path)

def get_models_home(models_home=None):
    """Return a path to the cache directory for example models.

    This directory is then used by :func:`load_model`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.environ.get("CF_MODELS", os.path.join(global_setting['home_directory'], "adversarialBP", "models", "saved"))
        print('models home', models_home)
    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home

def get_home(models_home=None):
    """Return a path to the cache directory for trained autoencoders.

    This directory is then used by :func:`save`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.path.join(global_setting['home_directory'], "adversarialBP","autoencoder", "saved")
        
    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home
