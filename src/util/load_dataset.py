import os
from typing import Optional
import pandas as pd
from util.settings import global_setting

def load_trained_dataset(
    save_name: str,
    data_name: str,
    datasets_home: Optional[str] = None,
):
    """
    Try to load the adversarial dataset from disk, else return None.

    Parameters
    ----------
    save_name: str
        The filename which is used for the saved dataset.
    data_name: str
        The subfolder which the dataset is saved in, corresponding to the dataset.
    datasets_home : string, optional
        The directory in which to cache data; see :func:`get_datasets_home`.

    Returns
    -------
    None if not able to load dataset, else returns loaded dataset.
    """

    ext = "csv"

    # save location
    cache_path = os.path.join(
        get_datasets_home(datasets_home), data_name, f"{save_name}.{ext}"
    )

    if os.path.exists(cache_path):
        # load the dataset
        dataset = pd.read_csv(cache_path)
        
        print(f"Loaded adversarial dataset from {cache_path}")
        return dataset


def save_dataset(
    dataset,
    save_name: str,
    data_name: str,
    datasets_home: Optional[str] = None,
):
    """
    Save an adversarial dataset to disk.

    Parameters
    ----------
    dataset: adversarial dataset
        dataset that we want to save to disk.
    save_name: str
        The filename which is used for the saved dataset.
    data_name: str
        The subfolder which the dataset is saved in, corresponding to the dataset.
    datasets_home : string, optional
        The directory in which to cache data; see :func:`get_datasets_home`.        

    Returns
    -------
    None
    """
    # set dataset extension

    ext = "csv"

    # save location
    cache_path = os.path.join(
        get_datasets_home(datasets_home), data_name, f"{save_name}.{ext}"
    )
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # save the dataset
    dataset.to_csv(cache_path)


def get_datasets_home(datasets_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is then used by :func:`load_dataset`.
    """

    if datasets_home is None:
        datasets_home = os.environ.get("CF_MODELS", os.path.join(global_setting['home_directory'], "datasets", "saved"))

    datasets_home = os.path.expanduser(datasets_home)
    if not os.path.exists(datasets_home):
        os.makedirs(datasets_home)

    return datasets_home
