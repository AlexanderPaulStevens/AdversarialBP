import torch
from adversarialBP.adversarial_example_generation import AdversarialAttacks_manifold
from util.settings import Training_Parameters
from util.DatasetManager import DatasetManager
import pandas as pd
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(22)
pd.set_option('display.max_rows', 100)
import logging
logger = logging.getLogger("adversarialBP")
attack_cols = ['Activity']
import click

@click.command()
@click.option(
    "--force_attack", type=bool, help="Attack forced", required=True
)
@click.option(
    "--force_train", type=bool, help="Train forced", required=True
)
@click.option("--dataset_name", type=str, help="name used for path in dataset manager", required=True)
@click.option("--cls_method", type=str, help="classifier to use", required=True)
@click.option(
    "--attack_type",
    type=str,
    help="Type of attack",
    required=True,
)


def main(force_attack:bool,
          force_train:bool,
          dataset_name:str,
          cls_method:str,
          attack_type:str,
          ):
    """ Generates adversarial examples.
    
    Args:
        force_attack: attack forced (y/n)
        force_train: train forced (y/n)
        cls_method: string
        attack_type: string
    """
#    force_attack = False
#    force_train = False
#    dataset_name = 'sepsis_cases_2'
#    cls_method = 'BNN'
#    attack_type = 'last_event'
    # Settings: to force the adversarial attacks or the training of the models
#    train_ratio = global_setting['train_ratio']

    # START
    logger.info('Dataset:', dataset_name)
    logger.info('Classifier:', cls_method)
    logger.info('Attack columns:', attack_cols)

    dataset_manager = DatasetManager(dataset_name, cls_method)
    dt_train, dt_test, train_y, test_y, train_cases, test_cases,vocab_size, max_prefix_length = dataset_manager.create_data(cat_cols = attack_cols, prefix_generation=False)

    # define parameters
    training_params = Training_Parameters(dataset_name, dataset_manager, cls_method, vocab_size, force_train)
    vae_params = training_params.get_vae_params()
    model = training_params.get_model(cls_method, dt_train, dt_test, train_y, test_y)
    attack_params = training_params.get_attack_params(vocab_size, max_prefix_length)

    # the adversarial attacks   
    attack_manifold = AdversarialAttacks_manifold(dataset_manager, max_prefix_length, 
                                                dt_train = dt_train, dt_test = dt_test, train_y = train_y, test_y = test_y, train_cases = train_cases,  test_cases = test_cases, 
                                                model = model,
                                                training_params = training_params, vae_params = vae_params)

    attack_manifold.regular_adversarial_examples(attack_type, attack_cols[0], cls_method, force_attack, force_train)
    


if __name__ == "__main__":
    main()