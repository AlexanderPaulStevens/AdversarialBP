from typing import List

import numpy as np
import torch
import pandas as pd

class Redundancy():
    """
    Computes redundancy for each adversarial_example
    """

    def __init__(self, mlmodel, hyperparameters):
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["Redundancy"]

    # computes the redudancy between the factual and the adversarial_example
    # quantifies how many features can be changed without alternating the model's prediction outcome
    def _compute_redundancy(
        self, factual: np.ndarray, adversarial_example: np.ndarray
    ) -> int:
        redundancy = 0
        for col_idx in range(len(adversarial_example)):
            # if feature is changed
            if factual[col_idx] != adversarial_example[col_idx]:
                temp_cf = np.copy(adversarial_example)
                temp_cf[col_idx] = factual[col_idx]
                # see if change is needed to flip the label
                temp_pred = np.argmax(
                    self.mlmodel.predict_proba(temp_cf.reshape((1, -1)))
                )
                if temp_pred == self.cf_label:
                    redundancy += 1
        return redundancy

    def _redundancy(
        self,
        factuals: torch.tensor,
        adversarial_examples: torch.tensor,
    ) -> List[List[int]]:
        """
        Computes Redundancy measure for every adversarial_example.

        Parameters
        ----------
        factuals:
            Encoded and normalized factual samples.
        adversarial_examples:
            Encoded and normalized adversarial_example samples.

        Returns
        -------
        List with redundancy values per adversarial_example sample
        """
        df_enc_norm_fact = factuals.reset_index(drop=True)
        df_cfs = adversarial_examples.reset_index(drop=True)

        df_cfs["redundancy"] = df_cfs.apply(
            lambda x: self._compute_redundancy(
                df_enc_norm_fact.iloc[x.name].values,
                x.values,
            ),
            axis=1,
        )
        return df_cfs["redundancy"].values.reshape((-1, 1)).tolist()

    def get_evaluation(self, adversarial_examples, factuals):
        print('this?')
        print(type(factuals), type(adversarial_examples))
        if len(adversarial_examples)>1:
            redundancies = []
        else:
            redundancies = self._redundancy(
                factuals,
                adversarial_examples,
            )

        return pd.DataFrame(redundancies, columns=self.columns)
