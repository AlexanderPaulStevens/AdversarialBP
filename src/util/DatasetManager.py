import util.dataset_confs as dataset_confs
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from collections import OrderedDict
from pandas.api.types import is_string_dtype
# torch packages
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from collections import Counter
from util.settings import global_setting

class DatasetManager:

    def __init__(self, dataset_name, cls_method):
        self.dataset_name = dataset_name
        self.cls_method = cls_method
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]

        self.sorting_cols = [self.timestamp_col, self.activity_col]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
        
        if self.cls_method in ['LR', 'RF', 'XGB']:
            self.algorithm = 'ML'
        elif self.cls_method in ['LSTM', 'GAN', 'BNN']:
            self.algorithm = 'DL'

    def read_dataset(self):
        # read dataset
        dtypes = {col: "object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data

    def split_data(self, data, train_ratio, split="temporal", seed=22):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)

    def split_data_strict(self, data, train_ratio, split="temporal"):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)

    def split_data_discard(self, data, train_ratio, split="temporal"):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        overlapping_cases = train[train[self.timestamp_col] >= split_ts][self.case_id_col].unique()
        train = train[~train[self.case_id_col].isin(overlapping_cases)]
        return (train, test)

    def split_val(self, data, val_ratio, split="random", seed=22):
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio*len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)

    def generate_prefix_data(self, data):
        # generate prefix data (each possible prefix becomes a trace)
        case_length = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        data.loc[:, 'case_length'] = case_length.copy()
        dt_prefixes = data[data['case_length'] >= self.min_prefix_length].groupby(self.case_id_col).head(self.min_prefix_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(self.min_prefix_length, self.max_prefix_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(self.max_prefix_length, x))

        return dt_prefixes
    
    def extract_args(self, data):

        self.cls_encoder_args = {'case_id_col': self.case_id_col,
                            'dynamic_cat_cols': self.dynamic_cat_cols,
                            'dynamic_num_cols': self.dynamic_num_cols,
                            'fillna': True}
        
        # determine min and max (truncated) prefix lengths
        case_length = data.groupby(self.case_id_col)[self.activity_col].transform(len)
        self.min_prefix_length = max(1, case_length.min())

        if "traffic_fines" in self.dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in self.dataset_name:
            max_prefix_length = min(20, self.get_pos_case_length_quantile(data, 0.90))
        elif "production" in self.dataset_name:
            max_prefix_length = 23
        else:
            max_prefix_length = min(40, self.get_pos_case_length_quantile(data, 0.90))

        self.max_prefix_length = max_prefix_length

        print('the case lenghts are between length', self.min_prefix_length, 'and', self.max_prefix_length)

        # resource col
        if self.dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
            self.resource_col = 'org:resource'
        else:
            self.resource_col = 'Resource'

    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.ceil(data[data[self.label_col] == self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids

    def get_label_numeric(self, data):
        y = self.get_label(data)  # one row per case
        return [1 if label == int(1) else 0 for label in y]

    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()

    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)

    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(dt_for_splitting, dt_for_splitting[self.label_col]):
            current_train_names = dt_for_splitting[self.case_id_col][train_index]
            current_test_names = dt_for_splitting[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)

    def prepare_inputs(self, X_train, X_test):
        global ce
        ce = ColumnEncoder()
        # converts the columns in the X_train and X_test to string types. 
        # This is typically done to ensure that all values in the DataFrame are treated as strings when encoding.

        X_train, X_test = X_train.astype(str), X_test.astype(str)
        

        X_train_enc = ce.fit_transform(X_train)
        X_test_enc = ce.transform(X_test)
        
        self.vocab_size = len(list(ce.get_maps().values())[0].keys()) + 2 # padding value and EoS token
        
        print('vocab size:', self.vocab_size)
        print('dictionary of activity values', list(list(ce.get_maps().values())))
        self.ce = ce
        return X_train_enc, X_test_enc, ce
    
    def pad_data(self, data):
        data[0] = nn.ConstantPad1d((0, self.max_prefix_length - data[0].shape[0]), 0)(data[0])
        padding = pad_sequence(data, batch_first=True, padding_value=0)
        return padding
    
    def groupby_caseID(self, data):
        deviant_sequences, regular_sequences = [], [] # the sequences
        deviant_labels, regular_labels = [], []   # the labels
        case_id_deviant, case_id_regular = [], [] # the case IDs
        groups = data[self.cols].groupby('Case ID', as_index=False) # group by 'Case ID' to ensure each group contains all rows for a 'Case ID'
        groups = groups.apply(lambda group: group.sort_values('event_nr')) # sort the rows in each group by 'event_nr'
        grouped = groups.groupby('Case ID') # grouping again on the 'Case ID' to ensure each group contains the sorted rows for its 'Case ID'
        
        for case_id, group in grouped:
            if self.is_ordered(list(group['event_nr'])): 
                label = group['label'].iloc[0]
                sequence = list(group[self.activity_col])
                sequence.extend([self.vocab_size-1]) # add EoS token
                if label == 1: # deviant is 1
                        deviant_sequences.append(sequence)
                        deviant_labels.append(label)
                        case_id_deviant.append(case_id)

                elif label == 0: # 0 is regular
                    if sequence not in deviant_sequences:
                        regular_sequences.append(sequence)
                        regular_labels.append(label)
                        case_id_regular.append(case_id)
            else:
                print('problem', group)
                breakpoint()
                AssertionError

        # Find sequences with the same activities but different labels
        duplicate_sequences = set()
        for i, regular_seq in enumerate(regular_sequences):
            for j, deviant_seq in enumerate(deviant_sequences):
                if regular_seq == deviant_seq:
                    duplicate_sequences.add(i)
                    duplicate_sequences.add(j)
        
        # Remove duplicate sequences, labels, and case IDs
        sequences_regular = [seq for i, seq in enumerate(regular_sequences) if i not in duplicate_sequences]
        labels_regular = [label for i, label in enumerate(regular_labels) if i not in duplicate_sequences]
        case_ids_regular = [case_id for i, case_id in enumerate(case_id_regular) if i not in duplicate_sequences]
        sequences_deviant = [seq for i, seq in enumerate(deviant_sequences) if i not in duplicate_sequences]
        labels_deviant = [label for i, label in enumerate(deviant_labels) if i not in duplicate_sequences]
        case_ids_deviant = [case_id for i, case_id in enumerate(case_id_deviant) if i not in duplicate_sequences]

        #print('before removing ambiguous traces', len(regular_sequences + deviant_sequences))
        #print('after removing', len(sequences_regular + sequences_deviant))
        return sequences_regular, sequences_deviant, labels_regular, labels_deviant, case_ids_regular, case_ids_deviant
    
    def groupby_pad(self,prefixes):
        ans_regular, ans_deviant, label_regular, label_deviant, cases_regular, cases_deviant = self.groupby_caseID(prefixes)
        ans = ans_regular + ans_deviant
        label = np.array(label_regular + label_deviant)
        cases = cases_regular + cases_deviant
        ###### CAT COL################
        return ans, label, cases
    
    def is_ordered(self, lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def ohe_cases(self, activity_lists):
        # Pad activity lists with zeros to ensure uniform length

        padded_activity = np.array([
            seq + [0] * (self.max_prefix_length +1  - len(seq)) if len(seq) < self.max_prefix_length +1  else seq
            for seq in activity_lists
        ]) # we add 1 because we added the EoS token
        
        # Initialize an empty numpy array
        num_instances = len(padded_activity)
        one_hot_matrix = torch.zeros((num_instances, self.max_prefix_length+1, self.vocab_size), dtype=int)
        
        # Iterate over sequences and populate the matrix
        for i, seq in enumerate(padded_activity):
            for j, activity in enumerate(seq):
                one_hot_matrix[i, j, activity] = 1
        #one_hot_matrix = one_hot_matrix[:,:,1:]
        return padded_activity, one_hot_matrix
    
    def generate_cases_of_length(self, train, test):
        # create the case lengths
        train['case_length'] = train.groupby(self.case_id_col)[self.activity_col].transform(len)
        test['case_length'] = test.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        # filter out the short cases
        train = train[train['case_length'] >= self.min_prefix_length].groupby(self.case_id_col).head(self.max_prefix_length)
        test = test[test['case_length'] >= self.min_prefix_length].groupby(self.case_id_col).head(self.max_prefix_length) 
        return train, test
    
    def create_data(self, cat_cols, prefix_generation):
        train_ratio = global_setting['train_ratio']
        data = self.read_dataset()
        self.extract_args(data) #extract the arguments
        self.cat_cols = cat_cols # the categorical columns
        train, test = self.split_data_strict(data, train_ratio, split="temporal")

        self.cols = cat_cols.copy()
        self.cols.extend([self.case_id_col, 'label', 'event_nr',  'case_length']) # the columns to be used

        # prefix generation of test data
        if prefix_generation:
            dt_train_prefixes = self.generate_prefix_data(train)
            dt_test_prefixes = self.generate_prefix_data(test)

        else:
            # create the case lengths
            train, test = self.generate_cases_of_length(train, test)
            dt_train_prefixes = train.copy()
            dt_test_prefixes = test.copy()

        print('there are', len(set(train[train['label']=='deviant']['Case ID'])), 'deviant cases and', len(set(train[train['label']=='regular']['Case ID'])), 'regular cases')

        dt_train_prefixes = dt_train_prefixes[self.cols].copy()
        dt_test_prefixes = dt_test_prefixes[self.cols].copy()

        dt_train_prefixes.loc[(dt_train_prefixes['label'] == 'deviant'), 'label'] = 1
        dt_train_prefixes.loc[(dt_train_prefixes['label'] == 'regular'), 'label'] = 0
        dt_test_prefixes.loc[(dt_test_prefixes['label'] == 'deviant'), 'label'] = 1
        dt_test_prefixes.loc[(dt_test_prefixes['label'] == 'regular'), 'label'] = 0
        train_cat_cols, test_cat_cols, ce = self.prepare_inputs(dt_train_prefixes.loc[:, self.cat_cols], dt_test_prefixes.loc[:, self.cat_cols]) # We can reverse (or extract the encoding mappings) with the variable 'ce'
        dt_train_prefixes[self.cat_cols] = train_cat_cols
        dt_test_prefixes[self.cat_cols] = test_cat_cols

        vocab_size = []
        for cat_col in range(len(ce.get_maps().values())):
                vocab_size.append(len(list(ce.get_maps().values())[cat_col].keys()) +2) # for the padding token and the EoS token
        x_train, train_y, train_cases = self.groupby_pad(dt_train_prefixes)
        x_test, test_y, test_cases = self.groupby_pad(dt_test_prefixes)

        return x_train, x_test, train_y, test_y, train_cases, test_cases, vocab_size, self.max_prefix_length
    
    def transform_data(self, dt, traintest):
        # Initialize a list of lists to store the frequency-based array
        frequency_array = []

        # Iterate over each sequence
        for seq in dt:
            # Count the occurrences of each value in the sequence
            freq_count = Counter(seq)
            
            # Create a list to store the frequencies of values
            freq_list = [freq_count.get(value, 0) for value in range(1, self.vocab_size)] # we exclude the padding token
            
            # Append the frequency list to the frequency array
            frequency_array.append(freq_list)
        frequency_array = np.array(frequency_array)
        if traintest == 'train':   
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(frequency_array)     
        else:
            data_scaled = self.scaler.transform(frequency_array)
            
        return data_scaled
    
    def prefix_lengths_adversarial(self, ans, x_hat_param):
        """Obtain the prefix lengths of the adversarial traces."""
        ans2 = ans.copy()
        for i in range(0, len(ans2)):
            ans2[i] = torch.argmax(x_hat_param[i], dim=1)[0:len(ans2[i])]
        return ans2
    
    def return_indices_correlated_guess(self, a, b):
        """Return the indices that are the same."""
        return [i for i, v in enumerate(a) if v == b[i]]

    def return_indices_adversarial_guess(self, a, b):
        """Return the indices that are different."""
        return [i for i, v in enumerate(a) if v != b[i]]
    
    def edit_distance(self, factual, counterfactual, verbose=False) -> int:
        """
        Calculate the word level edit (Levenshtein) distance between two sequences.

        .. devices:: CPU

        The function computes an edit distance allowing deletion, insertion and
        substitution. The result is an integer.

        For most applications, the two input sequences should be the same type. If
        two strings are given, the output is the edit distance between the two
        strings (character edit distance). If two lists of strings are given, the
        output is the edit distance between sentences (word edit distance). Users
        may want to normalize the output by the length of the reference sequence.

        Args:
            seq1 (Sequence): the first sequence to compare.
            seq2 (Sequence): the second sequence to compare.
        Returns:
            int: The distance between the first and second sequences.
        """
        factual_max = np.argmax(factual, axis=1)
        counterfactual_max = np.argmax(counterfactual, axis=1)
        if verbose:
            print('edit distance', factual_max.shape, counterfactual_max.shape)
            print(factual_max, counterfactual_max)
            print('factual', factual_max, 'counterfactual', counterfactual_max)
        len_sent2 = len(counterfactual_max)
        dold = list(range(len_sent2 + 1))
        dnew = [0 for _ in range(len_sent2 + 1)]

        for i in range(1, len(factual_max) + 1):
            dnew[0] = i
            for j in range(1, len_sent2 + 1):
                if factual_max[i - 1] == counterfactual_max[j - 1]:
                    dnew[j] = dold[j - 1]
                else:
                    substitution = dold[j - 1] + 1
                    insertion = dnew[j - 1] + 1
                    deletion = dold[j] + 1
                    dnew[j] = min(substitution, insertion, deletion)

            dnew, dold = dold, dnew

        return int(dold[-1])

# https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2
class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None
        self.maps = dict()

    def transform(self, X):
        # encodes its categorical columns using the encoding scheme stored in self.maps
        X_copy = X.copy()
        for col in self.columns:
            # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
            X_copy.loc[:, col] = X_copy.loc[:, col].apply(lambda x: self.maps[col].get(x, -1))
        # It returns a copy of X with the categorical columns replaced by their corresponding integer encodings.
        return X_copy
    
    def get_maps(self):
        # This method allows you to retrieve the encoding mappings stored in 
        return self.maps

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            values = list(self.maps[col].keys())
            # find value in ordered list and map out of range values to None
            X_copy.loc[:, col] = [values[i-1] if 0 < i <= len(values) else None for i in X_copy[col]]
        return X_copy

    def fit(self, X, y=None):
        # only apply to string type columns
        # This method is called during the fitting process. 
        # It identifies the categorical columns in the input DataFrame X and stores them in self.maps
        # These mappings are dictionaries that map each unique categorical value to an integer
        self.columns = [col for col in X.columns if is_string_dtype(X[col])]
        for col in self.columns:
            self.maps[col] = OrderedDict({value: num+1 for num, value in enumerate(sorted(set(X[col])))})
        return self
    
class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.LongTensor(x).to(device)
        self.Y_train = torch.LongTensor(y).to(device)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]