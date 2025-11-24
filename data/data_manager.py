import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from data.data_sets import FolderDataset
from utils.util import list_dir, load_image, load_audio

class CSVDataManager(object):
    def __init__(self, config):
        load_formats = {
            'image': load_image,
            'audio': load_audio
        }

        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.splits = config['splits']

        self.load_func = load_formats[config['format']]
    
        mfile = os.path.join(self.dir_path, 'metadata/UrbanSound8K.csv') 
        metadata_df = pd.read_csv(mfile).sample(frac=1)
        self.metadata_df = self._remove_too_small(metadata_df, 1)

        self.classes = self._get_classes(self.metadata_df[['class', 'classID']])
        self.data_splits = self._10kfold_split(self.metadata_df)

    def _remove_too_small(self, df, min_sec=0.5):
        # Could also make length 1 if it is below 1 and then implicitly we
        # are padding the results
        dur_cond = (df['end'] - df['start'])>=min_sec
        not_train_cond = ~df['fold'].isin(self.splits['train'])

        return df[(dur_cond)|(not_train_cond)]

    def _get_classes(self, df):
        c_col = df.columns[0]
        idx_col = df.columns[1]
        return df.drop_duplicates().sort_values(idx_col)[c_col].unique()

    def _10kfold_split(self, df):
        ret = {}
        for s, inds in self.splits.items():

            df_split = df[df['fold'].isin(inds)]
            ret[s] = []

            for row in df_split[['slice_file_name', 'class', 'classID', 'fold']].values:
                fold_mod = 'audio/fold%s'%row[-1]
                fname = os.path.join(self.dir_path, fold_mod, '%s'%row[0])
                ret[s].append( {'path':fname, 'class':row[1], 'class_idx':row[2]} )
        return ret

    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return data.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    def pad_seq(self, batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)
        
        lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

        # seqs_pad -> (batch, time, channel) 
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        #seqs_pad = seqs_pad_t.transpose(0,1)
        return seqs_pad, lengths, srs, labels
