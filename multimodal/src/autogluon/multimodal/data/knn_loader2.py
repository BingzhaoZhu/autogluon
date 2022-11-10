import copy

import numpy as np
import random
from sklearn.decomposition import PCA
from typing import List, Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler

from ..constants import PREDICT, TEST, TRAIN, VAL
from .collator import Dict
from .dataset import BaseDataset
from .preprocess_dataframe import MultiModalFeaturePreprocessor
from sklearn.preprocessing import OneHotEncoder


class KnnSampler(Sampler):
    def __init__(self, data_source, batch_size=128, n=1):
        super().__init__(data_source)
        self.raw_data_source = data_source
        self.batch_size = batch_size
        self.pca = PCA(n_components=n)
        self.encoders = {}
        self.n = n
        self.data_source = self.onehot_encode(data_source)
        self.fit_pca()
        self.perm = self.get_even_clusters()

    def __iter__(self):
        self.perm = self.get_even_clusters()
        for batch_ in self.perm:
            yield batch_

    def __len__(self) -> int:
        return len(self.data_source)

    def fit_pca(self):
        X = self.data_source
        i = random.randint(0, self.n - 1)
        X_ = self.pca.fit_transform(X)[:, i].reshape(-1)

    def onehot_encode(self, basedataset):
        df = []
        if hasattr(basedataset, "categorical_0"):
            for col in basedataset.categorical_0:
                jobs_encoder = OneHotEncoder(sparse=False)
                transformed = jobs_encoder.fit_transform(basedataset.categorical_0[col].reshape(-1, 1))
                ohe_df = pd.DataFrame(transformed)
                self.encoders[col] = jobs_encoder
                df.append(ohe_df)
        if hasattr(basedataset, "numerical_0"):
            df.append(pd.DataFrame.from_dict(basedataset.numerical_0))
        df = pd.concat(df, axis=1)
        return df

    def apply_onehot_encode(self, basedataset):
        df = []
        if hasattr(basedataset, "categorical_0"):
            for col in basedataset.categorical_0:
                jobs_encoder = self.encoders[col]
                transformed = jobs_encoder.transform(basedataset.categorical_0[col].reshape(-1, 1))
                ohe_df = pd.DataFrame(transformed)
                df.append(ohe_df)
        if hasattr(basedataset, "numerical_0"):
            df.append(pd.DataFrame.from_dict(basedataset.numerical_0))
        df = pd.concat(df, axis=1)
        return df

    def encode_and_bind(self, original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return res

    def get_even_clusters(self):
        X = self.data_source
        i = random.randint(0, self.n-1)
        X_ = self.pca.fit_transform(X)[:, i].reshape(-1)
        ind = np.argsort(X_, axis=0)
        return ind

    def transform(self, data):
        raw_data = data
        data = self.apply_onehot_encode(data)
        pc_val, pc_tr = self.pca.transform(data), self.pca.transform(self.data_source)
        idx_all = []
        for index in range(data.shape[0]):
            distance = abs(pc_tr-pc_val[index]).reshape(-1)
            idx = np.argpartition(distance, self.batch_size-1)
            idx_all.append(idx[:self.batch_size-1])
        data_with_support = copy.deepcopy(raw_data)
        if hasattr(data_with_support, "categorical_0"):
            for col in data_with_support.categorical_0:
                supported_col = []
                for i, idx in enumerate(idx_all):
                    supported_col.append(raw_data.categorical_0[col][i:i+1])
                    supported_col.append(self.raw_data_source.categorical_0[col][idx])
                data_with_support.categorical_0[col] = np.concatenate(supported_col, axis=0)

        if hasattr(data_with_support, "numerical_0"):
            for col in data_with_support.numerical_0:
                supported_col = []
                for i, idx in enumerate(idx_all):
                    supported_col.append(raw_data.numerical_0[col][i:i+1])
                    supported_col.append(self.raw_data_source.numerical_0[col][idx])
                data_with_support.numerical_0[col] = np.concatenate(supported_col, axis=0)

        if hasattr(data_with_support, "label_0"):
            for col in data_with_support.label_0:
                supported_col = []
                for i, idx in enumerate(idx_all):
                    supported_col.append(raw_data.label_0[col][i:i + 1])
                    supported_col.append(self.raw_data_source.label_0[col][idx])
                data_with_support.label_0[col] = np.concatenate(supported_col, axis=0)

        data_with_support.lengths = [i * self.batch_size for i in data_with_support.lengths]

        return data_with_support


class KnnDataModule(LightningDataModule):
    """
    Set up Pytorch DataSet and DataLoader objects to prepare data for single-modal/multimodal training,
    validation, testing, and prediction. We organize the multimodal data using pd.DataFrame.
    For some modalities, e.g, image, that cost much memory, we only store their disk path to do lazy loading.
    This class inherits from the Pytorch Lightning's LightningDataModule:
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
        data_processors: Union[dict, List[dict]],
        per_gpu_batch_size: int,
        num_workers: int,
        train_data: Optional[pd.DataFrame] = None,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        predict_data: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        df_preprocessor
            One or a list of dataframe preprocessors. The preprocessing of one modality is generic so that
            the preprocessed data can be used by different models requiring the modality.
            For example, formatting input data as strings is a valid preprocessing operation for text.
            However, tokenizing strings into ids is invalid since different models generally
            use different tokenizers.
        data_processors
            The data processors to prepare customized data for each model. Each processor is only charge of
            one modality of one model. This helps scale up training arbitrary combinations of models.
        per_gpu_batch_size
            Mini-batch size for each GPU.
        num_workers
            Number of workers for Pytorch DataLoader.
        train_data
            Training data.
        val_data
            Validation data.
        test_data
            Test data.
        predict_data
            Prediction data. No labels required in it.
        """
        super().__init__()
        self.prepare_data_per_node = True

        if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
            df_preprocessor = [df_preprocessor]
        if isinstance(data_processors, dict):
            data_processors = [data_processors]

        self.df_preprocessor = df_preprocessor
        self.data_processors = data_processors
        self.per_gpu_batch_size = per_gpu_batch_size
        self.num_workers = num_workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data

        # if self.train_data is not None:
        #     self.set_dataset(TRAIN)
        #     self.train_sampler = KnnSampler(self.train_dataset, batch_size=self.per_gpu_batch_size)

    def set_dataset(self, split):
        data_split = getattr(self, f"{split}_data")
        dataset = BaseDataset(
            data=data_split,
            preprocessor=self.df_preprocessor,
            processors=self.data_processors,
            is_training=split == TRAIN,
        )
        setattr(self, f"{split}_dataset", dataset)


    def setup(self, stage):
        """
        Set up datasets for different stages: "fit" (training and validation), "test", and "predict".
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup

        Parameters
        ----------
        stage
            Stage name including choices:
                - fit (For the fitting stage)
                - test (For the test stage)
                - predict (For the prediction stage)
        """
        if stage == "fit":
            self.set_dataset(TRAIN)
            self.set_dataset(VAL)
        elif stage == "test":
            self.set_dataset(TEST)
        elif stage == "predict":
            self.set_dataset(PREDICT)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        """
        Create the dataloader for training.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#train-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        self.train_sampler = KnnSampler(self.train_dataset, batch_size=self.per_gpu_batch_size, n=10)
        loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def val_dataloader(self):
        """
        Create the dataloader for validation.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#val-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        self.val_sampler = KnnSampler(self.val_dataset, batch_size=self.per_gpu_batch_size)
        val_dataset = self.val_dataset
        loader = DataLoader(
            val_dataset,
            sampler=self.val_sampler,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def test_dataloader(self):
        """
        Create the dataloader for test.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#test-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        self.test_sampler = KnnSampler(self.test_dataset, batch_size=self.per_gpu_batch_size)
        test_dataset = self.test_dataset
        loader = DataLoader(
            test_dataset,
            sampler=self.test_sampler,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def predict_dataloader(self):
        """
        Create the dataloader for prediction.
        This method is registered by Pytorch Lightning's LightningDataModule.
        Refer to: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#predict-dataloader

        Returns
        -------
        A Pytorch DataLoader object.
        """
        self.predict_sampler = KnnSampler(self.predict_dataset, batch_size=self.per_gpu_batch_size)
        predict_dataset = self.predict_dataset
        loader = DataLoader(
            predict_dataset,
            sampler=self.predict_sampler,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.get_collate_fn(),
        )
        return loader

    def get_collate_fn(self):
        """
        Collect collator functions for each modality input of every model.
        These collator functions are wrapped by the "Dict" collator function,
        which can then be used by the Pytorch DataLoader.

        Returns
        -------
        A "Dict" collator wrapping other collators.
        """
        collate_fn = {}
        for per_preprocessor, per_data_processors_group in zip(self.df_preprocessor, self.data_processors):
            for per_modality in per_data_processors_group:
                per_modality_column_names = per_preprocessor.get_column_names(modality=per_modality)
                if per_modality_column_names:
                    for per_model_processor in per_data_processors_group[per_modality]:
                        collate_fn.update(per_model_processor.collate_fn(per_modality_column_names))
        return Dict(collate_fn)
