import copy
import logging
import os
import time
import psutil
import numpy as np

from autogluon.common.features.types import R_BOOL, R_INT, R_FLOAT, R_CATEGORY
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION, MULTICLASS, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.utils import try_import_catboost

from .callbacks import EarlyStoppingCallback, MemoryCheckCallback, TimeCheckCallback
from .catboost_utils import get_catboost_metric_from_ag_metric
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)

class ContrastiveTransformations():
    def __init__(self, X, y, process_fn, sample_weight):
        self.X = X
        self.y = y
        self.process_fn = process_fn
        self.cat_features = list(X.select_dtypes(include='category').columns)
        self.sample_weight = sample_weight

    def random_block(self, corruption_rate=0.6):
        X = copy.deepcopy(self.X)
        for column in X:
            if np.random.uniform() < corruption_rate:
                dtype = X.dtypes[column]
                X[column] = X[column].mode()[0]
                X[column] = X[column].astype(dtype)
        return self.to_pool(X)

    def random_perm(self, corruption_rate=0.3):
        X = copy.deepcopy(self.X)
        n, m = X.shape
        corruption_len = int(n * corruption_rate)
        for column in X:
            random_idx = np.random.randint(low=0, high=n, size=(n,))
            random_idx = random_idx[:corruption_len]
            X[column].iloc[random_idx] = X[column].sample(n=corruption_len).values
        return self.to_pool(X)

    def identical(self):
        X = copy.deepcopy(self.X)
        return self.to_pool(X)

    def to_pool(self, X, cat_features=None):
        from catboost import Pool
        cat_features = self.cat_features if cat_features is None else cat_features
        X = self.process_fn(X)
        return Pool(data=X, label=self.y, cat_features=cat_features, weight=self.sample_weight)


# TODO: Consider having CatBoost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatBoostModel(AbstractModel):
    """
    CatBoost model: https://catboost.ai/

    Hyperparameter options: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value('allow_writing_files', False)  # Disables creation of catboost logging files during training by default
        if self.problem_type != SOFTCLASS:  # TODO: remove this after catboost 0.24
            self._set_default_param_value('eval_metric', get_catboost_metric_from_ag_metric(self.stopping_metric, self.problem_type))

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)
        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include='category').columns)
        if self._category_features:
            X = X.copy()
            for category in self._category_features:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    def _estimate_memory_usage(self, X, **kwargs):
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        approx_mem_size_req = data_mem_usage * 7 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved
        return approx_mem_size_req

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             num_gpus=0,
             num_cpus=-1,
             sample_weight=None,
             sample_weight_val=None,
             **kwargs):
        time_start = time.time()
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        params['thread_count'] = num_cpus
        if self.problem_type == SOFTCLASS:
            # FIXME: This is extremely slow due to unoptimized metric / objective sent to CatBoost
            from .catboost_softclass_utils import SoftclassCustomMetric, SoftclassObjective
            params['loss_function'] = SoftclassObjective.SoftLogLossObjective()
            params['eval_metric'] = SoftclassCustomMetric.SoftLogLossMetric()

        model_type = CatBoostClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        num_rows_train = len(X)
        num_cols_train = len(X.columns)
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem

        contrastive_transformer = ContrastiveTransformations(X, y, self.preprocess, sample_weight)

        X = self.preprocess(X)
        cat_features = list(X.select_dtypes(include='category').columns)
        X = Pool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

        if X_val is None:
            eval_set = None
            early_stopping_rounds = None
        else:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features, weight=sample_weight_val)
            eval_set = X_val
            early_stopping_rounds = ag_params.get('ag.early_stop', 'adaptive')
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if params.get('allow_writing_files', False):
            if 'train_dir' not in params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    params['train_dir'] = self.path + 'catboost_info'

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get('verbosity', 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        num_features = len(self._features)

        if num_gpus != 0:
            if 'task_type' not in params:
                params['task_type'] = 'GPU'
                logger.log(20, f'\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.')
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get('task_type', None) == 'GPU':
            if 'colsample_bylevel' in params:
                params.pop('colsample_bylevel')
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')
            if 'rsm' in params:
                params.pop('rsm')
                logger.log(30, f'\t\'rsm\' is not supported on GPU, using default value (Default = 1).')

        if self.problem_type == MULTICLASS and 'rsm' not in params and 'colsample_bylevel' not in params and num_features > 1000:
            # Subsample columns to speed up training
            if params.get('task_type', None) != 'GPU':  # RSM does not work on GPU
                params['colsample_bylevel'] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(30, f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).')
                logger.log(30, f'\tTo disable this functionality, explicitly specify \'colsample_bylevel\' in the model hyperparameters.')
            else:
                params['colsample_bylevel'] = 1.0
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')

        logger.log(15, f'\tCatboost model hyperparameters: {params}')

        extra_fit_kwargs = dict()
        if params.get('task_type', None) != 'GPU':
            callbacks = []
            if early_stopping_rounds is not None:
                callbacks.append(EarlyStoppingCallback(stopping_rounds=early_stopping_rounds, eval_metric=params['eval_metric']))

            if num_rows_train * num_cols_train * num_classes > 5_000_000:
                # The data is large enough to potentially cause memory issues during training, so monitor memory usage via callback.
                callbacks.append(MemoryCheckCallback())
            if time_limit is not None:
                time_cur = time.time()
                time_left = time_limit - (time_cur - time_start)
                if time_left <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                    raise TimeLimitExceeded
                callbacks.append(TimeCheckCallback(time_start=time_cur, time_limit=time_left))
            extra_fit_kwargs['callbacks'] = callbacks
        else:
            logger.log(30, f'\tWarning: CatBoost on GPU is experimental. If you encounter issues, use CPU for training CatBoost instead.')
            if time_limit is not None:
                params['iterations'] = self._estimate_iter_in_time_gpu(
                    X=X,
                    eval_set=eval_set,
                    time_limit=time_limit,
                    verbose=verbose,
                    params=params,
                    num_rows_train=num_rows_train,
                    time_start=time_start,
                    model_type=model_type,
                )
            if early_stopping_rounds is not None:
                if isinstance(early_stopping_rounds, int):
                    extra_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
                elif isinstance(early_stopping_rounds, tuple):
                    extra_fit_kwargs['early_stopping_rounds'] = 50

        # TODO: Custom metrics don't seem to work anymore
        # TODO: Custom metrics not supported in GPU mode
        # TODO: Callbacks not supported in GPU mode
        fit_final_kwargs = dict(
            eval_set=eval_set,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        if eval_set is not None:
            fit_final_kwargs['use_best_model'] = True

        is_pretrain = params.pop('pretrainer') if 'pretrainer' in params else False
        if is_pretrain:
            params_head = copy.deepcopy(params)
            max_iteration = params['iterations']
            params['iterations'] = 1
            params_head['iterations'] = max_iteration
            self.model = model_type(**params)
            dummy = model_type(**params_head)
            for _ in range(100):
                X_aug = contrastive_transformer.random_block()
                self.model.fit(X_aug,
                               init_model=None if _ == 0 else self.model,
                               #eval_set=eval_set,
                               verbose=False,
                               #use_best_model=False,
                               )

            dummy.fit(X,
                      init_model=self.model,
                      # eval_set=eval_set,
                      # verbose=False,
                      # use_best_model=False,
                      **fit_final_kwargs
                      )
            self.model = dummy

        else:
            self.model = model_type(**params)
            self.model.fit(X, **fit_final_kwargs)

        self.params_trained['iterations'] = self.model.tree_count_
        print(self.model.tree_count_)

    # FIXME: This logic is a hack made to maintain compatibility with GPU CatBoost.
    #  GPU CatBoost does not support callbacks or custom metrics.
    #  Since we use callbacks to check memory and training time in CPU mode, we need a way to estimate these things prior to training for GPU mode.
    #  This method will train a model on a toy number of iterations to estimate memory and training time.
    #  It will return an updated iterations to train on that will avoid running OOM and running over time limit.
    #  Remove this logic once CatBoost fixes GPU support for callbacks and custom metrics.
    def _estimate_iter_in_time_gpu(self, *, X, eval_set, time_limit, verbose, params, num_rows_train, time_start, model_type):
        import math
        import pickle
        import sys

        modifier = min(1.0, 10000 / num_rows_train)
        num_sample_iter_max = max(round(modifier * 50), 2)
        time_left_start = time_limit - (time.time() - time_start)
        if time_left_start <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
            raise TimeLimitExceeded
        default_iters = params['iterations']
        params_init = params.copy()
        num_sample_iter = min(num_sample_iter_max, params_init['iterations'])
        params_init['iterations'] = num_sample_iter
        sample_model = model_type(
            **params_init,
        )
        sample_model.fit(
            X,
            eval_set=eval_set,
            use_best_model=True,
            verbose=verbose,
        )

        time_left_end = time_limit - (time.time() - time_start)
        time_taken_per_iter = (time_left_start - time_left_end) / num_sample_iter
        estimated_iters_in_time = round(time_left_end / time_taken_per_iter)

        available_mem = psutil.virtual_memory().available
        if self.problem_type == SOFTCLASS:
            model_size_bytes = 1  # skip memory check
        else:
            model_size_bytes = sys.getsizeof(pickle.dumps(sample_model))

        max_memory_proportion = 0.3
        mem_usage_per_iter = model_size_bytes / num_sample_iter
        max_memory_iters = math.floor(available_mem * max_memory_proportion / mem_usage_per_iter)

        final_iters = min(default_iters, min(max_memory_iters, estimated_iters_in_time))
        return final_iters

    def _predict_proba(self, X, **kwargs):
        if self.problem_type != SOFTCLASS:
            return super()._predict_proba(X, **kwargs)
        # For SOFTCLASS problems, manually transform predictions into probabilities via softmax
        X = self.preprocess(X, **kwargs)
        y_pred_proba = self.model.predict(X, prediction_type='RawFormulaVal')
        y_pred_proba = np.exp(y_pred_proba)
        y_pred_proba = np.multiply(y_pred_proba, 1/np.sum(y_pred_proba, axis=1)[:, np.newaxis])
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:,1]
        return y_pred_proba

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_early_stopping_rounds(self, num_rows_train, strategy='auto'):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _ag_params(self) -> set:
        return {'ag.early_stop'}

    def _validate_fit_memory_usage(self, **kwargs):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        if approx_mem_size_req > 1e9:  # > 1 GB
            available_mem = psutil.virtual_memory().available
            ratio = approx_mem_size_req / available_mem
            if ratio > (1 * max_memory_usage_ratio):
                logger.warning('\tWarning: Not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
                raise NotEnoughMemoryError
            elif ratio > (0.75 * max_memory_usage_ratio):
                logger.warning('\tWarning: Potentially not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

    def _get_default_resources(self):
        # psutil.cpu_count(logical=False) is faster in training than psutil.cpu_count()
        num_cpus = psutil.cpu_count(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def _more_tags(self):
        # `can_refit_full=True` because iterations is communicated at end of `_fit`
        return {'can_refit_full': True}
