import numpy as np
import optuna
import logging
from math import ceil
from optuna import study
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

def get_params_space_and_org(input_len, patch_len):
    params_space = {
        'sampler_factor': {
            'type': 'int',
            # 'values': np.arange(1, 2 + 1, 1)  # 减小space
            'values': np.array([1])
        },
        'trimmer_seq_len': {
            'type': 'int',  # (40,96,1) Chronos+Cuda稳定报错
            # 'values': np.arange(5 * patch_len, 16 * patch_len, int(patch_len * 1))  # 补align危险？？？
            'values': np.array([input_len])
        },
        'aligner_mode': {
            'type': 'str',
            'values': ['none']  # 减小Space
        },
        'aligner_method': {
            'type': 'str',  # zero_pad会被明显排除 不过会影响整体可视化和debug
            'values': ['edge_pad']  # 减小Space
        },
        'normalizer_method': {
            'type': 'str',  # 'minmax', 'maxabs' 减小搜索空间 ????????????'minmax', 'maxabs'???????????
            'values': ['none', 'standard', 'robust']  # robust略慢...！！
        },
        'normalizer_mode': {
            'type': 'str',  # train overfit??? ???????????????????????leak！ 'history'导致Weather坏?????
            'values': ['input']
        },
        'normalizer_ratio': {  # new!!!
            'type': 'str',
            'values': [1]  # 减小space
        },
        'inputer_detect_method': {
            'type': 'str',  # iqr计算时间长了一点(1/2 model)！！
            'values': ['none', '3_sigma', '1.5_iqr']
        },
        'inputer_fill_method': {  # forward_fill感觉很多时候也不如不impute... # forward_fill在ETT上差！
            'type': 'str',  # 'forward_fill', 'backward_fill' 减小搜索空间 # rolling_mean有时候遇大倾斜很坏！！！
            'values': ['linear_interpolate']
        },
        'warper_method': {  # 'boxcox'+Uni2ts -> 有时候nan 而且秒级 # yeojohnson 有时候会-1847倍...overfit..??
            'type': 'str',  # 'log' 'sqrt' 坏
            'values': ['none', 'log']  # 减小space
        },
        'decomposer_period': {  # 有点太慢了...并行抢cpu # 现在还行 # 整体貌似会变差？rand？
            'type': 'str',
            'values': ['none']
        },
        'decomposer_components': {
            'type': 'str',
            'values': ['none']
        },
        # Differentiator
        'differentiator_n': {
            'type': 'int',
            'values': [0, 1]
            # if ablation != 'Differentiator' else [0],
        },
        'pipeline_name': {
            'type': 'str',
            'values': ['produce1', 'produce2']  # 减小Space
        },
        'denoiser_method': {
            'type': 'str',  # 'moving_average' 配上 forward—fill导致UTSD在ETT上很差！？ # 'fft'没用？
            'values': ['none', 'ewma']  # 减小Space
        },
        'clip_factor': {
            'type': 'str',
            'values': ['none', '0', '0.25']
            # if ablation != 'Clipper' else ['none'],
        }
        # 大约十万的space yes
    }
    # logging.info(f"params_space={params_space}")
    # 注意：这个setting其实是比较随意的，最终只是为了说明默认值的效果一般（最好follow论文的设置）
    origin_param_dict = {  # FIXME：
        'sampler_factor': 1,
        'trimmer_seq_len': input_len,
        'aligner_mode': 'none',
        'aligner_method': 'edge_pad',  # model_patch之后org一定需要是none！！！
        'normalizer_method': 'none',  # FIXME： 目前已经使用了Timer内置的std的scaler
        'normalizer_mode': 'input',
        'normalizer_ratio': 1,
        'inputer_detect_method': 'none',
        'inputer_fill_method': 'linear_interpolate',
        'warper_method': 'none',
        'decomposer_period': 'none',
        'decomposer_components': 'none',
        'differentiator_n': 0,
        'pipeline_name': 'produce2',
        'denoiser_method': 'none',
        'clip_factor': 'none'  # FIXME: 原来是0 但实际上不影响，只是算子内部的clip
    }
    # logging.info(f"origin_param_dict={origin_param_dict}")
    return params_space, origin_param_dict

class OptunaTuner:

    def __init__(self, args, params_space, direction, enqueue_param_dicts):
        self.distributions = {}
        for key, value in params_space.items():
            if value['type'] == 'float':
                self.distributions[key] = FloatDistribution(min(value['values']), max(value['values']))
            elif value['type'] == 'int':
                step = 1 if len(value['values']) == 1 else ceil(value['values'][1] - value['values'][0])
                self.distributions[key] = IntDistribution(min(value['values']), max(value['values']), step=step)
            elif value['type'] == 'str':
                self.distributions[key] = CategoricalDistribution(value['values'])
            else:
                raise ValueError(f"Unknown type: {value['type']}")
        sampler = optuna.samplers.TPESampler(seed=args.seed)

        # if args.pruner_name == 'MedianPruner':
        #     pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5,interval_steps=1, n_min_trials=10)
        # elif args.pruner_name == 'PercentilePruner':
        #     pruner = optuna.pruners.PercentilePruner(percentile=30, n_startup_trials=5, n_warmup_steps=5,interval_steps=1, n_min_trials=10)
        # elif args.pruner_name == 'NoPruner':
        pruner = optuna.pruners.NopPruner()
        # else:
        #     raise ValueError(f"Unknown pruner_name: {args.pruner_name}")

        self.study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
        self.random_study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=args.seed))
        self.param_dict_trial_dict = {}
        self.max_repeat = 10000  # FIXME：重复的后果很严重！，影响排序计数

        if enqueue_param_dicts is not None:
            for param_dict in enqueue_param_dicts:
                self.study.enqueue_trial(param_dict)

    def ask(self):
        trial = self.study.ask(self.distributions)
        param_dict = trial.params
        # FIXME: 如果重复了
        repeat = self.max_repeat
        while str(param_dict) in self.param_dict_trial_dict and repeat > 0:
            param_dict = self.random_study.ask(self.distributions).params
            self.study.enqueue_trial(param_dict)
            trial = self.study.ask(self.distributions)
            repeat -= 1
        if repeat != self.max_repeat:
            logging.warning(f"Randomly choose param_dict {self.max_repeat - repeat} times!")
        if repeat == 0:
            raise Exception('All params have been tried!')

        self.param_dict_trial_dict[str(param_dict)] = trial
        return param_dict

    def tell(self, param_dict, score):
        trial = self.param_dict_trial_dict[str(param_dict)]
        self.study.tell(trial, score)

    def report(self, param_dict, intermediate_value, step):
        trial = self.param_dict_trial_dict[str(param_dict)]
        trial.report(intermediate_value, step)

    def should_prune(self, param_dict):
        trial = self.param_dict_trial_dict[str(param_dict)]
        return trial.should_prune()