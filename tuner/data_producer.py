from utils.transforms import Trimmer, Aligner, Denoiser, Inputer, Warper, Differentiator, Sampler, Normalizer, Decomposer
import logging
import torch

class Data_Producer:
    def __init__(self, args, param_dict, data):
        self.args = args
        self.param_dict = param_dict

        label_len = self.args.label_len
        pred_len = self.args.pred_len
        trimmer_seq_len = param_dict['trimmer_seq_len']
        inputer_detect_method, inputer_fill_method =self.param_dict['inputer_detect_method'], self.param_dict['inputer_fill_method']
        denoiser_method = self.param_dict['denoiser_method']
        clip_factor, warper_method= self.param_dict['clip_factor'], self.param_dict['warper_method']
        decomposer_period,  decomposer_components= self.param_dict['decomposer_period'], self.param_dict['decomposer_components']
        differentiator_n = self.param_dict['differentiator_n']
        sampler_factor = self.param_dict['sampler_factor']
        aligner_mode, aligner_method = self.param_dict['aligner_mode'], self.param_dict['aligner_method']

        self.aligner = Aligner(aligner_mode, aligner_method, self.args.patch_len)
        self.sampler = Sampler(sampler_factor)
        self.differentiator = Differentiator(differentiator_n, clip_factor)
        self.decomposer = Decomposer(decomposer_period, decomposer_components)
        self.warper = Warper(warper_method, clip_factor)
        self.denoiser = Denoiser(denoiser_method)
        self.trimmer = Trimmer(trimmer_seq_len, pred_len)
        self.inputer = Inputer(inputer_detect_method, inputer_fill_method, data.numpy())
        self.normalizer = None
        self.clip_factor = clip_factor

    def produce(self, data):
        pipeline_name = self.param_dict['pipeline_name']
        logging.info(f"pipeline_name={pipeline_name}")
        data = data.numpy()
        if pipeline_name == "produce1":
            return torch.from_numpy(self.produce1(data))
        elif pipeline_name == "produce2":
            return torch.from_numpy(self.produce2(data))
        else:
            raise ValueError(f"pipeline_name={pipeline_name} not supported!")
        
    def deproduce(self, data):
        pipeline_name = self.param_dict['pipeline_name']
        logging.info(f"pipeline_name={pipeline_name}")
        data = data.numpy()
        if pipeline_name == "produce1":
            return torch.from_numpy(self.deproduce1(data))
        elif pipeline_name == "produce2":
            return torch.from_numpy(self.deproduce2(data))
    
    def produce1(self, data):
        data = self.trimmer.pre_process(data)
        # logging.info(f"seq_after_trimmer.shape={data.shape}")
        data = self.inputer.pre_process(data)
        # logging.info(f"seq_after_inputer.shape={data.shape}")
        data = self.denoiser.pre_process(data)
        # logging.info(f"seq_after_denoiser.shape={data.shape}")
        data = self.warper.pre_process(data)
        # logging.info(f"seq_after_warper.shape={data.shape}")
        data = self.decomposer.pre_process(data)
        # logging.info(f"seq_after_decomposer.shape={data.shape}")
        data = self.differentiator.pre_process(data)
        # logging.info(f"seq_after_differentiator.shape={data.shape}")

        normalizer_method, normalizer_mode, normalizer_ratio =\
            self.param_dict['normalizer_method'], self.param_dict['normalizer_mode'], self.param_dict['normalizer_ratio']
        self.normalizer = Normalizer(normalizer_method, normalizer_mode, data, normalizer_ratio, self.clip_factor)
        
        data = self.normalizer.pre_process(data)
        data = self.sampler.pre_process(data)
        # logging.info(f"seq_after_sampler.shape={data.shape}")
        data = self.aligner.pre_process(data)

        logging.info(f"seq_after_preprocess.shape={data.shape}")
        seq_after_preprocess = data
        return seq_after_preprocess
    
    def deproduce1(self, data):
        data = self.aligner.post_process(data)
        data = self.sampler.post_process(data)
        data = self.normalizer.post_process(data)
        data = self.differentiator.post_process(data)
        data = self.decomposer.post_process(data)
        data = self.warper.post_process(data)
        data = self.denoiser.post_process(data)
        data = self.inputer.post_process(data)
        data = self.trimmer.post_process(data)

        assert data.shape[1] == self.args.pred_len, f"data.shape[1]={data.shape[1]}, pred_len={self.args.pred_len}"
        return data

    def produce2(self, data):
        # 顺序：trimmer, sampler, inputer, denoiser, warper, decomposer, differentiator, normalizer, aligner, model
        data = self.trimmer.pre_process(data)
        # logging.info(f"seq_after_trimmer.shape={data.shape}")
        data = self.sampler.pre_process(data)
        # logging.info(f"seq_after_sampler.shape={data.shape}")
        data = self.inputer.pre_process(data)
        # logging.info(f"seq_after_inputer.shape={data.shape}")
        data = self.denoiser.pre_process(data)
        # logging.info(f"seq_after_denoiser.shape={data.shape}")
        data = self.warper.pre_process(data)
        # logging.info(f"seq_after_warper.shape={data.shape}")
        data = self.decomposer.pre_process(data)
        # logging.info(f"seq_after_decomposer.shape={data.shape}")
        data = self.differentiator.pre_process(data)
        # logging.info(f"seq_after_differentiator.shape={data.shape}")

        normalizer_method, normalizer_mode, normalizer_ratio =\
            self.param_dict['normalizer_method'], self.param_dict['normalizer_mode'], self.param_dict['normalizer_ratio']
        self.normalizer = Normalizer(normalizer_method, normalizer_mode, data, normalizer_ratio, self.clip_factor)

        data = self.normalizer.pre_process(data)
        data = self.aligner.pre_process(data)

        logging.info(f"seq_after_preprocess.shape={data.shape}")
        seq_after_preprocess = data
        return seq_after_preprocess

    def deproduce2(self, data):
        data = self.aligner.post_process(data)
        data = self.normalizer.post_process(data)
        data = self.differentiator.post_process(data)
        data = self.decomposer.post_process(data)
        data = self.warper.post_process(data)
        data = self.denoiser.post_process(data)
        data = self.inputer.post_process(data)
        data = self.sampler.post_process(data)
        data = self.trimmer.post_process(data)

        assert data.shape[1] == self.args.pred_len, f"data.shape[1]={data.shape[1]}, pred_len={self.args.pred_len}"
        return data