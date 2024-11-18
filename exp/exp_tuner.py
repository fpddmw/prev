from tuner.tuner_options import get_params_space_and_org
from tuner.tuner_terminator import TerminatorManager, MaxIterationsTerminator, NoImprovementTerminator
from tuner.tuner_options import OptunaTuner
from utils.transforms import nan_inf_clip_factor, my_clip
from tuner.data_producer import Data_Producer
from utils.metrics import metric
from tqdm import tqdm
import logging
from math import ceil
import numpy as np
import torch
import heapq


class Exp_Tuner:
    def __init__(self, args, device, flag = 'tuning'):
        self.args = args
        params_space, param_dict = get_params_space_and_org(self.args.seq_len, self.args.patch_len)
        self.params_space = params_space
        self.device = device
        self.param_dicts = []
        self.data_producers = []
        self.terminator_manager = TerminatorManager([  # FIXME
        # TimeLimitTerminator(60 * 10 if 'Chronos' in model_name else 60 * 5),
        MaxIterationsTerminator(self.args.train_echos if flag == 'train' else self.args.tuning_echos),
        # NoImprovementTerminator('minimize', 0, 30),
        # RelativeImprovementTerminator('minimize', 0, 20, 40)
        ], 0, 0)
        self.tuner = OptunaTuner(self.args, params_space, 'maximize', [param_dict])
        
        logging.info(f"params_space={params_space}")
        logging.info(f"origin_param_dict={param_dict}")

        assert param_dict.keys() == params_space.keys(), \
        f"param_dict.keys()={param_dict.keys()}, params_space.keys()={params_space.keys()}"
        for key in param_dict.keys():
            if isinstance(param_dict[key], str):
                assert param_dict[key] in params_space[key]['values'], \
                f"param_dict[{key}]={param_dict[key]}, params_space[{key}]['values']={params_space[key]['values']}"

    def train_or_tuning_tuner(self, dataset, dataloader, model, flag):
        params_result = []
        
        if flag =='train' :
            echos = self.args.train_echos
        else:
            echos = self.args.tuning_echos

        bar1 = tqdm(range(echos), desc='Processing Params', ncols = 100)
        for _ in bar1:
            param_dict = self.tuner.ask()
            logging.info(f"param_dict={param_dict}")
            max_step_idx = len(dataloader) - 1
            should_prune = False
            batch_mse_list = []  # !!!!!! 存储当前param_dict的mse列表，无论是否被prune都计算的全split_idx的mse
            
            # bar2 = tqdm(enumerate(dataloader), desc='Processing Split Idxes', ncols=100, total=len(dataloader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                logging.info(f"step ratio: {i + 1}/{max_step_idx + 1}")
                train_producer = Data_Producer(self.args, param_dict, batch_x)

                # batch_x, batch_y = batch_x.numpy(), batch_y.numpy()
                batch_x_pro = train_producer.produce(batch_x).float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                inference_steps = self.args.output_len // self.args.pred_len
                dis = self.args.output_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x_pro = torch.cat([batch_x_pro[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.output_attention:
                        outputs, attns = model(batch_x_pro, batch_x_mark, dec_inp, batch_y_mark)
                                                    
                    else:
                        outputs = model(batch_x_pro, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)

                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]

                if self.args.use_ims:
                    batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                else:
                    batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                outputs = train_producer.deproduce(outputs)
                if dataset.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = dataset.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = dataset.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:].numpy()
                batch_y = batch_y[:, :, f_dim:].numpy()

                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)
                batch_mse_list.append(mse)

                result_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
                logging.info(f"result:{result_dict}")

                self.tuner.report(param_dict, mse, i)
                if self.tuner.should_prune(param_dict):
                    logging.info(f"Prune at step_idx={i} mse={mse}")
                    should_prune = True
                
                if should_prune:
                    break
            # FIXME：无论是否被prune，都要告诉tuner这个param_dict已经结束了.这里的mean_mse在prune后不是在所有的split_idx上计算的！！！
            # FIXME：更高保真的mean_mse计算方式 -> 所有过去历史batch的全量split_idx的mse的mean
            # if should_prune:
            final_mse = np.mean(batch_mse_list)
            self.tuner.tell(param_dict, final_mse)

            logging.info(f"final_mse={final_mse} with pramas:{param_dict}")
            print(f"final_mse={final_mse} with pramas:{param_dict}")
            terminate_flag = self.terminator_manager.update_and_check(final_mse, should_prune)
            if should_prune is not True:
                params_result.append((final_mse, param_dict))

            if terminate_flag:
                logging.info(f"Experiment terminated by terminator!")
                break
        params_result.sort(key=lambda x: x[0])

        logging.info(f"Select {len(params_result[:self.args.num_params])} params!")
        if flag =='train':
            params_dicts = [param[1] for param in params_result[:self.args.num_params]]
            self.param_dicts = params_dicts
        return params_result
    
    def vali_tuner(self, dataset, dataloader, model):
        params_result = self.train_or_tuning_tuner(dataset, dataloader, model, flag = 'tune')
        bar1 = tqdm((self.param_dicts), desc='validating Params', ncols = 100)
        for param_dict in bar1:
            # bar2 = tqdm(enumerate(dataloader), desc='Processing Split Idxes', ncols=100, total=len(dataloader))
            batch_mse_list = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                train_producer = Data_Producer(self.args, param_dict, batch_x)

                # batch_x, batch_y = batch_x.numpy(), batch_y.numpy()
                batch_x_pro = train_producer.produce(batch_x).float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                inference_steps = self.args.output_len // self.args.pred_len
                dis = self.args.output_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x_pro = torch.cat([batch_x_pro[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.output_attention:
                        outputs, attns = self.model(batch_x_pro, batch_x_mark, dec_inp, batch_y_mark)
                                                    
                    else:
                        outputs = self.model(batch_x_pro, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)

                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]

                if self.args.use_ims:
                    batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                else:
                    batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                outputs = train_producer.deproduce(outputs)
                if dataset.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = dataset.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = dataset.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                
                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)
                batch_mse_list.append(mse)

                result_dict = {**param_dict,'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
                logging.info(f"result:{result_dict}")
            final_mse = np.mean(batch_mse_list)
            params_result.append((final_mse, param_dict))
        params_result.sort(key=lambda x: x[0])

        logging.info(f"Select {len(params_result[:self.args.num_params])} params!")
        params_dicts = [param[1] for param in params_result[:self.args.num_params]]
        self.param_dicts = params_dicts
        return params_result

    def get_batches(self, data):
        self.data_producers = [Data_Producer(self.args, param_dict, data) for param_dict in self.param_dicts]
        produced_data = [data_producer.produce(data) for data_producer in self.data_producers ]
        return torch.cat(produced_data)

    def get_best_batches(self, data, best_idx):
        produced_data = self.data_producers[best_idx].produce(data)
        return produced_data
    
    def get_result(self, data):
        result = [self.data_producers[i].deproduce(data[i:i+1]) for i in range(len(data)) ]
        return torch.cat(result)

