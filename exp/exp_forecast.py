import os
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from exp.exp_tuner import Exp_Tuner

import csv

warnings.filterwarnings('ignore')

def csv_init(dataname, param_dicts, seq_len, flag):
    folder = 'result' + '/' + dataname + '/'
    os.makedirs(folder, exist_ok=True)
    file_list = {'result': f"{folder}pred_result_{dataname}_{seq_len}.csv", 
                 'rev_result': f"{folder}pred_rev_result_{dataname}_{seq_len}.csv",
                 'hyper_params': f"{folder}params_{dataname}_{seq_len}.csv",
                 'result_record': f"{folder}result_record_{dataname}_{seq_len}.csv"}
    with open(file_list['result_record'], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['', 'mse', 'mae', 'mse_p', 'mae_p', 'best_idx'])
    with open(file_list['result'], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['', 'none'] + [str(param_dict) for param_dict in param_dicts])
    with open(file_list['rev_result'], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['', 'none'] + [str(param_dict) for param_dict in param_dicts])
    if flag:
        with open(file_list['hyper_params'], 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['', 'none'] + [ str(param_dict) for param_dict in param_dicts])
    return file_list



class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = criterion(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = criterion(pred, true)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)


        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def _pred(self, test_data, data_tuner, dec_inp, batch_x, batch_y, batch_x_mark, batch_y_mark):
        
        inference_steps = self.args.output_len // self.args.pred_len
        dis = self.args.output_len - inference_steps * self.args.pred_len
        if dis != 0:
            inference_steps += 1

        pred_y = []
        for j in range(inference_steps):
            if len(pred_y) != 0:
                batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                tmp = batch_y_mark[:, j - 1:j, :]
                batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

            if self.args.output_attention:
                outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                        
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            print(f"outputs shape after pred:{outputs.shape} dis:{dis}")
            pred_y.append(outputs[:, -self.args.pred_len:, :])
        pred_y = torch.cat(pred_y, dim=1)

        if dis != 0:
            pred_y = pred_y[:, :-dis, :]

        if self.args.use_ims:
            batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                self.device)
        else:
            batch_y = batch_y[:, :self.args.output_len, :].to(self.device)

        outputs = pred_y.detach().cpu()
        batch_y = batch_y.detach().cpu()

        print(f"outputs shape:{outputs.shape}")
        outputs = torch.cat([outputs[0:1], data_tuner.get_result(outputs[1:, : , :])])    ##修改##
        print(f"outputs shape after produce:{outputs.shape}")

        batch_y = batch_y.detach().cpu()

        if test_data.scale and self.args.inverse:
            shape = outputs.shape                        
            outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)                        
        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        pred = outputs
        true = batch_y
        return pred, true

    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        preds_list_rev = [[] for _ in range(len(self.args.output_len_list))]
        trues_list_rev = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()
        
        ##预训练tuner###
        train_tuner_flag = self.args.train_tuner ###
        if(train_tuner_flag == True):
            train_data, train_loader = data_provider(self.args, flag='train')
            data_tuner = Exp_Tuner(self.args, self.device, flag = 'train')
            data_tuner.train_or_tuning_tuner(train_data, train_loader, self.model, flag='train')
            param_dicts = data_tuner.param_dicts
            print(f"get {len(param_dicts)} params!")
        else:
            import ast
            import pandas as pd
            df = pd.read_csv('params_ETTh1_288.csv')
            df = df.iloc[:, 2:]

            param_dicts = []
            for col in df.columns:
                dict_cell = ast.literal_eval(col)  # 读取每一列的第一个元素并转化为字典
                param_dicts.append(dict_cell)
            print(f"read {len(param_dicts)} params!")

            data_tuner = Exp_Tuner(self.args, self.device, flag = 'tuning')
            data_tuner.param_dicts = param_dicts
        file_list = csv_init(self.args.data, param_dicts, seq_len=self.args.seq_len, flag=train_tuner_flag)


        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test', rev_flag=True)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, \
                        batch_x_rev, batch_y_rev, batch_x_mark_rev, batch_y_mark_rev) in enumerate(test_loader):
                    batch_y = batch_y.float().to(self.device)
                    batch_y_rev = batch_y_rev.float().to(self.device)
                    ## batch_x和batch_x_rev没有传入self.device ##
                    
                    print(i)
                    print(f"x:{batch_x.shape}")
                    print(f"y:{batch_y.shape}")
                    print(f"revx:{batch_x_rev.shape}")
                    print(f"revy:{batch_y_rev.shape}")

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    dec_inp = torch.cat([dec_inp for _ in range(len(param_dicts))]).to(self.device)
                    print(f"dec_inp shape:{dec_inp.shape}")

                    dec_inp_rev = torch.zeros_like(batch_y_rev[:, -self.args.pred_len:, :]).float()
                    dec_inp_rev = torch.cat([batch_y_rev[:, :self.args.label_len, :], dec_inp_rev], dim=1).float()
                    dec_inp_rev = torch.cat([dec_inp_rev for _ in range(len(param_dicts))]).to(self.device)
                    print(f"dec_inp_rev shape:{dec_inp_rev.shape}")

                    batch_x_mark = torch.cat([batch_x_mark for _ in range(len(param_dicts))])        
                    batch_y_mark = torch.cat([batch_y_mark for _ in range(len(param_dicts))])
                    print(f"batch_y_mark shape:{batch_y_mark.shape}")

                    batch_x_mark_rev = torch.cat([batch_x_mark_rev for _ in range(len(param_dicts))])                  
                    batch_y_mark_rev = torch.cat([batch_y_mark_rev for _ in range(len(param_dicts))])
                    print(f"batch_y_mark_rev shape:{batch_y_mark_rev.shape}")
                    
                    ##生成测试数据##
                    batch_x_pro = data_tuner.get_batches(batch_x)
                    batch_x_pro = torch.cat([batch_x, batch_x_pro])
                    print(f"batch_x_pro shape:{batch_x_pro.shape}")
                    batch_x = batch_x_pro.float().to(self.device)

                    batch_x_rev_pro = data_tuner.get_batches(batch_x_rev)
                    batch_x_rev_pro = torch.cat([batch_x_rev, batch_x_rev_pro])
                    print(f"batch_x_rev_pro shape:{batch_x_rev_pro.shape}")
                    batch_x_rev = batch_x_rev_pro.float().to(self.device)  ##直接进行覆盖##

                    ##使用反向数据得到最优超参数
                    pred_rev, true_rev = self._pred(test_data, data_tuner,\
                                            dec_inp_rev, batch_x_rev, batch_y_rev, batch_x_mark_rev, batch_y_mark_rev)
                    print(f"pred_rev shape:{pred_rev.shape}")
                    mse_rev = []
                    mae_rev = []
                    for line_index, aoutputs_rev in enumerate(pred_rev):
                        mse_rev.append(float(torch.mean((aoutputs_rev - true_rev) ** 2)))
                        mae_rev.append(float(torch.mean(torch.abs(aoutputs_rev - true_rev))))
                    with open(file_list['rev_result'], 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([i] + mse_rev)

                    ##记录最优idx##
                    best_idx_param = mse_rev.index(min(mse_rev))
                    best_idx_data = best_idx_param
                    print(f"best_idx:{best_idx_param},\
                           param_dict:{'none' if best_idx_param == 0 else param_dicts[best_idx_param - 1]}")

                    ##预测正向##
                    pred, true = self._pred(test_data, data_tuner,\
                                            dec_inp, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    print(f"pred shape:{pred.shape}")
                    mse = []
                    mae = []
                    for line_index, aoutputs in enumerate(pred):
                        mse.append(float(torch.mean((aoutputs - true) ** 2)))
                        mae.append(float(torch.mean(torch.abs(aoutputs - true))))
                    with open(file_list['result'], 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([i] + mse)

                    ##记录baseline##
                    with open(file_list['result_record'], 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([i, mse[0], mae[0], mse[best_idx_data], mae[best_idx_data], best_idx_data])

                    preds_list[output_ptr].append(pred)
                    trues_list[output_ptr].append(true)
                    preds_list_rev[output_ptr].append(pred_rev)
                    trues_list_rev[output_ptr].append(true_rev)
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        input_rev = batch_x_rev.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                        gt_p = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd_p = np.concatenate((input[0, -self.args.pred_len:, -1], pred[best_idx_data, :, -1]), axis=0)

                        gt_rev = np.concatenate((input_rev[0, -self.args.pred_len:, -1], true_rev[0, :, -1]), axis=0)
                        pd_rev = np.concatenate((input_rev[0, -self.args.pred_len:, -1], pred_rev[0, :, -1]), axis=0)
                        gt_rev_p = np.concatenate((input_rev[0, -self.args.pred_len:, -1], true_rev[0, :, -1]), axis=0)
                        pd_rev_p = np.concatenate((input_rev[0, -self.args.pred_len:, -1], pred_rev[best_idx_data, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))

                            visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))
                            visual(gt_rev, pd_rev, os.path.join(folder_path, f'{i}_{self.args.local_rank}_rev.pdf'))
                            visual(gt_p, pd_p, os.path.join(folder_path, f'{i}_{self.args.local_rank}_p.pdf'))
                            visual(gt_rev_p, pd_rev_p, os.path.join(folder_path, f'{i}_{self.args.local_rank}_rev_p.pdf'))                            

        if self.args.output_len_list is not None:
            for i in range(len(preds_list)):
                preds = preds_list[i]
                trues = trues_list[i]
                preds_rev = preds_list_rev[i]
                trues_rev = trues_list_rev[i]

                preds = torch.cat(preds, dim=1).numpy()
                true = torch.cat(trues, dim=1).numpy()
                preds_rev = torch.cat(preds_rev, dim=1).numpy()
                true_rev = torch.cat(trues_rev, dim=1).numpy()

                print(f"preds shape:{preds.shape}")
                print(f"true shape:{true.shape}")

                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                for param_idx, (pred, pred_rev) in enumerate(zip(preds, preds_rev)):
                    
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    mae_rev, mse_rev, rmse_rev, mape_rev, mspe_rev = metric(pred_rev, true_rev)

                    print(param_idx)
                    print('mse:{}, mae:{}'.format(mse, mae))
                    print('mse_rev:{}, mae_rev:{}'.format(mse_rev, mae_rev))
                    f.write(str(param_idx) + "\n")
                    f.write('mse:{}, mae:{}'.format(mse, mae))
                    f.write('\n')
                    f.write('mse_rev:{}, mae_rev:{}'.format(mse_rev, mae_rev))
                    f.write('\n')
                f.write('\n')
                f.close()

        return()
