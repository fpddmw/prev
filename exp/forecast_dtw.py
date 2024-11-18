import os
import time
import warnings

import heapq
import multiprocessing
import itertools

from dtaidistance import dtw
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
import multiprocessing as mp
warnings.filterwarnings('ignore')

def spl_dtw_task(train_loader, batch_x, top_k, cur_index):
    
    folder_path = './dtw_data'

    data_queue = mp.Queue()
    result_queue = mp.Queue()
    num_workers = 12
    processes = []

    print(f"starting {num_workers} multi process:")
    for _ in range(num_workers):
        p = mp.Process(target=recommand_serieses, args=(data_queue, result_queue, batch_x, top_k))
        p.start()
        processes.append(p)
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        data_queue.put((batch_x, batch_y, batch_x_mark, batch_y_mark))
    for _ in range(num_workers):
        data_queue.put(None)
    for p in processes:
        p.join()
    # print("main stream restart!")
    results = []
    while not result_queue.empty():
        result = result_queue.get()
        results.append(result)


    print(f"get {len(results)} results!")
    winner_group = []
    for result in results:
        for winner in zip(*result):
            winner_group.append(winner)
    winner_group = sorted(winner_group, key=lambda x:x[4])
    print(f"winner_group len:{len(winner_group)}")

    ans_x = []
    ans_y = []
    ans_x_mark = []
    ans_y_mark = []
    dis_list = []
    for i in range(top_k):
        ans_x.append(winner_group[i][0])
        ans_y.append(winner_group[i][1])
        ans_x_mark.append(winner_group[i][2])
        ans_y_mark.append(winner_group[i][3])
        dis_list.append(winner_group[i][4])

    print(f"get {top_k} serieses successfully!")
    print(f"distance for each seq: {dis_list}")

    torch.save(tuple(ans_x), f'{folder_path}/batch_x/batch_x_{cur_index}.pt')
    torch.save(tuple(ans_y), f'{folder_path}/batch_y/batch_y_{cur_index}.pt')
    torch.save(tuple(ans_x_mark), f'{folder_path}/batch_xmark/batch_xmark_{cur_index}.pt')
    torch.save(tuple(ans_y_mark), f'{folder_path}/batch_ymark/batch_xmark_{cur_index}.pt')
    torch.save(tuple(dis_list), f'{folder_path}/dis_list/dis_list_{cur_index}.pt')
    print(f"batch-{cur_index}: results written into files!")
    return ans_x, ans_y, ans_x_mark, ans_y_mark, dis_list

def recommand_serieses(data_queue, result_queue, test_each, top_k):
    # x, y = Series_preducer(data, samp_len, stride)
    # x_train, x_test, y_train, y_test  = train_test_split(x, y, train_size=0.8, random_state=42)
    # print('start 1')
    test_each = test_each[0,:,0]
    near_series_x = []
    near_series_y = []
    dis_heap = []
    i = 0
    while True:
        batch = data_queue.get()
        if batch is None:
            # print("dtw calculations finished!")
            break
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        xtrain_each = batch_x[0,:,0]
        dis = dtw.distance_fast(test_each.numpy(), xtrain_each.numpy())
        if(i < top_k):
            near_series_x.append((batch_x, batch_x_mark))
            near_series_y.append((batch_y,batch_y_mark))
            heapq.heappush(dis_heap, (-dis, len(dis_heap)))
        else:
            if(dis < -dis_heap[-1][0]):
                top_index = dis_heap[0][1]
                near_series_x[top_index] = (batch_x, batch_x_mark)
                near_series_y[top_index] = (batch_y,batch_y_mark)
                heapq.heapreplace(dis_heap, (-dis, top_index))
        i += 1
    ans_x, ans_y, ans_x_mark, ans_y_mark, dis_list = [], [], [], [], []
    for i in range(top_k):
        dis, top_index = heapq.heappop(dis_heap)
        ans_x.append(near_series_x[top_index][0])
        ans_y.append(near_series_y[top_index][0])
        ans_x_mark.append(near_series_x[top_index][1])
        ans_y_mark.append(near_series_y[top_index][1])
        dis_list.append(-dis)
    result = (ans_x, ans_y, ans_x_mark, ans_y_mark, dis_list)
    result_queue.put(result)

    result_queue.close()
    result_queue.join_thread()
    # print("return to main stream!")
    return

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

    def test(self, setting, test=0):

        top_k = 10

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + '/' + f'{self.args.output_len}/' 
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        preds_list_dtw = [[] for _ in range(len(self.args.output_len_list))]
        preds_list_cat = [[] for _ in range(len(self.args.output_len_list))]
        rec_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()


        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):

                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    #dtw data
                    train_data, train_loader = data_provider(self.args, flag='dtw_train')

                    near_series_x, near_series_y, near_series_xmark, near_series_ymark, dis_list =\
                        spl_dtw_task(train_loader, batch_x, top_k, i)

                    folder_dtw_path = './dtw_data'
                    # if not os.path.exists(f"{folder_dtw_path}/batch_x_{i}.pt"):
                    #     break
                    # near_series_x = torch.load(f'{folder_dtw_path}/batch_x_{i}.pt')[:top_k]
                    # near_series_y = torch.load(f'{folder_dtw_path}/batch_y_{i}.pt')[:top_k]
                    # near_series_xmark = torch.load(f'{folder_dtw_path}/batch_xmark_{i}.pt')[:top_k]
                    # near_series_ymark = torch.load(f'{folder_dtw_path}/batch_ymark_{i}.pt')[:top_k]
                    
                    select_averages = torch.cat(tuple([items.unsqueeze(0) for items in near_series_x])).mean(dim=0)
                    print(f"ave shape:{select_averages.shape}")
                    batch_x_dtw = (select_averages + batch_x) / 2
                    print(f"batch_x_dtw shape:{batch_x_dtw.shape}")
                    print(f"dtw_dis:{dtw.distance_fast(batch_x[0,:,0].numpy(), batch_x_dtw[0,:,0].numpy())}")
                    rec_seq = torch.cat(tuple([items.unsqueeze(0) for items in near_series_y])).mean(dim=0)
                    
                    batch_x = batch_x.float().to(self.device)
                    batch_x_dtw = batch_x_dtw.float().to(self.device) ###
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    pred_y_dtw = []
                    pred_y_cat = []

                    ###near series concat operations###

                    batch_x_cat = torch.cat(near_series_x, dim=1).float().to(self.device)
                    batch_xmark_cat = torch.cat(near_series_xmark, dim=1).float().to(self.device)
                    batch_y_cat = torch.cat(near_series_y, dim=1).float().to(self.device)
                    batch_ymark_cat = torch.cat(near_series_ymark, dim=1).float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    batch_x_cat = torch.cat([batch_x_cat, batch_x], dim=1)
                    batch_xmark_cat = torch.cat([batch_xmark_cat, batch_x_mark], dim=1)
                    batch_ymark_cat = torch.cat([batch_ymark_cat, batch_y_mark], dim=1)

                    dec_inp_cat = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp_cat = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    dec_inp_cat = torch.cat([batch_y_cat, dec_inp], dim=1).float()

                    # print(f"dec_inp_cat len:{dec_inp_cat.shape}")
                    # print(f"batch_x_cat len:{batch_x_cat.shape}")
                    # print(f"batch_ymark_cat len:{batch_ymark_cat.shape}")
                    # break

                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            batch_x_dtw = torch.cat([batch_x_dtw[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        if self.args.output_attention:
                            outputs, attns, dec_out_save = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_dtw, attns, dec_out_save_dtw = self.model(batch_x_dtw, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_cat, attns, dec_out_save_cat = self.model(batch_x_cat, batch_xmark_cat, dec_inp_cat, batch_ymark_cat)                            
                        else:
                            outputs, dec_out_save = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_dtw, dec_out_save_dtw = self.model(batch_x_dtw, batch_x_mark, dec_inp, batch_y_mark)
                            outputs_cat, dec_out_save_cat = self.model(batch_x_cat, batch_xmark_cat, dec_inp_cat, batch_ymark_cat)                            

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                        pred_y_dtw.append(outputs_dtw[:, -self.args.pred_len:, :])
                        pred_y_cat.append(outputs_cat[:, -self.args.pred_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)
                    pred_y_dtw = torch.cat(pred_y_dtw, dim=1)
                    pred_y_cat = torch.cat(pred_y_cat, dim=1)


                    # print(f"pred_y:{pred_y[:, :10, :]}")
                    # print(f"pred_y_dtw:{pred_y_dtw[:, :10, :]}")
                    # print(f"pred_y_cat:{pred_y_cat[:, :10, :]}")
                    # print(f"pred_y shape:{pred_y.shape}")
                    # print(f"pred_y_dtw shape:{pred_y_dtw.shape}")
                    # print(f"pred_y_cat shape:{pred_y_cat.shape}")


                    if dis != 0:
                        pred_y = pred_y[:, :-dis, :]
                        pred_y_dtw = pred_y_dtw[:, :-dis, :]
                        pred_y_cat = pred_y_cat[:, :-dis, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                        rec_seq = rec_seq[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                        rec_seq = rec_seq[:, :self.args.output_len, :].to(self.device)
                    outputs = pred_y.detach().cpu()
                    outputs_dtw = pred_y_dtw.detach().cpu()
                    outputs_cat = pred_y_cat.detach().cpu()
                    batch_y = batch_y.detach().cpu()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        outputs_dtw = test_data.inverse_transform(outputs_dtw.squeeze(0)).reshape(shape)
                        outputs_cat = test_data.inverse_transform(outputs_cat.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                        rec_seq = test_data.inverse_transform(rec_seq.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    outputs_dtw = outputs_dtw[:, :, f_dim:]
                    outputs_cat = outputs_cat[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    rec_seq = rec_seq[:, :, f_dim:]

                    pred = outputs
                    pred_dtw = outputs_dtw
                    pred_cat = outputs_cat
                    true = batch_y

                    preds_list[output_ptr].append(pred)
                    preds_list_dtw[output_ptr].append(pred_dtw)
                    preds_list_cat[output_ptr].append(pred_cat)
                    trues_list[output_ptr].append(true)
                    rec_list[output_ptr].append(rec_seq.cpu())
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                        pd_dtw = np.concatenate((input[0, -self.args.pred_len:, -1], pred_dtw[0, :, -1]), axis=0)
                        pd_cat = np.concatenate((input[0, -self.args.pred_len:, -1], pred_cat[0, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))

                            visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))
                            visual(gt, pd_dtw, os.path.join(folder_path, f'{i}_{self.args.local_rank}_dtw.pdf'))
                            visual(gt, pd_cat, os.path.join(folder_path, f'{i}_{self.args.local_rank}_cat.pdf'))                            

        if self.args.output_len_list is not None:
            for i in range(len(preds_list)):
                preds = preds_list[i]
                preds_dtw = preds_list_dtw[i]
                preds_cat = preds_list_cat[i]
                trues = trues_list[i]
                rec_seq = rec_list[i]

                preds = torch.cat(preds, dim=0).numpy()
                preds_dtw = torch.cat(preds_dtw, dim=0).numpy()
                preds_cat = torch.cat(preds_cat, dim=0).numpy()                
                trues = torch.cat(trues, dim=0).numpy()
                rec_seq = torch.cat(rec_seq, dim=0).numpy()


                mae, mse, rmse, mape, mspe = metric(preds, trues)
                mae_dtw, mse_dtw, rmse_dtw, mape_dtw, mspe_dtw = metric(preds_dtw, trues)
                mae_cat, mse_cat, rmse_cat, mape_cat, mspe_cat = metric(preds_cat, trues)
                mae_rec, mse_rec, rmse_rec, mape_rec, mspe_rec = metric(rec_seq, trues)
                print(f"output_len: {self.args.output_len_list[i]}")
                print('mse:{}, mae:{}'.format(mse, mae))
                print('mse_dtw:{}, mae_dtw:{}'.format(mse_dtw, mae_dtw))
                print('mse_cat:{}, mae_cat:{}'.format(mse_cat, mae_cat))
                print('mse_rec:{}, mae_rec:{}'.format(mse_rec, mae_rec))


                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('mse_dtw:{}, mae_dtw:{}'.format(mse_dtw, mae_dtw))
                f.write('mse_cat:{}, mae_cat:{}'.format(mse_cat, mae_cat))
                f.write('mse_rec:{}, mae_rec:{}'.format(mse_rec, mae_rec))
                f.write('\n')
                f.write('\n')
                f.close()
        return
