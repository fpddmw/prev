import logging
from math import ceil

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from scipy.fft import ifft, fft
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from statsmodels.tsa.stl._stl import STL

nan_inf_clip_factor = 5

def smart_clip(seq, min_allowed, max_allowed, seq_in_last_values):
    assert seq.ndim == 3, "Input sequence must be 3D: (batch, time, feature)"
    assert min_allowed.shape == max_allowed.shape == (seq.shape[0], 1, seq.shape[2]), \
        "Min and max must have shape (batch, 1, feature)"

    batch, time, feature = seq.shape
    first_elements = seq[:, 0:1, :]  # Preserve the first elements
    # assert np.all(first_elements < max_values) and np.all(first_elements > min_values), \
    #     f"The first elements must be within min and max values: \n" \
    #     f"first_elements:{first_elements}, \nmin_values:{min_values}, \nmax_values:{max_values}"
    # FIXME:
    if np.any(first_elements > max_allowed) or np.any(first_elements < min_allowed):
        logging.info(f"The first elements must be within min and max allowed!!!\n")
        logging.debug(f"The first elements must be within min and max allowed: \n"
                      f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
        # return np.clip(seq, min_allowed, max_allowed)
        # FIXME：平移first使得跟last一样,即在(first,last)之间进行scale
        seq = seq - first_elements + seq_in_last_values

    # Calculate scaling factors for each batch
    seq_max_values = np.max(seq, axis=1, keepdims=True)  # Include the first element
    seq_min_values = np.min(seq, axis=1, keepdims=True)  # Include the first element

    # Apply scaling to the sequences that exceed the max values
    for i in range(batch):
        for j in range(feature):
            # 如果存在大于max的值，则把值介于(first,max)的数值进行scale
            tmp_seq = seq[i, :, j]
            first_value = first_elements[i, 0, j]
            max_value = max_allowed[i, 0, j]
            min_value = min_allowed[i, 0, j]
            seq_max_value = seq_max_values[i, 0, j]
            seq_min_value = seq_min_values[i, 0, j]
            if seq_max_value > max_value:
                scale = (max_value - first_value) / (seq_max_value - first_value)
                upper_mask = tmp_seq > first_value
                seq[i, upper_mask, j] = first_value + (seq[i, upper_mask, j] - first_value) * scale
            if seq_min_value < min_value:
                scale = (min_value - first_value) / (seq_min_value - first_value)
                lower_mask = tmp_seq < first_value
                seq[i, lower_mask, j] = first_value + (seq[i, lower_mask, j] - first_value) * scale
    return seq

def my_clip(seq_in, seq_out, nan_inf_clip_factor=None, min_max_clip_factor=None):
    # nan_inf_clip_factor=3, min_max_clip_factor=2 ...
    # mean+1.5IQR-> max+0.25range
    max_values = np.max(seq_in, axis=1, keepdims=True)
    min_values = np.min(seq_in, axis=1, keepdims=True)
    range_values = max_values - min_values

    assert nan_inf_clip_factor is not None or min_max_clip_factor is not None, \
        "nan_inf_clip_factor and min_max_clip_factor cannot be both None!"

    if nan_inf_clip_factor is not None and (np.isnan(seq_out).any() or np.isinf(seq_out).any()):
        max_allowed = max_values + nan_inf_clip_factor * range_values
        min_allowed = min_values - nan_inf_clip_factor * range_values
        logging.info(f"seq_out contains NaN values!!! \n")
        logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
        # seq_out = np.nan_to_num(seq_out, nan=(max_values + min_values) / 2, posinf=max_allowed, neginf=min_allowed)
        seq_out = np.nan_to_num(seq_out, nan=max_allowed, posinf=max_allowed, neginf=min_allowed)  # nan hard punish
        # logging.warning(f"seq_out after filling NaN values: {seq_out}")
    if min_max_clip_factor is not None:
        max_allowed = max_values + min_max_clip_factor * range_values
        min_allowed = min_values - min_max_clip_factor * range_values
        seq_in_last_values = seq_in[:, -1:, :]
        if (seq_out > max_allowed).any() or (seq_out < min_allowed).any():
            logging.info(f"seq_out out of range!!!: \n")
            logging.debug(f"seq_out out of range!!!: {seq_out}")
            # seq_out = np.clip(seq_out, min_allowed, max_allowed)
            # logging.warning(f"seq_out after clipping: {seq_out}")
            # FIXME：助长scale的气焰....危险 # 使用allowed！Ok ->对scale的处理还是有点问题？weizhi
            seq_out = smart_clip(seq_out, min_allowed, max_allowed, seq_in_last_values)
            # logging.warning(f"seq_out after smart scaling: {seq_out}")
    return seq_out


def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3 and data.shape[2] == 1, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'


# 值域的归一化
# minmax和maxabs都太差影响了整体效果（outlier)... -》现在Timer内置了scaler
class Normalizer:
    def __init__(self, method, mode, input_data, ratio, clip_factor):  # FIXME:ratio
        assert method in ['none', 'standard', 'minmax', 'maxabs', 'robust']
        assert mode in ['none', 'dataset', 'input', 'history']
        self.method = method
        self.mode = mode
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

        if mode == 'none' or method == 'none':
            return
        elif mode == 'input':
            data = input_data
            assert 0 < ratio <= 1, 'Invalid ratio: {}'.format(ratio)
            self.scaler_params = self._compute_scaler_params(data, ratio)
        else:
            raise Exception('Invalid normalizer mode: {}'.format(self.mode))

    def _compute_scaler_params(self, data, look_back_ratio):
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape

        look_back_len = int(time * look_back_ratio)

        self.data_in = data
        params = {}
        for i in range(feature):
            feature_data = data[:, :, i].reshape(batch, -1)[:, -look_back_len:]
            if self.method == 'standard':
                mean = np.mean(feature_data, axis=1, keepdims=True)
                std = np.std(feature_data, axis=1, keepdims=True)
                params[i] = (mean, std)
            elif self.method == 'minmax':
                min_val = np.min(feature_data, axis=1, keepdims=True)
                max_val = np.max(feature_data, axis=1, keepdims=True)
                params[i] = (min_val, max_val)
            elif self.method == 'maxabs':
                max_abs_val = np.max(np.abs(feature_data), axis=1, keepdims=True)
                params[i] = max_abs_val
            elif self.method == 'robust':
                median = np.median(feature_data, axis=1, keepdims=True)
                q1 = np.percentile(feature_data, 25, axis=1, keepdims=True)
                q3 = np.percentile(feature_data, 75, axis=1, keepdims=True)
                params[i] = (median, q1, q3)
        return params

    def pre_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        res = np.zeros_like(data)
        if self.mode == 'dataset':
            # 使用 sklearn 的 scaler 进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - mean) / (std + 1e-8)).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - min_val) / (max_val - min_val + 1e-8)).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data / (max_abs_val + 1e-8)).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - median) / (q3 - q1 + 1e-8)).reshape(batch, time)

        return res

    def post_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        res = np.zeros_like(data)
        if self.mode == 'dataset':
            # 使用 sklearn 的 scaler 进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.inverse_transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = (feature_data * std + mean).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (max_val - min_val) + min_val).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * max_abs_val).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (q3 - q1) + median).reshape(batch, time)

        if np.isnan(res).any() or np.isinf(res).any():
            logging.error(f"NaN or Inf values in restored data: {res}")
            res = my_clip(self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)  # 表示出结果很差
        elif self.clip_factor is not None:
            res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格

        return res


# 序列的分解方法(废弃
class Decomposer:
    def __init__(self, period, component_for_model, parallel=True):
        self.period = int(period) if period != 'none' else None
        self.component_for_model = component_for_model.split('+')
        self.none_flag = component_for_model == 'none' or self.period is None
        self.trend = None
        self.season = None
        self.residual = None
        self.parallel = parallel

    def pre_process(self, data):
        if self.none_flag:
            return data

        def _decompose_series1(series):
            # Period=365, Target=trend
            # Preprocess time: 2.588042974472046
            # Period=365, Target=season
            # Preprocess time: 0.3718416690826416
            # Period=365, Target=residual
            # Preprocess time: 0.3341329097747803
            # Period=60, Target=trend
            # Preprocess time: 0.09929013252258301
            # Period=60, Target=season
            # Preprocess time: 0.07264876365661621
            # Period=60, Target=residual
            # Preprocess time: 0.17017793655395508
            stl = STL(series, period=self.period, seasonal=13)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid

        def _decompose_series5(series):
            # Period=365, Target=trend
            # Preprocess time: 1.6404809951782227
            # Period=365, Target=season
            # Preprocess time: 0.0523838996887207
            # Period=365, Target=residual
            # Preprocess time: 0.05175185203552246
            # Period=60, Target=trend
            # Preprocess time: 0.051614999771118164
            # Period=60, Target=season
            # Preprocess time: 0.05163908004760742
            # Period=60, Target=residual
            # Preprocess time: 0.0512700080871582
            freqs = np.fft.fftfreq(len(series))
            fft_values = fft(series)
            trend_fft = fft_values.copy()
            trend_fft[np.abs(freqs) > 1 / self.period] = 0
            trend = ifft(trend_fft).real

            # Compute seasonal component using FFT high frequency components
            season_fft = fft_values.copy()
            season_fft[np.abs(freqs) <= 1 / self.period] = 0
            season = ifft(season_fft).real

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        def _decompose_series6(series):
            # Compute trend using linear regression
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)

            # Compute seasonal component by subtracting the trend and detrending the result
            detrended = series - trend
            # season = self._compute_seasonal(detrended, self.period)
            season = np.zeros_like(data)
            period = self.period
            for i in range(period):
                seasonal_mean = np.mean(data[i::period])
                season[i::period] = seasonal_mean

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        if self.parallel:
            # 使用 joblib 并行处理每个批次的数据
            results = Parallel(n_jobs=-1)(delayed(_decompose_series1)(data[i, :, 0]) for i in range(data.shape[0]))
            # 将结果分解成趋势、季节性和残差
            trends, seasons, residuals = zip(*results)
        else:
            trends, seasons, residuals = [], [], []
            for i in range(data.shape[0]):
                series = data[i, :, 0]  # Extract the series for each batch
                trend, seasonal, residual = _decompose_series5(series)
                # stl = STL(series, period=self.period, seasonal=13)
                # result = stl.fit()
                trends.append(trend)
                seasons.append(seasonal)
                residuals.append(residual)
        self.trend = np.array(trends).reshape(data.shape)
        self.season = np.array(seasons).reshape(data.shape)
        self.residual = np.array(residuals).reshape(data.shape)

        # Determine which components to return based on the target
        combined = np.zeros_like(data)
        if 'trend' in self.component_for_model:
            combined += self.trend
        if 'season' in self.component_for_model:
            combined += self.season
        if 'residual' in self.component_for_model:
            combined += self.residual

        return combined

    def post_process(self, pred):
        if self.none_flag:
            return pred
        pred_trend, pred_season, pred_residual = 0, 0, 0

        if 'trend' not in self.component_for_model:
            pred_trend = self._predict_trend(pred.shape[1])

        if 'season' not in self.component_for_model:
            pred_season = self._predict_season(pred.shape[1])

        if 'residual' not in self.component_for_model:
            pred_residual = self._predict_residual(pred.shape[1])

        return pred + pred_trend + pred_season + pred_residual

    def _predict_trend(self, pred_len):  # 用线性回归预测
        pred_trend = np.zeros((self.trend.shape[0], pred_len, 1))
        for i in range(self.trend.shape[0]):
            y = self.trend[i, :, 0]
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            future_x = np.arange(len(y), len(y) + pred_len).reshape(-1, 1)
            pred_trend[i, :, 0] = model.predict(future_x)
        return pred_trend

    # def _predict_trend(self, pred_len):
    #     pred_trend = np.zeros((self.trend.shape[0], pred_len, 1))
    #     for i in range(self.trend.shape[0]):
    #         y = self.trend[i, :, 0]
    #         x = np.arange(len(y))
    #         future_x = np.arange(len(y), len(y) + pred_len)
    #         pred_trend[i, :, 0] = np.interp(future_x, x, y)
    #     return pred_trend

    # def _predict_season(self, pred_len):  # 复制最近的单周期数据
    #     pred_season = np.zeros((self.season.shape[0], pred_len, 1))
    #     for i in range(self.season.shape[0]):
    #         y = self.season[i, :, 0]
    #         season_length = self.period
    #         pred_season[i, :, 0] = np.tile(y[-season_length:], int(np.ceil(pred_len / season_length)))[:pred_len]
    #     return pred_season

    def _predict_season(self, pred_len):
        pred_season = np.zeros((self.season.shape[0], pred_len, 1))
        for i in range(self.season.shape[0]):
            y = self.season[i, :, 0]
            n = len(y)
            f = fft(y)
            frequencies = np.fft.fftfreq(n)
            # 只保留重要的频率
            threshold = 0.1
            f[np.abs(frequencies) > threshold] = 0
            future_frequencies = np.fft.fftfreq(n + pred_len)
            future_f = np.zeros_like(future_frequencies, dtype=complex)
            future_f[:n] = f
            future_f = ifft(future_f).real
            pred_season[i, :, 0] = future_f[-pred_len:]
        return pred_season

    # def _predict_residual(self, pred_len):  # 滑动平均做预测值
    #     pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
    #     for i in range(self.residual.shape[0]):
    #         y = self.residual[i, :, 0]
    #         pred_residual[i, :, 0] = np.mean(y[-pred_len:])
    #     return pred_residual

    # def _predict_residual(self, pred_len):
    #     pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
    #     for i in range(self.residual.shape[0]):
    #         y = self.residual[i, :, 0]
    #         # 使用滑动平均模型进行预测
    #         model = ARMA(y, order=(0, 1))  # MA(1) 模型
    #         model_fit = model.fit(disp=False)
    #         forecast = model_fit.forecast(steps=pred_len)[0]
    #         pred_residual[i, :, 0] = forecast
    #     return pred_residual

    def _predict_residual(self, pred_len):
        pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
        for i in range(self.residual.shape[0]):
            y = self.residual[i, :, 0]
            # 使用最后一个窗口的均值进行预测
            mean_value = np.mean(y[-pred_len:])
            pred_residual[i, :, 0] = mean_value
        return pred_residual


# 序列的采样方法
class Sampler:
    def __init__(self, factor):
        self.factor = factor

    def pre_process(self, data):
        if self.factor == 1:
            return data
        assert_timeseries_3d_np(data)
        # FIXME:resample不一样！！！可能去均值？？？/ 更差了。。。
        # if int(self.factor) == self.factor:
        #     return data[:, ::int(self.factor), :]
        batch, time, feature = data.shape
        res = np.zeros((batch, ceil(time / self.factor), feature))
        for b in range(batch):
            for f in range(feature):
                res[b, :, f] = signal.resample(data[b, :, f], ceil(time / self.factor))
        # 用mean_pad补全原来的shape的time的部分 -> 补全前面而不是后面！！！
        # res = Aligner(time, 'mean_pad').pre_process(res)
        return res

    def post_process(self, data):
        if self.factor == 1:
            return data
        assert_timeseries_3d_np(data)
        batch, time, feature = data.shape
        res = np.zeros((batch, ceil(time * self.factor), feature))
        for b in range(batch):
            for f in range(feature):
                res[b, :, f] = signal.resample(data[b, :, f], ceil(time * self.factor))
        return res


# 序列的上下文长度选择
class Trimmer:
    def __init__(self, seq_l, pred_l):
        self.seq_l = seq_l
        self.pred_l = pred_l

    def pre_process(self, data):
        if data.shape[1] == self.seq_l:
            return data
        assert_timeseries_3d_np(data)
        assert data.shape[1] >= self.seq_l, f'Invalid data shape: {data.shape} for seq_l={self.seq_l}'

        res = data[:, -self.seq_l:, :]
        return res

    def post_process(self, data):
        if data.shape[1] == self.pred_l:
            return data
        assert_timeseries_3d_np(data)
        assert data.shape[1] >= self.pred_l
        res = data[:, :self.pred_l, :]
        return res


# 序列输入到模型前需要对序列进行对齐到Patch整数倍
class Aligner:
    def __init__(self, mode, method, data_patch_len):
        assert mode in ['none', 'data_patch', 'model_patch']
        assert method in ['none', 'trim', 'zero_pad', 'mean_pad', 'edge_pad']
        self.mode = mode
        self.method = method
        self.patch_len = data_patch_len

    def pre_process(self, data):  # padding mostly
        if self.mode == 'none' or self.method == 'none':
            return data
        assert_timeseries_3d_np(data)
        batch, time, feature = data.shape
        if time % self.patch_len == 0:
            return data
        pad_l = self.patch_len - time % self.patch_len if time % self.patch_len != 0 else 0
        res = np.zeros((batch, pad_l + time, feature))
        if self.method == 'trim':  # trim其实应该相信model自己完成。。。-》可能是0-pad
            if time < self.patch_len:  # 理论上不会出现
                self.method = 'edge_pad'
            else:
                valid_len = time // self.patch_len * self.patch_len
                return data[:, -valid_len:, :]

        for b in range(batch):
            for f in range(feature):
                # FIXME：应在头部而不是尾部填充数据！！！(到这时单array了！
                if self.method == 'zero_pad':
                    res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=0)
                elif self.method == 'mean_pad':  # FIXME:mean?axis?
                    res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=np.mean(data[b, :, f]))
                elif self.method == 'edge_pad':
                    res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'edge')
                else:
                    raise Exception('Invalid aligner: {}'.format(self.method))

        return res

    def post_process(self, data):
        return data


# 异常检测和填充
class Inputer:
    def __init__(self, detect_method, fill_method, history_seq):
        # history_seq: (batch, time, feature)
        self.detect_method = detect_method
        self.fill_method = fill_method
        self.statistics_dict = self.get_statistics_dict(history_seq)

    def get_statistics_dict(self, history_seq):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return None
        if 'sigma' in self.detect_method:
            mean = np.mean(history_seq, axis=1, keepdims=True)
            std = np.std(history_seq, axis=1, keepdims=True)
            statistics_dict = {'mean': mean, 'std': std}
        elif 'iqr' in self.detect_method:
            q1 = np.percentile(history_seq, 25, axis=1, keepdims=True)
            q3 = np.percentile(history_seq, 75, axis=1, keepdims=True)
            statistics_dict = {'q1': q1, 'q3': q3}
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")
        return statistics_dict

    def pre_process(self, data):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if 'sigma' in self.detect_method:
            k_sigma = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_k_sigma(data, k_sigma)
        elif 'iqr' in self.detect_method:
            ratio = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_iqr(data, ratio)
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")

        # # FIXME: 取出为不大量连续的fill_indices
        # tail_ratio = 1 / 4
        tail_ratio = 1
        batch_size, seq_len, feature_dim = data.shape
        rm_indices = set()

        # for idx in range(1, len(fill_indices[0])):
        #     # 若在同一个batch和同一个feature
        #     batch_idx_last, batch_idx_cur = fill_indices[0][idx - 1], fill_indices[0][idx]
        #     feature_idx_last, feature_idx_cur = fill_indices[2][idx - 1], fill_indices[2][idx]
        #     time_idx_last, time_idx_cur = fill_indices[1][idx - 1], fill_indices[1][idx]
        #     if batch_idx_cur == batch_idx_last and feature_idx_last == feature_idx_cur \
        #             and time_idx_cur - time_idx_last == 1:
        #         if time_idx_cur > seq_len * (1 - tail_ratio):
        #             rm_indices.add(idx - 1)

        consecutive_count = 0  # 用于记录连续异常点的数量
        threshold = 1  # FIXME:需要移除的连续异常点数量阈值 （认为n个趋势！大了10和3不太好
        # threshold = seq_len / 4
        for idx in range(1, len(fill_indices[0])):
            # 若在同一个batch和同一个feature
            batch_idx_last, batch_idx_cur = fill_indices[0][idx - 1], fill_indices[0][idx]
            feature_idx_last, feature_idx_cur = fill_indices[2][idx - 1], fill_indices[2][idx]
            time_idx_last, time_idx_cur = fill_indices[1][idx - 1], fill_indices[1][idx]
            if batch_idx_cur == batch_idx_last and feature_idx_last == feature_idx_cur \
                    and time_idx_cur - time_idx_last == 1 and time_idx_cur > seq_len * (1 - tail_ratio):
                consecutive_count += 1
                if consecutive_count >= threshold:
                    # rm_indices.extend(range(idx - threshold + 1, idx + 1))  # 移除连续的异常点
                    rm_indices.update(range(idx - threshold, idx))  # 移除连续的异常点
            else:
                consecutive_count = 1  # 重新计数
        # TODO
        new_fill_indices = [[], [], []]
        for idx in range(len(fill_indices[0])):
            if idx not in rm_indices:
                new_fill_indices[0].append(fill_indices[0][idx])
                new_fill_indices[1].append(fill_indices[1][idx])
                new_fill_indices[2].append(fill_indices[2][idx])
        new_fill_indices[0] = np.array(new_fill_indices[0])
        new_fill_indices[1] = np.array(new_fill_indices[1])
        new_fill_indices[2] = np.array(new_fill_indices[2])
        new_fill_indices = tuple(new_fill_indices)
        logging.debug(f"fill_indices: {fill_indices}")
        logging.debug(f"new_fill_indices: {new_fill_indices}")
        fill_indices = new_fill_indices

        filled_data = self.fill_outliers(data, fill_indices)
        if np.isnan(filled_data).any() or np.isinf(filled_data).any():
            logging.error(f"NaN or Inf values in filled data: {filled_data}")
            return data
        return filled_data

    def post_process(self, data):
        return data

    # def detect_outliers_k_sigma(self, data, k_sigma):
    #     mean = self.statistics_dict['mean']
    #     std = self.statistics_dict['std']
    #     lower_bound = mean - k_sigma * std
    #     upper_bound = mean + k_sigma * std
    #     return np.where((data < lower_bound) | (data > upper_bound))

    def detect_outliers_k_sigma(self, data, k_sigma):
        seq_len = data.shape[1]
        # cutoff_index = seq_len - seq_len // 4  # Exclude the last 1/10 of the sequence for separate handling
        cutoff_index = seq_len  # 相信2-sigma
        mean = self.statistics_dict['mean']
        std = self.statistics_dict['std']
        lower_bound = mean - k_sigma * std
        upper_bound = mean + k_sigma * std
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        fill_indices = np.where(mask)
        return fill_indices

    # def detect_outliers_iqr(self, data, ratio):
    #     q1 = self.statistics_dict['q1']
    #     q3 = self.statistics_dict['q3']
    #     iqr = q3 - q1
    #     lower_bound = q1 - ratio * iqr
    #     upper_bound = q3 + ratio * iqr
    #     return np.where((data < lower_bound) | (data > upper_bound))

    def detect_outliers_iqr(self, data, ratio):
        seq_len = data.shape[1]
        # cutoff_index = seq_len - seq_len // 4  # Exclude the last 1/10 of the sequence for separate handling
        cutoff_index = seq_len  # 相信2-sigma
        q1 = self.statistics_dict['q1']
        q3 = self.statistics_dict['q3']
        iqr = q3 - q1
        lower_bound = q1 - ratio * iqr
        upper_bound = q3 + ratio * iqr
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        fill_indices = np.where(mask)
        return fill_indices

    def fill_outliers(self, data, fill_indices):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if self.fill_method == 'linear_interpolate':
            filled_data = self.linear_interpolate(data, fill_indices)
        elif self.fill_method == 'rolling_mean':
            filled_data = self.rolling_mean(data, fill_indices)
        elif self.fill_method == 'forward_fill':
            filled_data = self.forward_fill(data, fill_indices)
        elif self.fill_method == 'backward_fill':
            filled_data = self.backward_fill(data, fill_indices)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")

        return filled_data

    def linear_interpolate(self, data, fill_indices):
        batch_size, seq_len, feature_dim = data.shape
        filled_data = data.copy()
        for b in range(batch_size):
            for f in range(feature_dim):
                # 使用布尔索引从 fill_indices[1] 中筛选出属于当前批次 b 的时间索引
                indices = fill_indices[1][fill_indices[0] == b]
                if len(indices) > 0:
                    normal_indices = np.setdiff1d(np.arange(seq_len), indices)
                    if len(normal_indices) == 0:
                        logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                        continue
                    filled_data[b, indices, f] = np.interp(indices, normal_indices, data[b, normal_indices, f])
        return filled_data

    def rolling_mean(self, data, fill_indices):
        window_size = 1000  # FIXME: magic number
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    start = max(0, idx - window_size)
                    end = min(seq_len, idx + window_size + 1)
                    neighbors = data[b, start:end, f]
                    valid_neighbors = neighbors[neighbors != 0]
                    if len(valid_neighbors) > 0:
                        filled_data[b, idx, f] = np.mean(valid_neighbors)
        return filled_data

    def forward_fill(self, data, fill_indices):
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    if idx > 0:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
        return filled_data

    def backward_fill(self, data, fill_indices):
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices[::-1]:
                    if idx < seq_len - 1:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
        return filled_data


# 值域的非线性变换
# 都不太行，过模型之后逆变换差距很大。。。感觉是异常值和MinMax算子的相互作用和mean问题，只平移试试 -> 现在Timer内置了scaler 可以了 -> 不行，老问题
class Warper:
    def __init__(self, method, clip_factor):
        self.method = method
        self.shift_values = None
        self.box_cox_lambda = None
        self.fail = False
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

    def pre_process(self, data):
        assert data.ndim == 3, f'Invalid data shape: {data.shape}'
        if self.method == 'none':
            return data

        batch_size, time_len, feature_dim = data.shape
        self.data_in = data

        if self.method == 'log':
            min_values = np.min(data, axis=1, keepdims=True)
            self.shift_values = np.where(min_values <= 1, 1 - min_values, 0)
            data_shifted = data + self.shift_values
            res = np.log(data_shifted)

        elif self.method == 'sqrt':
            min_values = np.min(data, axis=1, keepdims=True)
            self.shift_values = np.where(min_values < 0, 1 - min_values, 0)
            data_shifted = data + self.shift_values
            res = np.sqrt(data_shifted)
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        assert res.ndim == data.ndim, f'Invalid data shape: {res.shape}'
        assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
            f'Invalid data: {res}, method: {self.method}'

        if np.isnan(res).any() or np.isinf(res).any():
            logging.error(f"NaN or Inf values in transformed data: {res}")
            self.fail = True
            return data
        return res

    def post_process(self, data):
        if self.method == 'none':
            return data

        if self.fail:
            return data

        if self.method == 'log':
            _data = np.exp(data)
            data_restored = _data - self.shift_values

        elif self.method == 'sqrt':
            data_restored = np.square(data)
            data_restored = data_restored - self.shift_values

        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        res = data_restored
        if np.isnan(res).any() or np.isinf(res).any():
            logging.error(f"NaN or Inf values in restored data: {res}")
            res = my_clip(self.data_in, res, nan_inf_clip_factor=5)  # 表示出结果很差
        elif self.clip_factor is not None:
            res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        return res


# 差分
class Differentiator:  # 能预测趋势啦，但是误差累计很严重，短预测感觉可
    def __init__(self, n, clip_factor):
        self.n = n
        self.history_diff_data = []
        self.diff_data = None
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

    def pre_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape
        self.data_in = data

        self.history_diff_data = []
        # diff_data = data.copy()
        diff_data = data

        for _ in range(self.n):
            self.history_diff_data.append(diff_data[:, 0:1, :])  # 记录差分前的第一个值
            diff_data = np.diff(diff_data, axis=1)
        self.diff_data = diff_data

        aligner = Aligner('data_patch', 'zero_pad', time)  # FIXME
        res = aligner.pre_process(diff_data)
        return res

    def post_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape

        inv_diff_data_total = np.concatenate([self.diff_data, data], axis=1)
        for i in range(self.n - 1, -1, -1):
            inv_diff_data_total = np.concatenate([self.history_diff_data[i], inv_diff_data_total], axis=1)
            inv_diff_data_total = np.cumsum(inv_diff_data_total, axis=1)

        pre_time = self.diff_data.shape[1]
        assert pre_time + time + self.n == inv_diff_data_total.shape[1], \
            f"{pre_time} + {self.n} + {time} != {inv_diff_data_total.shape[1]}"
        inv_diff_data = inv_diff_data_total[:, pre_time:pre_time + time, :]

        # 填充大于最大值和小于最小值的数据，避免数值爆炸（log
        # inv_diff_data[inv_diff_data > self.max * 2] = self.max
        # inv_diff_data[inv_diff_data < self.min * 2] = self.min

        res = inv_diff_data
        # IQR-variant
        if np.isnan(res).any() or np.isinf(res).any():
            logging.error(f"NaN or Inf values in restored data: {res}")
            res = my_clip(self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)  # 表示出结果很差
        elif self.clip_factor is not None:
            res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        return res


# 降噪
class Denoiser:
    def __init__(self, method):
        assert method in ['none', 'moving_average', 'ewma', 'fft']  # median比较慢
        self.method = method
        self.window_size = 3
        self.alpha = 0.1

    def pre_process(self, data):
        if self.method == 'none':
            return data

        if self.method == 'moving_average':
            return self.moving_average(data)
        elif self.method == 'ewma':
            return self.ewma(data)
        # elif self.method == 'median':
        #     return self.median_filter(data)
        elif self.method == 'fft':
            return self.fft_filter(data)
        else:
            raise ValueError(f"Unsupported denoise method: {self.method}")

    def moving_average(self, data):
        window_size = self.window_size
        kernel = np.ones(window_size) / window_size
        smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
        return smoothed_data

    def ewma(self, data):
        alpha = self.alpha
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        smoothed_data[:, 0, :] = data[:, 0, :]  # Initialize with the first value

        for t in range(1, time):
            smoothed_data[:, t, :] = alpha * data[:, t, :] + (1 - alpha) * smoothed_data[:, t - 1, :]
        return smoothed_data

    def _apply_median_filter(self, data, window_size):
        pad_size = window_size // 2
        padded_data = np.pad(data, pad_size, mode='edge')
        smoothed_data = np.zeros_like(data)
        for i in range(len(data)):
            smoothed_data[i] = np.median(padded_data[i:i + window_size])
        return smoothed_data

    # def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
    #     b, a = butter(4, cutoff, btype='low', fs=fs)
    #     smoothed_data = np.apply_along_axis(lambda m: filtfilt(b, a, m), axis=1, arr=data)
    #     return smoothed_data

    def fft_filter(self, data):
        percentile = 80
        batch, time, feature = data.shape
        denoised_data = np.zeros_like(data)

        for b in range(batch):
            for f in range(feature):
                fft_coeffs = np.fft.fft(data[b, :, f])
                magnitudes = np.abs(fft_coeffs)
                upper_magnitude = np.percentile(magnitudes, percentile)
                fft_coeffs[magnitudes < upper_magnitude] = 0 + 0j
                denoised_data[b, :, f] = np.fft.ifft(fft_coeffs).real
        return denoised_data

    def post_process(self, data):
        return data
