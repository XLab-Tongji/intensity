import sys
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy import optimize
import copy
import os


class DecomposeIntensityService:


    def __init__(self):
        self.shapes = ["linear", "poly2"]


    def decompose(self, ts, period, method):
        if method == "x11":
            decomposition = seasonal_decompose(ts, period=period, model="add", extrapolate_trend="freq")
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            resid = decomposition.resid
            decomposition.plot()
            plt.show()
        else:
            raise Exception("Wrong decomposition method!")
        return trend, seasonal, resid


    def fit_interval(self, ts, start_x, end_x, shape, param_values=None):
        """

        :param ts:
        :param start_x:
        :param end_x:
        :param shape:
        :param param_values:
        :return:
        """
        assert 0 <= start_x < end_x <= len(ts)-1
        assert shape in self.shapes
        x_group = np.arange(start_x, end_x+1)
        y_group = ts[start_x: end_x+1]
        # 调用诸如self._linear等函数。注意这里不能用私有函数。
        __fit_function = getattr(self, "_" + shape)
        param_names, param_values, new_y_group, f_str = __fit_function(x_group, y_group, param_values)
        tmp_ts = np.concatenate((ts[:start_x], new_y_group, ts[end_x+1:]), axis=0)
        print("The function between [{}, {}] is:\n**{}** {}".format(start_x, end_x, shape, f_str))
        for name, value in zip(param_names, param_values):
            print("{} = {:.2}".format(name, value))
        return param_names, param_values, tmp_ts
    
    def update(self, ts, start_x, end_x, tmp_ts, auto_pad=False, period=-1):
        """
        :param ts: np.array, 待更新的时间序列
        :param start_x: int, 起始点x值
        :param end_x: int, 中止点x值
        :param tmp_ts: np.array, 用于更新ts的、由用户控制形状的时间序列，只在[start_x, end_x]与ts不同
        :param auto_pad: bool, 默认为False, 是否填充周期项
        :param period: int, 如果auto_pad=True, 则必须提供周期。可以考虑将其改进为自动检测。
        :return:
        """
        """
        ^^^*^^^^^*^^^^^*^^^^^*^^^^^*^   ts
        ^^^*^^^^^*^///^*^^^^^*^^^^^*^   tmp_ts
        """
        target_arr = tmp_ts[start_x: end_x+1]
        ts[start_x: end_x+1] = target_arr
        """
        ^^^*^^^^^*^///^*^^^^^*^^^^^*^   ts
        """
        if auto_pad:
            i = 1
            l, r = start_x - i * period, end_x - i * period
            while l >= 0:
                ts[l: r+1] = target_arr
                i += 1
                l, r = start_x - i * period, end_x - i * period
            """
            ^^^*^///^*^///^*^^^^^*^^^^^*^^  ts
            """
            if r >= 0:
                ts[:r+1] = target_arr[-(r+1):]
            """
            //^*^///^*^///^*^^^^^*^^^^^*^^  ts
            """
            i = 1
            l, r = start_x + i * period, end_x + i * period
            while r < len(ts):
                ts[l: r+1] = target_arr
                i += 1
                l, r = start_x + i*period, end_x + i*period
            """
            //^*^///^*^///^*^///^*^///^*^^  ts
            """
            if r >= len(ts):
                ts[l:] = target_arr[:len(ts)-l]
            """
            //^*^///^*^///^*^///^*^///^*^/  ts
            """


    def _linear(self, x_group, y_group, param_values):
        def f_linear(x, a, b):
            return a * x + b
        if param_values is None:
            param_values, pcov = optimize.curve_fit(f_linear, x_group, y_group)
        assert len(param_values) == 2
        new_y_group = f_linear(x_group, param_values[0], param_values[1])
        f_str = "y = {:.2} * x + {:.2}".format(param_values[0], param_values[1])
        param_names = ['a', 'b']
        return param_names, param_values, new_y_group, f_str

    def _poly2(self, x_group, y_group, param_values):
        def f_ploy2(x, a1, a2, b):
            return a1 * x * x + a2 * x + b
        if param_values is None:
            param_values, pcov = optimize.curve_fit(f_ploy2, x_group, y_group)
        assert len(param_values) == 3
        new_y_group = f_ploy2(x_group, param_values[0], param_values[1], param_values[2])
        f_str = "y = {:.2} * x^2 + {:.2} * x + {:.2}".format(param_values[0], param_values[1], param_values[2])
        param_names = ['a1', 'a2', 'b']
        return param_names, param_values, new_y_group, f_str

    def auto_choose(self, ts, start_x, end_x):
        """
        自动选择拟合函数
        :param ts:
        :param start_x:
        :param end_x:
        :return:
        """
        def abs_error(ts1, ts2):
            return sum([abs(v1-v2) for v1, v2 in zip(ts1, ts2)])
        best_shape = best_param_names = best_param_values = best_tmp_ts = None
        min_error = float("Inf")
        for shape in self.shapes:
            param_names, param_values, tmp_ts = self.fit_interval(ts, start_x, end_x, shape)
            error = abs_error(ts[start_x: end_x+1], tmp_ts[start_x: end_x+1])
            if error < min_error:
                best_shape = shape
                best_param_names = param_names
                best_param_values = param_values
                best_tmp_ts = tmp_ts
        print("The best shape is **{}**.".format(best_shape))
        return best_shape, best_param_names, best_param_values, best_tmp_ts


    def fit_trend(self, x, n_predict=100):
        n = x.size
        n_harm = 18  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        self.plot_ts(x, x)

    def fit_seasonal(self, x, n_predict=100):
        n = x.size
        n_harm = 20  # number of harmonics in model
        x_freqdom = fft.fft(x)  # detrended x in frequency domain
        f = fft.fftfreq(n)  # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        self.plot_ts(x, restored_sig)


    def plot_ts(self, orig_ts, extrapolation_ts):
        plt.clf()
        plt.plot(np.arange(0, len(extrapolation_ts)), extrapolation_ts, 'r', label='extrapolation')
        plt.plot(np.arange(0, len(orig_ts)), orig_ts, 'b', label='origin', linewidth=1)
        plt.legend()
        plt.show()

    def mock_user_interaction(self, ts, trend, seasonal, resid, period=-1):
        # 选择待拟合的时间序列
        ts_types = [ts, trend, seasonal, resid]
        ts_type_names = ["ts", "trend", "seasonal", "resid"]
        print("Please choose the time series type (e.g., 2):")
        for i, name in enumerate(ts_type_names, start=1):
            print("({}) {}".format(i, name))
        id = int(sys.stdin.readline())
        ts = ts_types[id-1]
        print("You select: ({}) {}".format(id, ts_type_names[id-1]))
        # 选择起止点
        print("Please choose the interval of [start_x, end_x] to fit (e.g., 10 21):")
        start_x, end_x = [int(v) for v in sys.stdin.readline().split()]
        print("You select: start_x={}, end_x={}".format(start_x, end_x))
        # 选择拟合函数
        print("Please choose the extrapolation method (e.g., 1):")
        shape_names = self.shapes + ["auto choose"]
        for i, name in enumerate(shape_names, start=1):
            print("({}) {}".format(i, name))
        id = int(sys.stdin.readline())
        shape = shape_names[id-1]
        print("You select: ({}) {}".format(id, shape))
        # 自动选择拟合函数
        if shape == "auto choose":
            shape, param_names, param_values, tmp_ts = self.auto_choose(ts, start_x, end_x)
        # 使用指定的拟合函数
        else:
            param_names, param_values, tmp_ts = self.fit_interval(ts, start_x, end_x, shape)
        print("Close the figure to continue..")
        self.plot_ts(ts, tmp_ts)
        # 调整拟合函数的参数
        flag = True
        while flag:
            print("Please enter the modified parameters (or 'k' to keep, 'q' to quit):")
            modified_param_values = copy.deepcopy(param_values)
            for i, name in enumerate(param_names):
                print("{} = ".format(name), end="")
                s = sys.stdin.readline()
                if str(s.strip()) == 'k':
                    modified_param_values[i] = param_values[i]
                elif str(s.strip()) == 'q':
                    flag = False
                    break
                else:
                    modified_param_values[i] = float(s)
            if not flag:
                break

            param_names, param_values, tmp_ts = self.fit_interval(ts, start_x, end_x, shape, modified_param_values)
            print("Close the figure to continue..")
            self.plot_ts(ts, tmp_ts)
        # 是否自动填充
        print("Whether to auto pad the seasonal pattern? (y/n):")
        s = sys.stdin.readline()
        if str(s.strip()) == 'y':
            new_ts = copy.deepcopy(ts)
            self.update(new_ts, start_x, end_x, tmp_ts, auto_pad=True, period=period)
            print("Close the figure to continue..")
            self.plot_ts(ts, new_ts)

        print("Bye bye")
        os._exit(0)

    





