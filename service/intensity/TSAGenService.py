import math
import matplotlib.pyplot as plt
import numpy as np
from service.intensity.BaseIntensityService import BaseIntensityService
from utils.RMDF import RMDF
from scipy.stats import pearson3


def param_sample(param):
    if not isinstance(param, list):
        param = [param for n in range(5)]
    pos = np.random.choice([1, 2, 3, 4], 1, p=[1/4, 1/4, 1/4, 1/4])[0]
    return np.random.uniform(param[pos - 1], param[pos])


def moving_average(data, n):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class TSAGenService(BaseIntensityService):
    """
    TSAGen方法建模时间序列，生成指定的负载强度
    """

    def __init__(self):
        self.ts_seasonal_y = []
        self.ts_trend_y = []
        self.ts_noise_y = []

        self.params = {
            'theta': {
                # trend params
                'theta1': [],  # trend level
                'theta2': [],  # trend slope
                # season params
                'theta3': [],  # amplitude
                'theta4': [],  # cycle length, i.e., 1/frequency
                'theta5': [],  # num of cycles
                # noise params
                'theta6': [],  # location
                'theta7': [],  # scale
                'theta8': [],  # skew
            },
            'k': {
                'k1': [],  # amplitude drift
                'k2': []   # frequency drift
            },
            'd': {
                'd': 10,  # recursion depth
                'd_hat': 8  # forking depth
            },
            'size': 1000  # length of ts
        }

    def init_from_meta_params(self, params, num):
        """
        从元数据生成参数
        """
        for key, value in params['theta'].items():
            for i in range(num):
                self.params['theta'][key].append(param_sample(value))

        for key, value in params['k'].items():
            for i in range(num):
                self.params['k'][key].append(param_sample(value))
        self.params['d'] = params['d']
        self.params['size'] = params['size']

    def trend_part(self, length):
        """
        生成trend。

        :return: 生成的trend序列.
        """
        result = []
        for i in range(len(self.params['theta']['theta1'])):
            for j in range(length):
                result.append(self.params['theta']['theta1'][i] + math.tan(self.params['theta']['theta2'][i]) * j)
        return result

    def seasonal_part(self):
        """
        生成season。

        :return: 生成的season序列.
        """
        cycle_list = []

        cycle_generator = RMDF(depth=10)
        cycle_generator.gen_anchor()

        for i in range(len(self.params['theta']['theta3'])):
            drift_a_of_cycles = np.random.uniform(1, 1 + self.params['k']['k1'][i],
                                                  int(self.params['theta']['theta5'][i]))
            drift_f_of_cycles = np.random.uniform(1, 1 + self.params['k']['k2'][i],
                                                  int(self.params['theta']['theta5'][i]))
            for drift_a, drift_f in zip(drift_a_of_cycles, drift_f_of_cycles):
                length = int(drift_f * self.params['theta']['theta4'][i])
                season = self.params['theta']['theta3'][i] * drift_a * cycle_generator.gen(self.params['d']['d_hat'],
                                                                                           length)

                cycle_list.append(season)

        result = np.concatenate(cycle_list)
        return result

    def noise_part(self, length, distribution):
        """
        生成noise。

        :param length: 生成的noise项序列长度.
        :param distribution: 可选项包括"pearson"与"normal".
        :return: 生成的noise序列.
        """
        result = []

        assert distribution in ["pearson", "normal"]
        if distribution == "normal":
            for i in range(len(self.params['theta']['theta6'])):
                for j in range(length):
                    result.append(np.random.normal(loc=self.params['theta']['theta6'][i],
                                                   scale=self.params['theta']['theta7'][i]))
        elif distribution == "pearson":
            for i in range(len(self.params['theta']['theta6'])):
                for j in range(length):
                    result.append(pearson3.rvs(skew=self.params['theta']['theta8'][i],
                                               loc=self.params['theta']['theta6'][i],
                                               scale=self.params['theta']['theta7'][i]))
        else:
            raise Exception("Wrong distribution!")
        return result

    def pearson_gen(self, param_num):
        seasonal_part = []

        # 防止生成的点过少，不足以生成足够数量的点
        valid = False
        for i in range(5):
            seasonal_part = self.seasonal_part()
            if seasonal_part.size < self.params['size']:
                continue
            else:
                valid = True
                break

        if not valid:
            raise Exception("Too few points! Check theta4, theta5 and param_num.")

        trend_part = self.trend_part(int(len(seasonal_part) / param_num))
        noise_part = self.noise_part(int(len(seasonal_part) / param_num), 'pearson')

        n = int(len(trend_part) / self.params['size'])

        self.ts_trend_y = moving_average(np.array(trend_part), n)[:self.params['size']]
        self.ts_noise_y = moving_average(np.array(noise_part), n)[:self.params['size']]
        self.ts_seasonal_y = moving_average(np.array(seasonal_part), n)[:self.params['size']]

        BaseIntensityService.plot_ts(self.ts_trend_y, "tsagen_trend")
        BaseIntensityService.plot_ts(self.ts_seasonal_y, "tsagen_seasonal")
        BaseIntensityService.plot_ts(self.ts_noise_y, "tsagen_noise")


    def merge(self, add_or_mul="add"):
        """
        合并各组成项
        :param add_or_mul: 各项的组合模式，取值范围为["add", "mul"]
        :return: 合并后的时间序列数据
        """
        assert len(self.ts_seasonal_y) > 0 and len(self.ts_trend_y) > 0 and len(self.ts_noise_y) > 0
        if add_or_mul == "add":
            ts = np.add(self.ts_seasonal_y, self.ts_trend_y)
            ts = np.add(ts, self.ts_noise_y)
        elif add_or_mul == "mul":
            ts = np.multiply(self.ts_seasonal_y, self.ts_trend_y)
            ts = np.multiply(ts, self.ts_noise_y)
        else:
            raise Exception("Wrong add_or_mul!")

        BaseIntensityService.plot_ts(ts, "tsagen_ts")
        return list(ts)

