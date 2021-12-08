import math
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from service.intensity.BaseIntensityService import BaseIntensityService


class LIMBOIntensityService(BaseIntensityService):
    """
    LIMBO 方法建模时间序列，生成指定的负载强度
    """

    def __init__(self, ts_length):
        """
        :param ts_length: 时间序列总长度
        """
        self.ts_length = ts_length
        self.ts_seasonal_y = []
        # self.ts_trend_y采用"值替换"的方式构建，所以其初始化方式与其他项不同
        self.ts_trend_y = [0] * ts_length
        self.ts_burst_y = []
        self.ts_noise_y = []

    def seasonal_part(self, period, n_peaks, start_or_end_y, trough_y,
                      first_peak_x, first_peak_y, last_peak_x, last_peak_y, shape="quadratic"):
        """
        生成seasonal。
        :param period: 每个周期的长度
        :param n_peaks: 每个周期包含的波峰数量
        :param start_or_end_y: 每个周期开始/结束处的y值（这两个值应该相同）
        :param trough_y: 两个波峰之间最低点（波谷）的y值
        :param first_peak_x: 每个周期第一个波峰的x值
        :param first_peak_y: 每个周期第一个波峰的y值
        :param last_peak_x: 每个周期最后一个波峰的x值
        :param last_peak_y: 每个周期最后一个波峰的y值
        :param shape: 插值函数，具体取值参考函数self.__interpolate()
        :return:
        """
        # 确保peak数量不为负
        assert n_peaks >= 0
        # 确保时间序列总长度ts_length不小于period
        assert self.ts_length >= period
        # 没有peak，相当于一条水平线
        if n_peaks == 0:
            print("n_peaks is 0!")
            assert period >= 1
            seasonal_y = [start_or_end_y] * period
        # 一个或多个peak
        else:
            # 确保可以画出n个peak
            assert (last_peak_x - first_peak_x + 1) >= (2 * n_peaks - 1)
            # 确保first_peak_x、last_peak_x顺序正确
            assert first_peak_x <= last_peak_x <= period - 1
            # 确保peak高于start_or_end_y
            assert min(first_peak_y, last_peak_y) > start_or_end_y
            if n_peaks > 1:
                # 确保peak高于trough_y
                assert min(first_peak_y, last_peak_y) > trough_y

            # [每个周期开始处的x值, 第一个peak的x值，最后一个peak的x值，每个周期结束处的x值]
            x = [0, first_peak_x]
            # [每个周期开始处的y值, 第一个peak的y值，最后一个peak的y值，每个周期结束处的y值]
            y = [start_or_end_y, first_peak_y]

            if n_peaks > 1:
                # int，表示两个peak之间的x轴间隔
                x_step = (last_peak_x - first_peak_x) // (n_peaks - 1)
                # float，表示两个peak之间的y轴间隔
                y_step = (last_peak_y - first_peak_y) / (n_peaks - 1)
                # 第一个trough的x、y值
                interval_trough_x = first_peak_x + (x_step) // 2
                x.append(interval_trough_x)
                y.append(trough_y)
                for i in range(n_peaks - 2):
                    # 中间的peak的x、y值
                    interval_peak_x = first_peak_x + x_step * (i + 1)
                    interval_peak_y = first_peak_y + y_step * (i + 1)
                    x.append(interval_peak_x)
                    y.append(interval_peak_y)
                    # 最后一个peak后是(period-1, start_or_end_y)，没有新的trough
                    if i != n_peaks - 1:
                        # 中间的trough的x、y值
                        interval_trough_x = interval_peak_x + (x_step) // 2
                        x.append(interval_trough_x)
                        y.append(trough_y)
            x.extend([last_peak_x, period - 1])
            y.extend([last_peak_y, start_or_end_y])
            # n_peaks >= 1时需要插值
            seasonal_y = self.__interpolate(x, y, shape, list(range(period)))

        # 重复单个周期，得到最终的seasonal
        while len(self.ts_seasonal_y) < self.ts_length:
            self.ts_seasonal_y += seasonal_y
        self.ts_seasonal_y = self.ts_seasonal_y[:self.ts_length]

        BaseIntensityService.plot_ts(self.ts_seasonal_y, "limbo_seasonal")

    def trend_part(self, start_x, start_y, end_x, end_y, mid_x=None, mid_y=None, shape="linear"):
        """
        生成trend的**一部分**，x轴的起止点为[start_x, end_x]。
        该函数可设置不同的参数并执行多次，以生成不同位置的trend。
        :param start_x：trend开始处的x坐标
        :param start_y: trend开始处的y坐标
        :param end_x: trend结束处的x坐标
        :param end_y: trend结束处的y坐标
        :param mid_x, trend中间处的x坐标，如果shape不为"linear""slinear"，则需要提供该参数，否则无法进行插值
        :param mid_y: trend中间处的y坐标，如果shape不为"linear""slinear"，则需要提供该参数，否则无法进行插值
        :param shape: 插值函数，具体取值参考函数self.__interpolate()，推荐使用"linear"
        :return:
        """
        if shape in ["linear", "slinear"]:
            assert mid_x is None and mid_y is None
            assert 0 <= start_x < end_x < self.ts_length
            x = [start_x, end_x]
            y = [start_y, end_y]
        elif shape in ["quadratic", "cubic"]:
            assert mid_x is not None and mid_y is not None
            assert 0 <= start_x < mid_x < end_x < self.ts_length
            x = [start_x, mid_x, end_x]
            y = [start_y, mid_y, end_y]
        else:
            raise Exception("Wrong shape!")
        trend_y = self.__interpolate(x, y, shape, [i for i in range(start_x, end_x + 1)])
        # 将trend_y替换至self.ts_trend_y中的对应位置
        self.ts_trend_y = self.ts_trend_y[:start_x] + trend_y + self.ts_trend_y[end_x + 1:]
        BaseIntensityService.plot_ts(self.ts_trend_y, "limbo_trend")

    def burst_part(self, first_burst_x, first_burst_y, gap, width, shape):
        """
        生成burst。
        :param first_burst_x: 第一个burst最高点对应的x值
        :param first_burst_y: 第一个burst最高点对应的y值，也是其他burst最高点对应的y值
        :param gap: 两个burst最高点之间的x轴间隔
        :param width: burst的宽度
        :param shape: 插值函数，具体取值参考函数self.__interpolate()
        :return:
        """
        # 确保各参数合法
        assert first_burst_x >= 0 and gap > 0 and width > 0 and width % 2 == 0 and gap >= width
        half_width = width // 2
        x = [first_burst_x - half_width, first_burst_x]
        y = [0, first_burst_y]
        # burst的左半部分
        left = self.__interpolate(x, y, shape, [i for i in range(first_burst_x - half_width, first_burst_x + 1)])
        # burst的右半部分
        right = []
        for i in range(half_width):
            right.append(left[-i - 2])
        # 创建第一个burst。如果first_burst_x-half_width < 0，该burst会被x=0截断
        if first_burst_x - half_width >= 0:
            self.ts_burst_y = [0] * (first_burst_x - half_width) + left + right
        else:
            self.ts_burst_y = left[half_width - first_burst_x:] + right
        # 创建所有的burst
        while len(self.ts_burst_y) <= self.ts_length:
            # gap、width二者相等时，共用同一个y=0的点，所以left数组舍弃第一项
            if gap == width:
                self.ts_burst_y += left[1:] + right
            else:
                self.ts_burst_y += [0] * (gap - width - 1) + left + right
        self.ts_burst_y = self.ts_burst_y[:self.ts_length]
        BaseIntensityService.plot_ts(self.ts_burst_y, "limbo_burst")

    def noise_part(self, min_noise_y, max_noise_y, distribution):
        """
        生成noise。
        :param min_noise_y: 噪声项的下界
        :param max_noise_y: 噪声项的上界
        :param distribution: 可选项包括"uniform"与"normal"。
                            如果是"normal"，则99.7%的数据在[min_noise_y, max_noise_y]的范围内，即离平均值3个标准差。
        :return:
        """
        assert min_noise_y < max_noise_y and distribution in ["uniform", "normal"]
        for i in range(self.ts_length):
            if distribution == "uniform":
                v = np.random.uniform(min_noise_y, max_noise_y)
            elif distribution == "normal":
                mu = (min_noise_y + max_noise_y) / 2
                # 3个标准差
                sigma = (max_noise_y - min_noise_y) / 6
                v = np.random.normal(loc=mu, scale=sigma)
            else:
                raise Exception("Wrong distribution!")
            self.ts_noise_y.append(v)
        BaseIntensityService.plot_ts(self.ts_noise_y, "limbo_noise")

    def __interpolate(self, x, y, shape, x_new):
        """
        如果shape为"linear"且x、y长度大于2，则生成折线图；
        如果shape为'quadratic'、'cubic'，则x、y长度至少为3；
        如果shape为"log""exp""sin"，则x、y长度必须为2。（不推荐使用）
        :param x: 样本点的x值组成的数组, e.g., x=[1, 3, 5]
        :param y: 样本点的y值组成的数组, e.g., y=[1, 3, 5]
        :param shape: ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
                    ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order;
                    ‘previous’ and ‘next’ simply return the previous or next value of the point;
                    ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5)
                    in that ‘nearest-up’ rounds up and ‘nearest’ rounds down.
                    Default is ‘linear’.
                    'pchip'，对应PchipInterpolator
                    burst项可以使用"log""exp""sin"，seasonal项、trend项不可以
        :param x_new: 待拟合的x值组成的数组, e.g., x_new=[1, 2，3, 4, 5, 6]
        :return:
        """

        def log_interp1d(x, y, delta_x=0, delta_y=0):
            """
            样本点数量必须为2（超过2，则不一定能通过log拟合）。
            设插值函数为：y = ln(Kx+B)。
            使用log插值时，假设输入坐标为(a1, b1)、(a2, b2)。
            则函数y = ln(Kx+B)必须经过(x=a1, y=b1)、(x=a2, y=b2)
            左右取exp，有：e^y = Kx + B，同理，该函数必须经过(x=a1, y=b1)、(x=a2, y=b2)
            记h = e^y = Kx + B，则该函数必须经过(x=a1, h=b1)、(x=a2, h=b2)，对这两点做线性插值，可以解出K、B
            :param x: 长度必须为2
            :param y: 长度必须为2
            :param delta_x: 超参数，影响log的形状
            :param delta_y: 超参数，影响log的形状
            :return:
            """
            for i in range(len(x)):
                x[i] += delta_x
            for i in range(len(y)):
                y[i] += delta_y
            expy = np.exp(y)
            lin_interp = interpolate.interp1d(x, expy, kind='linear', fill_value="extrapolate")
            log_interp = lambda x_new: np.log(
                lin_interp([x_new[j] + delta_x for j in range(len(x_new))])
            ) - delta_y
            return log_interp

        def exp_interp1d(x, y, delta_x=0, delta_y=0):
            """
            相当于 log插值的逆过程。
            :param x: 长度必须为2
            :param y: 长度必须为2
            :param delta_x: 超参数，影响log的形状，必须大于等于0
            :param delta_y: 超参数，影响log的形状，必须大于等于0
            :return:
            """
            delta_x = delta_x + 1 - min(x)  # 这个其实不是很重要
            delta_y = delta_y + 1 - min(y)  # 确保y大于0，从而后续计算np.log(y)
            for i in range(len(x)):
                x[i] += delta_x
            for i in range(len(y)):
                y[i] += delta_y
            logy = np.log(y)
            lin_interp = interpolate.interp1d(x, logy, kind='linear', fill_value="extrapolate")
            exp_interp = lambda x_new: np.exp(
                lin_interp([x_new[j] + delta_x for j in range(len(x_new))])
            ) - delta_y
            return exp_interp

        def sin_interp1d(x, y):
            """
            样本点数量必须为2（超过2，则不一定能通过sin拟合）。
            相当于求解y = Asin(Kx+B)+D。该函数参数众多，仅靠两个样本点无法求解。因此，必须对这两个样本点的位置进行假设。
            使用sin插值时，假设输入坐标为(a1, b1)、(a2, b2)。
            假设(a1, b1)为左下，相当于y = Asin(Kx+B)+D的最低点；假设(a2, b2)为右上，相当于y = Asin(Kx+B)+D的最高点。s
            此时，根据sin函数的特点，可以求解函数y = Asin(Kx+B)+D。
            :param x: 长度必须为2
            :param y: 长度必须为2
            :return:
            """
            print(x)
            print(y)
            basex = -min(x)
            basey = -min(y)
            x_scale = (max(x) - min(x)) / math.pi
            y_scale = (max(y) - min(y)) / 2

            sin_interp = lambda x_new: (np.sin((np.array(x_new) + basex) / x_scale - math.pi / 2) + 1) * y_scale + basey
            return sin_interp

        if shape in ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
                     'cubic', 'previous', 'next', 'log', 'exp', 'sin', 'pchip']:
            if shape == 'log':
                f_interpolate = log_interp1d(x, y)
            elif shape == 'exp':
                f_interpolate = exp_interp1d(x, y)
            elif shape == 'sin':
                f_interpolate = sin_interp1d(x, y)
            elif shape == 'pchip':
                f_interpolate = interpolate.PchipInterpolator(x, y)
            else:
                f_interpolate = interpolate.interp1d(x, y, kind=shape, fill_value="extrapolate")
            y = f_interpolate(x_new)
            return list(y)
        else:
            raise Exception("Wrong shape")



    def merge(self, add_or_mul="add"):
        """
        合并各组成项
        :param add_or_mul: 各项的组合模式，取值范围为["add", "mul"]
        :return:
        """
        assert len(self.ts_seasonal_y) > 0 and len(self.ts_burst_y) > 0 and len(self.ts_noise_y) > 0
        if add_or_mul == "add":
            a = np.add(self.ts_seasonal_y, self.ts_trend_y)
            b = np.add(self.ts_burst_y, self.ts_noise_y)
            ts = np.add(a, b)
        elif add_or_mul == "mul":
            a = np.multiply(self.ts_seasonal_y, self.ts_trend_y)
            b = np.multiply(self.ts_burst_y, self.ts_noise_y)
            ts = np.multiply(a, b)
        else:
            raise Exception("Wrong add_or_mul!")
        BaseIntensityService.plot_ts(ts, "limbo_ts")
        return list(ts)

