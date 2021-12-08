import pickle
import os
import matplotlib.pyplot as plt
import arrow

plt.figure(figsize=(4, 3))

class BaseIntensityService:

    @staticmethod
    def dump_intensity(intensity, filename):
        with open('output/' + filename, 'wb') as f:
            pickle.dump(intensity, f)
        print(filename + " has been dumped.")

    @staticmethod
    def load_intensity(filename):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        file_path = os.path.join(project_dir, 'service', 'intensity', 'output', filename)
        with open(file_path, 'rb') as f:
            intensity = pickle.load(f)
            print(filename + " has been loaded.")
        return intensity

    # todo: t_interval可以从intensity的key中计算，无需通过参数传递
    @staticmethod
    def plot_intensity(intensity, t_interval, filename):
        """
        绘制散点图（如果intensity波动较大，折线图不美观）
        :param intensity: intensity是一个字典，参见本文件中build_workload_intensity()的参数说明
        :param t_interval: 以秒为单位的时间间隔，e.g., 3600，代表3600秒
        :param filename: 文件名
        :return:
        """
        plt.scatter([i for i in range(len(intensity.keys()))], intensity.values(),
                    label="interval=" + str(t_interval) + "s")
        plt.legend()
        plt.savefig("output/" + filename)

    @staticmethod
    def plot_ts(data, label):
        """
        绘制limbo或tsagen生成的时间序列或时间序列组成部分
        :param data: 一维数组
        :param label: 标签&文件名（无后缀）
        :return:
        """
        plt.plot(data, label=label)
        plt.legend()
        plt.savefig("output/" + label + ".png")
        plt.clf()


    #todo: t_interval可以从intensity的key中计算，无需通过参数传递
    @staticmethod
    def get_intensity_by_timestamp(intensity, timestamp, t_interval):
        """
        根据时间戳，返回该时间段t_interval内的intensity
        :param intensity: intensity是一个字典，参见本文件中build_workload_intensity()的参数说明
        :param timestamp: e.g., arrow.get('2021-07-09T11:16:49.338460317Z')
        :param t_interval: 以秒为单位的时间间隔，e.g., 3600，代表3600秒
        :return:
        """
        start_timestamp = list(intensity.keys())[0][0]
        index = (timestamp - start_timestamp).seconds // t_interval
        left = start_timestamp.shift(seconds=index * t_interval).floor("second")
        right = start_timestamp.shift(seconds=(index + 1) * t_interval - 0.1).ceil("second")
        return intensity[(left, right)]

    @staticmethod
    def build_workload_intensity(ts, start_t, t_interval, png, pkl):
        """
        将数组形式的时间序列ts转为包含arrow的intensity形式。
        其中，intensity是一个字典：
        key为元组，e.g.，(<class 'tuple'>: (<Arrow [2021-07-09T00:00:00+00:00]>, <Arrow [2021-07-09T00:59:59.999999+00:00]>))，
        value为session数量，e.g.，310
        :param ts: limbo/tsagen生成的时间序列，形式为一维数组，e.g., ts=[310, 100, 220, 480]
        :param start_t: 开始时间戳，能够被arrow读取的字符串形式，e.g., start_t="2021-07-09 00:00:00"
        :param t_interval: 时间间隔，以秒（s）为单位，e.g., ts[0]对应的时间区间为[start_t, start_t+t_interval]
        :param png: intensity_{method}.png的实际文件名，参见config.FilenameConfig.py
        :param pkl: intensity_{method}.pkl的实际文件名，参见config.FilenameConfig.py
        :return:
        """
        cur_t = arrow.get(start_t)
        last_t = cur_t.shift(seconds=t_interval * len(ts))
        print("Generating time series from {} to {} with {} points..".format(cur_t, last_t, len(ts)))
        intensity = {}
        for i in range(len(ts)):
            next_t = cur_t.shift(seconds=t_interval)
            timespan = (cur_t, next_t)
            intensity[timespan] = int(ts[i])  # 负载强度为int型数据
            cur_t = next_t
        BaseIntensityService.plot_intensity(intensity=intensity, t_interval=t_interval, filename=png)
        BaseIntensityService.dump_intensity(intensity=intensity, filename=pkl)
