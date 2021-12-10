from datetime import datetime
import pandas as pd
import copy
from service.intensity.DecomposeIntensityService import DecomposeIntensityService


# 将时间序列分解为季节项、趋势项、噪声，再通过函数拟合解析出函数参数
if __name__ == "__main__":
    def read_from_airpassenger_csv():
        date_parse = lambda x: datetime.strptime(x, '%Y-%m')
        data = pd.read_csv('input/AirPassengers.csv',
                           index_col='Month',  # 指定索引列
                           parse_dates=['Month'],  # 将指定列按照日期格式来解析
                           date_parser=date_parse  # 日期格式解析器
                           )

        ts = data.values.ravel()
        period = 12
        return ts, period

    def run_plan_A():
        # A. 模拟的用户交互
        ts, period = read_from_airpassenger_csv()
        decompose_service = DecomposeIntensityService()
        trend, seasonal, resid = decompose_service.decompose(ts, period, "x11")
        decompose_service.mock_user_interaction(ts, trend, seasonal, resid, period)

    def run_plan_B():
        # B. 算法的基本运行思路
        # 1. 读取数据
        ts, period = read_from_airpassenger_csv()
        # 2. 分解
        decompose_service = DecomposeIntensityService()
        trend, seasonal, resid = decompose_service.decompose(ts, period, "x11")
        new_ts, new_trend, new_seasonal, new_resid = copy.deepcopy(ts), copy.deepcopy(trend), copy.deepcopy(seasonal), copy.deepcopy(resid)
        # 3. 控制时间序列的形状【反复循环】
        #   3-1 选择目标时间序列（e.g., trend）、起止点x值、拟合函数形状，获得拟合函数的参数
        param_names, param_values, tmp_ts = decompose_service.fit_interval(ts=trend, start_x=0, end_x=50, shape="linear")
        decompose_service.plot_ts(trend, tmp_ts)
        #   3-2、3-3【反复循环】
        #       3-2 用户修改参数
        for i, name in enumerate(param_names):
            param_values[i] = param_values[i] + 0.1
        #       3-3 用新的参数查看拟合效果
        param_names, param_values, tmp_ts = decompose_service.fit_interval(ts=trend, start_x=0, end_x=50, shape="linear", param_values=param_values)
        decompose_service.plot_ts(trend, tmp_ts)
        #   3-4 找到合适的参数后，更新时间序列
        decompose_service.update(ts=new_trend, start_x=0, end_x=50, tmp_ts=tmp_ts)
        # 4. 如果修改ts，则此时直接得到修改后的时间序列new_ts;
        #    如果修改trend/seasonal/resid，则合并new_trend, new_seasonal, new_resid，得到new_ts
        new_ts = new_trend + new_seasonal + new_resid
        decompose_service.plot_ts(ts, new_ts)


    # A、B二选一运行
    run_plan_A()
