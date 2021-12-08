from service.intensity.TSAGenService import TSAGenService
from config.FilenameConfig import filename_config


# 通过tsagen方法生成负载
if __name__ == "__main__":
    tsa_gen = TSAGenService()
    param_num = 10
    params = {
        'theta': {
            # trend params
            'theta1': [-3, -1, 0, 1, 3],  # trend level
            'theta2': 0.01,  # trend slope
            # season params
            'theta3': [1, 2, 3, 5, 8],  # amplitude
            'theta4': 100,  # cycle length, i.e., 1/frequency
            'theta5': 5,  # num of cycles
            # noise params
            'theta6': 0,  # location
            'theta7': [0.1, 0.2, 0.3, 0.4, 1],  # scale
            'theta8': 0,  # skew
        },
        'k': {
            'k1': 0.1,  # amplitude drift
            'k2': 0.1  # frequency drift
        },
        'd': {
            'd': 10,  # recursion depth
            'd_hat': 8  # forking depth
        },
        'size': 500  # length of intensity series
    }
    tsa_gen.init_from_meta_params(params, param_num)
    tsa_gen.pearson_gen(param_num)
    ts = tsa_gen.merge(add_or_mul="add")
    tsa_gen.build_workload_intensity(ts=ts, start_t="2021-07-09 00:00:00", t_interval=3600,
                                     png=filename_config.tsagen_intensity_png, pkl=filename_config.tsagen_intensity_pkl)
