import os, sys
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from service.intensity.LIMBOIntensityService import LIMBOIntensityService
from config.FilenameConfig import filename_config



# 通过limbo方法生成负载
if __name__ == "__main__":
    limbo = LIMBOIntensityService(ts_length=450)
    limbo.seasonal_part(period=100, n_peaks=4, start_or_end_y=0, trough_y=20,
                        first_peak_x=20, first_peak_y=100, last_peak_x=80, last_peak_y=200,
                        shape="pchip")
    limbo.trend_part(start_x=0, start_y=0, end_x=150, end_y=150,
                     shape="linear")
    limbo.trend_part(start_x=150, start_y=150, end_x=449, end_y=300,
                     mid_x=300, mid_y=200, shape="quadratic")
    limbo.burst_part(first_burst_x=0, first_burst_y=50, gap=150, width=50, shape="exp")
    limbo.noise_part(min_noise_y=1, max_noise_y=4, distribution="normal")
    ts = limbo.merge(add_or_mul="add")
    limbo.build_workload_intensity(ts=ts, start_t="2021-07-09 00:00:00", t_interval=3600,
                                   png=filename_config.limbo_intensity_png, pkl=filename_config.limbo_intensity_pkl)
