class FilenameConfig:

    def __init__(self):

        # service/cluster/xx
        # 用作synoptic输入的.log文件的base, 从0开始编号。
        # e.g., hipstershop_userbehavior_0.log、hipstershop_userbehavior_1.log
        self.synoptic_input_base = "hipstershop_userbehavior_"
        # 根据聚类结果，存储behavior mix
        self.behavior_mix_pkl = "behavior_mix.pkl"

        # service/intensity/xx
        # 由原始日志生成的intensity.pkl以及绘制的png
        self.log_intensity_pkl = "intensity_raw.pkl"
        self.log_intensity_png = "intensity_raw.png"
        # 由limbo方法生成的intensity_limbo.pkl以及绘制的png
        self.limbo_intensity_pkl = "intensity_limbo.pkl"
        self.limbo_intensity_png = "intensity_limbo.png"
        # 由TSAGen方法生成的intensity_tsagen.pkl以及绘制的png
        self.tsagen_intensity_pkl = "intensity_tsagen.pkl"
        self.tsagen_intensity_png = "intensity_tsagen.png"

        # service/transition/xx
        # SynopticTransitionService.py生成模拟流量存储位置
        self.mock_workload_txt = "mock_workload.txt"
        # SynopticTransitionService.__evaluate()生成.png形式的behavior数量分布图
        self.behavior_length_distribution_png = "length_distribution.png"



filename_config = FilenameConfig()