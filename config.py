"""
雷达多目标智能探测与识别 - 最终版配置文件
数据路径指向final/data目录
"""
import numpy as np

# ============================================================
# 雷达系统参数
# ============================================================
C = 3e8
FC = 150e6
B = 5e6
TP = 20e-6
PRF = 500
FS = 10e6
LAMBDA = C / FC

N_PULSES = 300
N_RANGE_GATES = 8000

RANGE_MIN = 65000
RANGE_MAX = 95000
VELOCITY_MIN = -1000
VELOCITY_MAX = -800
ACCEL_MIN = -100
ACCEL_MAX = -85

DELTA_R = C / (2 * B)
R_UNAMB = N_RANGE_GATES * C / (2 * FS)
V_UNAMB = LAMBDA * PRF / 4
V_AMB_INTERVAL = LAMBDA * PRF / 2
DELTA_V = LAMBDA * PRF / (2 * N_PULSES)
T_OBS = N_PULSES / PRF

# 加速度搜索 (优化版: 256点超细网格, 边界扩展3m/s^2)
ACCEL_SEARCH_NUM = 256
ACCEL_SEARCH_RANGE = (-108, -77)

# ============================================================
# CFAR 检测参数
# 调优: 保护窗适当增大防泄漏, 参考窗缩小增强局部适应
# ============================================================
CFAR_GUARD_RANGE = 5
CFAR_GUARD_DOPPLER = 5
CFAR_REF_RANGE = 10
CFAR_REF_DOPPLER = 10
CFAR_PFA = 1e-6
CFAR_SNR_THRESHOLD = 5.0
# OS-CFAR rank比例: 0.70更敏感(低虚警场景), 0.80更稳健(密集目标)
OS_RANK_RATIO = 0.70

CLUSTER_RANGE_BINS = 3
CLUSTER_DOPPLER_BINS = 8

# ============================================================
# 航迹跟踪参数
# 调优: 收紧确认/删除逻辑, 减少虚假起批
# ============================================================
# PDAF初始门 (马氏距离单位)
PDAF_INITIAL_GATE = 4.5
# 点云聚类门 (sin/cos聚类)
CLUSTER_EPS_RANGE = 35
CLUSTER_EPS_AZ = 3.5
CLUSTER_EPS_EL = 3.5
CLUSTER_EPS_VEL = 18.0

# 航迹管理: M/N确认逻辑
TRACK_CONFIRM_FRAMES = 4   # 确认需>=4次命中
TRACK_DELETE_FRAMES = 4    # 丢失计数>=4则删除
MIN_TRACK_LENGTH = 15
TENTATIVE_MAX_AGE = 10     # 未确认航迹最大存活帧数
TENTATIVE_MAX_MISSES = 3   # 未确认航迹最大丢失次数

# 航迹合并阈值
MERGE_RANGE_THRESH = 45
MERGE_AZ_THRESH = 4
MERGE_EL_THRESH = 2.5
MERGE_VEL_THRESH = 8

KALMAN_RANGE_NOISE = 10.0
KALMAN_AZ_NOISE = 0.5
KALMAN_EL_NOISE = 0.5
KALMAN_PROCESS_NOISE = 5.0

MIN_SNR_THRESHOLD = 50
DOPPLER_VALID_FILTER = True

# ============================================================
# 路径配置 (final/data目录)
# ============================================================
BASE_DIR = r'C:\Users\ASUS\OneDrive\Desktop\睿创杯赛题4\agant'
DATA_DIR = f'{BASE_DIR}/data'
OUTPUT_DIR = f'{BASE_DIR}/output'

TASK_A_DATA = f'{DATA_DIR}/a)_Target_dataset.mat'
TRACK_A_DATA = f'{DATA_DIR}/b)_Track_A_dataset.csv'
TRACK_B_DATA = f'{DATA_DIR}/b)_Track_B_dataset.csv'

# 任务C路径
TASKC_TRAIN_DIR = f'{DATA_DIR}/train'
TASKC_TEST_DIR = f'{DATA_DIR}/test'
TASKC_SUBMISSION = f'{OUTPUT_DIR}/TaskC_submission_final.csv'

# 分类标签
TASKC_CLASSES = ['Bird', 'Fixed-wing UAV', 'Helicopter', 'Passenger ship', 'Rotary drone', 'Speedboat']
TASKC_FEATURES = ['A_m', 'E_m', 'R_m', 'SNR', 'V_m']
