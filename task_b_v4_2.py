"""
任务B v4-2: PDAF-KF + UCM + DBSCAN凝聚 (DBSCAN condensation variant)

架构:
  v1的PDAF-KF核心 (6D极坐标, 标准KF, 概率数据关联)
  + v4-1的增强预处理 (SNR/DBSCAN可选凝聚, 距离环抑制, UCM无偏转换)
  + 增强后处理 (3D距离合并, 命中率过滤, 仅输出命中帧)

v4-2新增:
  - dbscan_condensation(): 马氏距离DBSCAN, σ自适应缩放, 噪声点自动丢弃
  - CONDENSATION_METHOD 开关: 'snr' | 'dbscan'
"""

import numpy as np
import pandas as pd
import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from scipy.linalg import cholesky
from scipy.spatial import KDTree
from config import *
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 雷达传感器噪声参数
# ============================================================
SIG_R = 5.0
SIG_THETA = np.deg2rad(0.25)
SIG_PHI = np.deg2rad(0.16)
SIG_VD = 1.0

# ============================================================
# PDAF / 跟踪参数 (固定, 不过度自适应)
# ============================================================
PDAF_PD = 0.9           # 检测概率
PDAF_PG = 0.99          # 门概率
PDAF_GATE = 5.0         # 初始马氏距离平方门限
KALMAN_R_NOISE = 10.0   # 距离测量噪声 (m)
KALMAN_AZ_NOISE = 0.5   # 方位测量噪声 (deg)
KALMAN_EL_NOISE = 0.5   # 俯仰测量噪声 (deg)
PROCESS_NOISE_BASE = 5.0

# 航迹管理 (固定参数)
TRACK_CONFIRM_HITS = 4     # 确认需命中数
TRACK_DELETE_MISSES = 5    # 连续漏检删除数 (比v1的4更保守)
MIN_TRACK_LENGTH = 15      # 最低航迹长度
MIN_TRACK_HITS = 7         # 最低命中数 (v4r1: 8→7)
MIN_HIT_RATIO = 0.33       # 最低命中率 (对齐v4-35分: 0.33)
TENTATIVE_MAX_AGE = 10     # 试探航迹最大存活
TENTATIVE_MAX_MISSES = 3   # 试探航迹最大漏检
MAX_TENTATIVE_PER_FRAME = 5

# 远场起批参数 (M/N逻辑: N中M)
FAR_CONFIRM_N = 6          # 远场观察窗口
FAR_CONFIRM_M = 2          # 远场最少命中
NEAR_CONFIRM_N = 4         # 近场观察窗口
NEAR_CONFIRM_M = 3         # 近场最少命中
DIST_NEAR_FAR = 3000.0     # 近/远场分界

# 合并参数
MERGE_3D_DIST = 500.0      # 首尾衔接3D距离 (对齐v4-35分: 500)
MERGE_OVERLAP_3D = 800.0   # 重叠航迹3D距离 (对齐v4-35分: 800)
MERGE_AZ_TH = 6.0          # 合并方位差 (deg)
MERGE_EL_TH = 4.0          # 合并俯仰差 (deg)
MERGE_VEL_TH = 12.0        # 合并速度差 (m/s)

# 凝聚方法开关
CONDENSATION_METHOD = 'dbscan'  # 'snr' | 'dbscan'
DBSCAN_EPS = 2.0               # DBSCAN 邻域半径 (马氏距离)
DBSCAN_MIN_SAMPLES = 1         # DBSCAN 最小邻居数

# UCM相关
DIST_NEAR_UCM = 2000.0


# ============================================================
# 1. UCM 无偏坐标转换
# ============================================================
def ucm_transform(r, theta, phi, sig_r=SIG_R, sig_theta=SIG_THETA, sig_phi=SIG_PHI):
    lambda_combined = np.exp(-(sig_theta**2 + sig_phi**2) / 2)
    lambda_phi = np.exp(-sig_phi**2 / 2)

    x = (r * np.cos(phi) * np.cos(theta)) / lambda_combined
    y = (r * np.cos(phi) * np.sin(theta)) / lambda_combined
    z = (r * np.sin(phi)) / lambda_phi

    J = np.array([
        [np.cos(phi)*np.cos(theta), -r*np.cos(phi)*np.sin(theta), -r*np.sin(phi)*np.cos(theta)],
        [np.cos(phi)*np.sin(theta),  r*np.cos(phi)*np.cos(theta), -r*np.sin(phi)*np.sin(theta)],
        [np.sin(phi),                0,                            r*np.cos(phi)]
    ])
    P_polar = np.diag([sig_r**2, sig_theta**2, sig_phi**2])
    R_cart = J @ P_polar @ J.T
    return x, y, z, R_cart


# ============================================================
# 2. 数据读取与预处理
# ============================================================
def read_track_data(filepath):
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except:
        return pd.read_csv(filepath, encoding='gb18030')


def extract_columns(df):
    mapping = {}
    for cl in df.columns:
        lo = cl.lower()
        if 'fpga' in lo and '时间' in cl:
            mapping['time'] = cl
        elif '点迹距离' in cl:
            mapping['range'] = cl
        elif '点迹方位' in cl:
            mapping['azimuth'] = cl
        elif '点迹俯仰' in cl:
            mapping['elevation'] = cl
        elif '和路强度' in cl:
            mapping['intensity'] = cl
        elif '点迹速度' in cl:
            mapping['velocity'] = cl
        elif '点迹多普勒标志' in cl:
            mapping['doppler_flag'] = cl
        elif '距离基底' in cl:
            mapping['range_base'] = cl
    return df.rename(columns={v: k for k, v in mapping.items()})


def group_frames_by_time(df, gap_threshold=1500):
    df = df.sort_values('time').reset_index(drop=True)
    times = df['time'].values
    boundaries = np.where(np.diff(times) > gap_threshold)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(df)]])
    frames = {}
    for i in range(len(boundaries) - 1):
        frames[i] = df.iloc[boundaries[i]:boundaries[i+1]]
    return frames


def filter_physical(df):
    cond = np.ones(len(df), dtype=bool)
    if 'range' in df.columns:
        cond &= (df['range'] >= 30.0) & (df['range'] <= 100000.0)
    if 'range' in df.columns and 'elevation' in df.columns:
        z_approx = df['range'].values * np.sin(np.deg2rad(df['elevation'].values))
        z_min = np.where(df['range'].values < 1000, -100, -1000)
        cond &= (z_approx >= z_min) & (z_approx <= 25000)
    if 'velocity' in df.columns:
        cond &= (np.abs(df['velocity'].values) >= 0.5) & (np.abs(df['velocity'].values) <= 1500.0)
    if 'doppler_flag' in df.columns:
        sample = str(df['doppler_flag'].values[0]) if len(df) > 0 else ''
        if '有效' in sample or '無效' in sample:
            cond &= df['doppler_flag'].str.contains('有效', na=False)
    return df[cond].copy()


def distance_ring_suppression(df, ring_bin_size=100.0, ring_points_th=3, ring_v_static=1.5):
    if len(df) == 0:
        return df
    r_vals = df['range'].values
    v_vals = np.abs(df['velocity'].values) if 'velocity' in df.columns else np.zeros(len(df))
    ring_edges = np.arange(0, r_vals.max() + ring_bin_size, ring_bin_size)
    bin_idx = np.digitize(r_vals, ring_edges) - 1
    weights = np.ones(len(df))
    for b in np.unique(bin_idx):
        in_bin = bin_idx == b
        n_bin = in_bin.sum()
        if n_bin >= ring_points_th:
            v_avg = np.mean(v_vals[in_bin])
            if v_avg < ring_v_static:
                weights[in_bin] = 0.1
    df = df.copy()
    df['_weight'] = weights
    return df


def snr_based_condensation(df, dr=20.0, d_cross_th=40.0, d_cross_ph=40.0, dvd_max=15.0):
    """基于SNR排序的能量加权凝聚 (from v2)"""
    if len(df) == 0:
        return []
    r_f = df['range'].values
    th_f = np.deg2rad(df['azimuth'].values)
    ph_f = np.deg2rad(df['elevation'].values)
    vd_f = df['velocity'].values if 'velocity' in df.columns else np.zeros(len(df))
    int_f = df['intensity'].values if 'intensity' in df.columns else np.ones(len(df))
    weights = df['_weight'].values if '_weight' in df.columns else np.ones(len(df))
    if 'range_base' in df.columns:
        snr_f = int_f / np.maximum(df['range_base'].values, 1.0)
    else:
        snr_f = int_f / np.maximum(np.median(int_f[int_f > 0]) if np.any(int_f > 0) else 1.0, 1.0)

    sort_idx = np.argsort(snr_f)[::-1]
    visited = np.zeros(len(df), dtype=bool)
    N = len(df)
    centroids = []
    for idx in range(N):
        i = sort_idx[idx]
        if visited[i]:
            continue
        ri = r_f[i]; thi = th_f[i]; phi = ph_f[i]; vdi = vd_f[i]
        diff_r = np.abs(r_f - ri)
        diff_th = np.abs(np.arctan2(np.sin(th_f - thi), np.cos(th_f - thi)))
        diff_ph = np.abs(np.arctan2(np.sin(ph_f - phi), np.cos(ph_f - phi)))
        diff_vd = np.abs(vd_f - vdi)
        cross_th = ri * diff_th
        cross_ph = ri * diff_ph
        neighbors = (~visited) & (diff_r < dr) & (cross_th < d_cross_th) & \
                    (cross_ph < d_cross_ph) & (diff_vd < dvd_max)
        cluster_idx = np.where(neighbors)[0]
        if len(cluster_idx) == 0:
            continue
        visited[cluster_idx] = True
        cluster_int = int_f[cluster_idx]
        cluster_w = weights[cluster_idx]
        sum_w = (cluster_int * cluster_w).sum()
        if sum_w <= 0:
            r_cent = np.mean(r_f[cluster_idx])
            vd_cent = np.mean(vd_f[cluster_idx])
            sin_th = np.mean(np.sin(th_f[cluster_idx]))
            cos_th = np.mean(np.cos(th_f[cluster_idx]))
            th_cent = np.arctan2(sin_th, cos_th)
            sin_ph = np.mean(np.sin(ph_f[cluster_idx]))
            cos_ph = np.mean(np.cos(ph_f[cluster_idx]))
            ph_cent = np.arctan2(sin_ph, cos_ph)
        else:
            weighted_int = cluster_int * cluster_w
            r_cent = np.sum(r_f[cluster_idx] * weighted_int) / sum_w
            vd_cent = np.sum(vd_f[cluster_idx] * weighted_int) / sum_w
            sin_th = np.sum(np.sin(th_f[cluster_idx]) * weighted_int) / sum_w
            cos_th = np.sum(np.cos(th_f[cluster_idx]) * weighted_int) / sum_w
            th_cent = np.arctan2(sin_th, cos_th)
            sin_ph = np.sum(np.sin(ph_f[cluster_idx]) * weighted_int) / sum_w
            cos_ph = np.sum(np.cos(ph_f[cluster_idx]) * weighted_int) / sum_w
            ph_cent = np.arctan2(sin_ph, cos_ph)
        avg_snr = np.mean(snr_f[cluster_idx])
        centroids.append({
            'range': r_cent, 'azimuth': np.rad2deg(th_cent), 'elevation': np.rad2deg(ph_cent),
            'velocity': vd_cent, 'snr': avg_snr,
            'n_points': len(cluster_idx),
            'weight_sum': cluster_w.sum(),
            'time': df['time'].values[cluster_idx[0]] if 'time' in df.columns else 0
        })
    return centroids


def dbscan_condensation(df, eps=2.0, min_samples=1):
    """基于马氏距离的 DBSCAN 凝聚 — σ自动随距离缩放, 噪声点自动丢弃

    特征空间归一化: d² = (Δr/σ_r)² + (cross_Δaz/σ_cross)² + (cross_Δel/σ_ph)² + (Δv/σ_v)²
      σ_r = 5.0m
      σ_cross = r * sin(0.25°) ≈ 0.00436*r  (自适应)
      σ_ph    = r * sin(0.16°) ≈ 0.00279*r  (自适应)
      σ_v = 1.0 m/s
    eps=2.0 → 2σ 邻域. min_samples=1 → 允许孤立点成簇 (由后续置信度过滤).
    """
    if len(df) == 0:
        return []

    r_f = df['range'].values
    th_f = np.deg2rad(df['azimuth'].values)
    ph_f = np.deg2rad(df['elevation'].values)
    vd_f = df['velocity'].values if 'velocity' in df.columns else np.zeros(len(df))
    int_f = df['intensity'].values if 'intensity' in df.columns else np.ones(len(df))
    weights = df['_weight'].values if '_weight' in df.columns else np.ones(len(df))

    sig_theta_rad = np.deg2rad(0.25)
    sig_phi_rad = np.deg2rad(0.16)
    sig_cross = r_f * np.sin(sig_theta_rad)
    sig_ph = r_f * np.sin(sig_phi_rad)

    # 马氏归一化特征: cross_az = r * Δθ
    features = np.column_stack([
        r_f / SIG_R,
        (r_f * th_f) / sig_cross,
        (r_f * ph_f) / sig_ph,
        vd_f / SIG_VD,
    ])

    tree = KDTree(features)
    neighbors = tree.query_ball_point(features, r=eps)

    # 简单 DBSCAN
    n = len(features)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighbors[i]) < min_samples:
            continue

        labels[i] = cluster_id
        queue = list(neighbors[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                if len(neighbors[j]) >= min_samples:
                    queue.extend(neighbors[j])
        cluster_id += 1

    # 逐簇 SNR 加权质心
    centroids = []
    for cid in range(cluster_id):
        mask = labels == cid
        idx = np.where(mask)[0]

        cluster_int = int_f[idx]
        cluster_w = weights[idx]
        sum_w = (cluster_int * cluster_w).sum()

        if sum_w <= 0:
            r_cent = np.mean(r_f[idx])
            vd_cent = np.mean(vd_f[idx])
            th_cent = np.arctan2(np.mean(np.sin(th_f[idx])), np.mean(np.cos(th_f[idx])))
            ph_cent = np.arctan2(np.mean(np.sin(ph_f[idx])), np.mean(np.cos(ph_f[idx])))
        else:
            w = cluster_int * cluster_w
            r_cent = np.sum(r_f[idx] * w) / sum_w
            vd_cent = np.sum(vd_f[idx] * w) / sum_w
            th_cent = np.arctan2(np.sum(np.sin(th_f[idx]) * w) / sum_w,
                                 np.sum(np.cos(th_f[idx]) * w) / sum_w)
            ph_cent = np.arctan2(np.sum(np.sin(ph_f[idx]) * w) / sum_w,
                                 np.sum(np.cos(ph_f[idx]) * w) / sum_w)

        if 'range_base' in df.columns:
            snr_vals = int_f[idx] / np.maximum(df['range_base'].values[idx], 1.0)
        else:
            ref = np.median(int_f[int_f > 0]) if np.any(int_f > 0) else 1.0
            snr_vals = int_f[idx] / max(ref, 1.0)

        centroids.append({
            'range': r_cent, 'azimuth': np.rad2deg(th_cent), 'elevation': np.rad2deg(ph_cent),
            'velocity': vd_cent, 'snr': np.mean(snr_vals),
            'n_points': len(idx),
            'weight_sum': cluster_w.sum(),
            'time': df['time'].values[idx[0]] if 'time' in df.columns else 0
        })

    return centroids


def confidence_filter(centroids):
    """近场/远场置信度判决"""
    filtered = []
    for pt in centroids:
        r = pt['range']
        snr = pt['snr']
        w_sum = pt.get('weight_sum', 1.0)
        n_pts = pt.get('n_points', 1)
        if snr < 0.5 or w_sum < 0.1:
            continue
        if r < DIST_NEAR_UCM:
            if snr >= 1.0:
                conf = 2 if (snr >= 5.0 and n_pts >= 2) else 1
                pt['confidence'] = conf
                filtered.append(pt)
        else:
            if snr >= 1.5 or (snr >= 0.8 and n_pts >= 2):
                pt['confidence'] = 2 if snr >= 3.0 else 1
                filtered.append(pt)
    return filtered


# ============================================================
# 3. 杂波密度估计 (from v1)
# ============================================================
def estimate_clutter_density(measurements, gate_volume_factor=50.0):
    if len(measurements) == 0:
        return 1e-5
    n_meas = len(measurements)
    if n_meas >= 2:
        r_vals = [m['range'] for m in measurements]
        az_vals = [m['azimuth'] for m in measurements]
        el_vals = [m['elevation'] for m in measurements]
        r_span = max(abs(np.ptp(r_vals)), 100)
        az_span = max(abs(np.ptp(az_vals)), 1)
        el_span = max(abs(np.ptp(el_vals)), 1)
    else:
        r_span, az_span, el_span = 500, 10, 5
    volume = r_span * az_span * el_span * gate_volume_factor
    lambda_est = n_meas / max(volume, 1.0)
    return min(lambda_est, 5e-5)


# ============================================================
# 4. PDAF-KF 跟踪器 (6D极坐标, from v1 增强)
# ============================================================
class PDAF_KF_Tracker:
    """PDAF + 标准KF, 6D极坐标状态 [r, vr, az, v_az, el, v_el]"""

    def __init__(self, init_meas, dt=3.6):
        self.dt = dt
        r0 = init_meas['range']
        v0 = init_meas.get('velocity', 0.0)

        # 6D状态初始化
        self.x = np.array([r0, v0,
                           init_meas['azimuth'], 0.0,
                           init_meas['elevation'], 0.0])

        # 初始协方差
        r_var = max(100, (r0 * 0.005)**2)
        self.P = np.diag([r_var, 25.0, 2.0, 0.5, 2.0, 0.5])

        # 状态转移矩阵 (CV模型)
        self.F = np.eye(6)
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt

        # 过程噪声 (基于距离自适应基值, 但不用自适应缩放)
        q_r = max(1.0, r0 / 2000.0)
        G = np.array([
            [dt**2/2, 0, 0], [dt, 0, 0],
            [0, dt**2/2, 0], [0, dt, 0],
            [0, 0, dt**2/2], [0, 0, dt],
        ])
        Q_compact = np.diag([q_r, 0.1, 0.1])
        self.Q = G @ Q_compact @ G.T

        # 测量矩阵 (直接取子集)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0  # r
        self.H[1, 2] = 1.0  # az
        self.H[2, 4] = 1.0  # el

        # 测量噪声
        r_noise = KALMAN_R_NOISE * (1 + r0 / 100000.0)
        self.R = np.diag([r_noise**2, KALMAN_AZ_NOISE**2, KALMAN_EL_NOISE**2])

        # 波门
        self.gate_size = PDAF_GATE

        # 航迹状态
        self.id = -1
        self.hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.age = 1
        self._confirmed = False

        # 速度历史
        self._prev_velocity = v0
        self._velocity_history = [v0]

        # 历史记录
        self.history = [self.x.copy()]
        self.P_history = [self.P.copy()]
        self.x_pred_history = []
        self.P_pred_history = []
        self.t_history = [init_meas.get('time', 0)]
        self.z_history = []
        self._hit_frames = [True]

        # 质量评分
        self.quality_score = 1.0

    def predict(self):
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        self.P_pred = 0.5 * (self.P_pred + self.P_pred.T)

        # 保证正定性
        try:
            eig_min = np.linalg.eigvalsh(self.P_pred).min()
            if eig_min < 1e-10:
                self.P_pred += np.eye(6) * (1e-8 - eig_min)
        except:
            pass

        # 状态裁剪
        self.x_pred[0] = max(0.0, min(20000.0, self.x_pred[0]))
        self.x_pred[2] = self.x_pred[2] % 360.0
        self.x_pred[4] = max(-90.0, min(90.0, self.x_pred[4]))

        self.x_pred_history.append(self.x_pred.copy())
        self.P_pred_history.append(self.P_pred.copy())

    def mahalanobis(self, z):
        """计算马氏距离"""
        nu = z - self.H @ self.x_pred
        S = self.H @ self.P_pred @ self.H.T + self.R
        nu[1] = (nu[1] + 180) % 360 - 180  # 角度wrap
        try:
            d2 = nu @ np.linalg.solve(S, nu)
        except np.linalg.LinAlgError:
            d2 = 1e9
        return d2, nu, S

    def velocity_gate(self, z_vel):
        """速度一致性门"""
        if abs(self._prev_velocity) < 0.1:
            return True
        vel_change = abs(z_vel - self._prev_velocity)
        expected_change = 25.0 + 0.15 * abs(self._prev_velocity)
        return vel_change < expected_change

    def pdaf_update(self, measurements, meas_indices, clutter_lambda):
        """PDAF概率数据关联更新"""
        m_k = len(meas_indices)
        if m_k == 0:
            self.miss()
            return

        n_z = 3
        gate = self.gate_size

        # 新息协方差 (用第一个量测计算的S, 所有量测共用)
        z0 = np.array([measurements[meas_indices[0]]['range'],
                       measurements[meas_indices[0]]['azimuth'],
                       measurements[meas_indices[0]]['elevation']])
        _, _, S = self.mahalanobis(z0)
        det_S = max(np.linalg.det(S), 1e-15)
        inv_S = np.linalg.inv(S)
        V_gate = (gate * np.pi / 180)**3 * np.sqrt(det_S)

        # 杂波项
        lambda_eff = max(clutter_lambda, 1e-7)
        clutter_term = (2 * np.pi)**(n_z / 2) * lambda_eff * V_gate * (1 - PDAF_PD * PDAF_PG) / PDAF_PD

        # 各量测关联概率
        e = np.zeros(m_k)
        innovations = []
        for i_idx, mi in enumerate(meas_indices):
            z = np.array([measurements[mi]['range'],
                          measurements[mi]['azimuth'],
                          measurements[mi]['elevation']])
            z_vel = measurements[mi].get('velocity', 0.0)
            if not self.velocity_gate(z_vel):
                e[i_idx] = 0.0
                innovations.append(np.zeros(3))
                continue
            d2, nu, _ = self.mahalanobis(z)
            e[i_idx] = np.exp(-0.5 * d2)
            innovations.append(nu)

        e_sum = clutter_term + e.sum()
        if e_sum < 1e-15:
            self.miss()
            return
        beta = e / e_sum

        # 组合新息
        nu_combined = np.zeros(n_z)
        for i, nu in enumerate(innovations):
            nu_combined += beta[i] * nu

        # 卡尔曼增益
        K = self.P_pred @ self.H.T @ inv_S

        # 状态更新
        self.x = self.x_pred + K @ nu_combined

        # Joseph形式协方差
        I_KH = np.eye(6) - K @ self.H
        P_joseph = I_KH @ self.P_pred @ I_KH.T + K @ self.R @ K.T

        # PDAF协方差展开
        P_c = I_KH @ self.P_pred
        nu_spread = np.zeros((6, 6))
        for i, nu in enumerate(innovations):
            Knu = K @ nu
            nu_spread += beta[i] * np.outer(Knu, Knu)
        nu_spread -= np.outer(K @ nu_combined, K @ nu_combined)
        P_pdaf = P_c + nu_spread

        self.P = 0.7 * P_joseph + 0.3 * P_pdaf
        self.P = 0.5 * (self.P + self.P.T)

        # 保正定
        try:
            eig_min = np.linalg.eigvalsh(self.P).min()
            if eig_min < 1e-10:
                self.P = P_joseph
                self.P = 0.5 * (self.P + self.P.T)
        except:
            self.P = P_joseph
            self.P = 0.5 * (self.P + self.P.T)

        # 状态裁剪
        self.x[0] = max(0.0, min(20000.0, self.x[0]))
        self.x[2] = self.x[2] % 360.0
        self.x[4] = max(-90.0, min(90.0, self.x[4]))
        self.x[1] = max(-200.0, min(200.0, self.x[1]))
        self.x[3] = max(-30.0, min(30.0, self.x[3]))
        self.x[5] = max(-30.0, min(30.0, self.x[5]))

        # 速度历史
        new_vel = self.x[1]
        self._prev_velocity = new_vel
        self._velocity_history.append(new_vel)

        # 状态更新
        self.hits += 1
        self.consecutive_misses = 0
        self.misses = 0
        self.age += 1
        self.history.append(self.x.copy())
        self.P_history.append(self.P.copy())
        self._hit_frames.append(True)

        best_mi = meas_indices[np.argmax(beta)]
        self.z_history.append(np.array([measurements[best_mi]['range'],
                                        measurements[best_mi]['azimuth'],
                                        measurements[best_mi]['elevation']]))
        self.t_history.append(measurements[best_mi].get('time', 0))

        # 质量评分
        hit_ratio = self.hits / max(self.age, 1)
        vel_consistency = 1.0 / (1.0 + np.std(self._velocity_history[-min(10, len(self._velocity_history)):]))
        self.quality_score = 0.5 * hit_ratio + 0.3 * vel_consistency + 0.2 * (1.0 / (1.0 + self.misses))

        # 波门收窄 (稳定航迹)
        if self.hits > 15:
            self.gate_size = max(3.0, self.gate_size * 0.98)
        elif self.hits > 8:
            self.gate_size = max(3.5, self.gate_size * 0.99)

    def miss(self):
        """外推 (coast)"""
        self.misses += 1
        self.consecutive_misses += 1
        self.age += 1
        self.quality_score *= 0.85
        self.gate_size = min(7.0, self.gate_size * 1.15)
        self.history.append(self.x.copy())
        self.P_history.append(self.P.copy())
        self.t_history.append(0)
        self._hit_frames.append(False)
        self._velocity_history.append(self._prev_velocity)

    def rts_smooth(self):
        """Rauch-Tung-Striebel 后向平滑"""
        n = len(self.history)
        if n < 2:
            self.history_smoothed = self.history.copy()
            return
        self.history_smoothed = [self.history[-1].copy()]
        for k in range(n - 2, -1, -1):
            if k < len(self.x_pred_history) and k < len(self.P_pred_history):
                x_pred = self.x_pred_history[k]
                P_pred = self.P_pred_history[k]
                try:
                    P_pred_reg = P_pred + np.eye(6) * 1e-6
                    G = self.P_history[k] @ self.F.T @ np.linalg.inv(P_pred_reg)
                except np.linalg.LinAlgError:
                    self.history_smoothed.insert(0, self.history[k].copy())
                    continue
                x_s = self.history[k] + G @ (self.history_smoothed[0] - x_pred)
                x_s[0] = max(0.0, min(20000.0, x_s[0]))
                x_s[2] = x_s[2] % 360.0
                x_s[4] = max(-90.0, min(90.0, x_s[4]))
                x_s[1] = max(-200.0, min(200.0, x_s[1]))
                x_s[3] = max(-30.0, min(30.0, x_s[3]))
                x_s[5] = max(-30.0, min(30.0, x_s[5]))
                self.history_smoothed.insert(0, x_s)
            else:
                self.history_smoothed.insert(0, self.history[k].copy())


# ============================================================
# 5. 航迹管理器 (PDAF关联 + M/N起批 + 保守删除)
# ============================================================
class TrackManager:
    """航迹管理器: PDAF关联 + 试探航迹 + 确认/删除"""

    def __init__(self, dt=3.6):
        self.dt = dt
        self.tentative_tracks = []   # 试探航迹 (PDAF_KF_Tracker)
        self.confirmed_tracks = []   # 确认航迹
        self.dead_tracks = []        # 已删除航迹
        self.next_id = 1

    def process_frame(self, measurements):
        clutter_lambda = estimate_clutter_density(measurements)

        all_tracks = self.tentative_tracks + self.confirmed_tracks

        # 无量测, 全部外推
        if len(measurements) == 0:
            for track in all_tracks:
                track.miss()
            self._manage_lifecycle()
            return

        # 预测
        for track in all_tracks:
            try:
                track.predict()
            except:
                track.misses = TRACK_DELETE_MISSES + 1
                continue

        # 构建马氏距离代价矩阵
        n_tracks = len(all_tracks)
        n_meas = len(measurements)
        cost_matrix = np.full((n_tracks, n_meas), np.inf)

        for ti, track in enumerate(all_tracks):
            for mi, m in enumerate(measurements):
                z = np.array([m['range'], m['azimuth'], m['elevation']])
                d2, _, _ = track.mahalanobis(z)
                if d2 < track.gate_size:
                    cost_matrix[ti, mi] = d2

        # 关联: 贪心最优分配 (每个量测配给最佳航迹, 但收集所有波门内量测供PDAF)
        assigned_meas = set()
        assigned_tracks = set()

        # 按代价排序分配
        flat_indices = np.argsort(cost_matrix.ravel())
        for flat_idx in flat_indices:
            ti, mi = np.unravel_index(flat_idx, cost_matrix.shape)
            if cost_matrix[ti, mi] == np.inf:
                continue
            if ti in assigned_tracks or mi in assigned_meas:
                continue
            assigned_tracks.add(ti)
            assigned_meas.add(mi)

        # PDAF更新: 每条已分配航迹用其波门内所有量测
        for ti in assigned_tracks:
            all_in_gate = []
            for mi in range(n_meas):
                if cost_matrix[ti, mi] < all_tracks[ti].gate_size and cost_matrix[ti, mi] != np.inf:
                    all_in_gate.append(mi)
            if len(all_in_gate) > 0:
                all_tracks[ti].pdaf_update(measurements, all_in_gate, clutter_lambda)
            else:
                all_tracks[ti].miss()

        # 未分配航迹: 外推
        for ti in range(n_tracks):
            if ti not in assigned_tracks:
                all_tracks[ti].miss()

        # 新增试探航迹: 未被任何航迹波门覆盖的量测
        all_gated = set()
        for ti in range(n_tracks):
            for mi in range(n_meas):
                if cost_matrix[ti, mi] < all_tracks[ti].gate_size:
                    all_gated.add(mi)

        unassigned = [m for i, m in enumerate(measurements)
                      if i not in all_gated and m.get('n_points', 1) >= 2]

        # 空间去重: 新试探不与已有航迹过近
        new_count = 0
        for m in unassigned:
            if new_count >= MAX_TENTATIVE_PER_FRAME:
                break
            m_cart = np.array(ucm_transform(m['range'], np.deg2rad(m['azimuth']), np.deg2rad(m['elevation']))[:3])

            too_close = False
            for ct in self.confirmed_tracks:
                ct_cart = np.array(ucm_transform(ct.x[0], np.deg2rad(ct.x[2]), np.deg2rad(ct.x[4]))[:3])
                gate = 500.0 if ct.x[0] > 4000 else 300.0
                if np.linalg.norm(m_cart - ct_cart) < gate:
                    too_close = True
                    break
            if not too_close:
                for tt in self.tentative_tracks:
                    tt_cart = np.array(ucm_transform(tt.x[0], np.deg2rad(tt.x[2]), np.deg2rad(tt.x[4]))[:3])
                    if np.linalg.norm(m_cart - tt_cart) < 150.0:
                        too_close = True
                        break
            if too_close:
                continue

            track = PDAF_KF_Tracker(m, self.dt)
            track.id = self.next_id
            self.next_id += 1
            track._confirmed = False
            self.tentative_tracks.append(track)
            new_count += 1

        self._manage_lifecycle()

    def _manage_lifecycle(self):
        """航迹生命周期管理: 试探→确认, 确认→删除"""

        # 试探→确认: M/N逻辑
        newly_confirmed = []
        for track in self.tentative_tracks:
            r = track.x[0]
            if r > DIST_NEAR_FAR:
                # 远场: N帧中M次
                if track.hits >= FAR_CONFIRM_M and track.age <= FAR_CONFIRM_N:
                    track._confirmed = True
                    newly_confirmed.append(track)
                elif track.hits >= NEAR_CONFIRM_M:  # 宽容兜底
                    track._confirmed = True
                    newly_confirmed.append(track)
            else:
                # 近场: 需命中>=4
                if track.hits >= TRACK_CONFIRM_HITS:
                    track._confirmed = True
                    newly_confirmed.append(track)

        for track in newly_confirmed:
            if track in self.tentative_tracks:
                self.tentative_tracks.remove(track)
                # 空间去重: 确认前再次检查 (自适应门限)
                track_cart = np.array(ucm_transform(track.x[0], np.deg2rad(track.x[2]), np.deg2rad(track.x[4]))[:3])
                duplicate = False
                for ct in self.confirmed_tracks:
                    ct_cart = np.array(ucm_transform(ct.x[0], np.deg2rad(ct.x[2]), np.deg2rad(ct.x[4]))[:3])
                    gate = 500.0 if ct.x[0] > 4000 else 300.0
                    if np.linalg.norm(track_cart - ct_cart) < gate:
                        duplicate = True
                        break
                if not duplicate:
                    self.confirmed_tracks.append(track)

        # 试探航迹老化删除
        self.tentative_tracks = [t for t in self.tentative_tracks
                                 if t.consecutive_misses < TENTATIVE_MAX_MISSES
                                 and t.age < TENTATIVE_MAX_AGE]

        # 确认航迹删除 (连续漏检 >= TRACK_DELETE_MISSES)
        alive = []
        for track in self.confirmed_tracks:
            if track.consecutive_misses >= TRACK_DELETE_MISSES:
                self.dead_tracks.append(track)
            else:
                alive.append(track)
        self.confirmed_tracks = alive

    def get_valid_tracks(self):
        """提取有效航迹, 经长度+命中数+命中率过滤"""
        all_tracks = self.confirmed_tracks + self.dead_tracks
        valid = []
        for t in all_tracks:
            n = len(t.history)
            if n < MIN_TRACK_LENGTH:
                continue
            if t.hits < MIN_TRACK_HITS:
                continue
            hit_ratio = t.hits / max(n, 1)
            if hit_ratio < MIN_HIT_RATIO:
                continue
            valid.append(t)
        valid.sort(key=lambda t: t.quality_score * t.hits, reverse=True)
        return valid


# ============================================================
# 6. 航迹合并 (3D欧氏距离 + 重叠检测)
# ============================================================
def merge_tracks(tracks):
    """航迹合并: 首尾衔接 + 重叠检测 + 全程空间重叠扫描"""
    if len(tracks) < 2:
        return tracks

    def state_to_cart(state):
        r, az, el = state[0], state[2], state[4]
        x = r * np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(az))
        y = r * np.cos(np.deg2rad(el)) * np.sin(np.deg2rad(az))
        z = r * np.sin(np.deg2rad(el))
        return np.array([x, y, z])

    def check_proximity(sa, sb, thresh):
        ca = state_to_cart(sa)
        cb = state_to_cart(sb)
        dist_3d = np.linalg.norm(ca - cb)
        if dist_3d > thresh:
            return False
        daz = min(abs(sa[2] - sb[2]), 360 - abs(sa[2] - sb[2]))
        del_ = abs(sa[4] - sb[4])
        dv = abs(sa[1] - sb[1])
        return daz < MERGE_AZ_TH and del_ < MERGE_EL_TH and dv < MERGE_VEL_TH

    merged = []
    used = set()

    for i, t1 in enumerate(tracks):
        if i in used:
            continue
        for j, t2 in enumerate(tracks):
            if j <= i or j in used:
                continue

            should_merge = False
            sm1 = getattr(t1, 'history_smoothed', None)
            if sm1 is None:
                sm1 = t1.history
            sm2 = getattr(t2, 'history_smoothed', None)
            if sm2 is None:
                sm2 = t2.history

            # 条件1: 首尾衔接
            if check_proximity(sm1[-1], sm2[0], MERGE_3D_DIST):
                should_merge = True

            # 条件2: 同时段重叠 (终点接近)
            if not should_merge:
                if check_proximity(sm1[-1], sm2[-1], MERGE_OVERLAP_3D):
                    should_merge = True

            # 条件3: 全程空间重叠扫描 (任意时刻接近即合并)
            if not should_merge:
                th1 = getattr(t1, 't_history', [0]*len(sm1))
                th2 = getattr(t2, 't_history', [0]*len(sm2))
                for k1, (s1, t1_time) in enumerate(zip(sm1, th1)):
                    if t1_time == 0:
                        continue
                    for k2, (s2, t2_time) in enumerate(zip(sm2, th2)):
                        if t2_time == 0:
                            continue
                        if abs(t1_time - t2_time) < 25000:  # 25秒内的时空重叠 (v4r5: 30→15s, r1: 15→25s)
                            if check_proximity(s1, s2, MERGE_OVERLAP_3D + 100.0):
                                should_merge = True
                                break
                    if should_merge:
                        break

            if should_merge:
                used.add(j)
                combined = list(zip(t1.history, t1.t_history,
                                    getattr(t1, '_hit_frames', [True]*len(t1.history))))
                combined += list(zip(t2.history, t2.t_history,
                                     getattr(t2, '_hit_frames', [True]*len(t2.history))))
                combined.sort(key=lambda x: x[1] if x[1] > 0 else 1e9)

                t1.history = [x[0] for x in combined]
                t1.t_history = [x[1] for x in combined]
                t1._hit_frames = [x[2] for x in combined]
                t1.hits = sum(t1._hit_frames)
                t1.age = len(combined)
                t1.quality_score = max(t1.quality_score, t2.quality_score)
                t1.history_smoothed = None

        merged.append(t1)
        used.add(i)

    return merged


# ============================================================
# 7. 主处理函数
# ============================================================
def process_task_b_v4_2(filepath, output_path, dataset_name, return_metrics=False, verbose=True):
    if verbose:
        print(f'\n{"="*60}')
        print(f'Task B v3: PDAF-KF + UCM + Enhanced Preprocessing ({dataset_name})')
        print(f'{"="*60}')

    # [1] 读取
    if verbose: print('[1/8] Reading data...')
    df = read_track_data(filepath)
    df = extract_columns(df)
    if verbose: print(f'  Total points: {len(df)}')

    # [2] 帧分组
    if verbose: print('[2/8] Frame grouping...')
    frames = group_frames_by_time(df)
    if len(frames) > 1:
        t0 = sorted(frames.keys())[:2]
        dt_vals = []
        for k in t0:
            if 'time' in frames[k].columns:
                dt_vals.append(frames[k]['time'].mean())
        nom_dt = (dt_vals[1] - dt_vals[0]) / 1000.0 if len(dt_vals) >= 2 else 3.6
    else:
        nom_dt = 3.6
    if verbose: print(f'  Frames: {len(frames)}, dt: {nom_dt:.3f}s')

    # [3] 预处理: 物理过滤 → 距离环抑制 → 凝聚(SNR/DBSCAN) → 置信度过滤
    method_label = 'DBSCAN' if CONDENSATION_METHOD == 'dbscan' else 'SNR'
    if verbose:
        print(f'[3/8] Preprocessing: physical filter → ring suppression → '
              f'{method_label} condensation → confidence...')
    all_measurements = {}
    total_raw, total_filtered, total_condensed = 0, 0, 0
    for fid in sorted(frames.keys()):
        fdf = frames[fid]
        total_raw += len(fdf)
        fdf = filter_physical(fdf)
        total_filtered += len(fdf)
        if len(fdf) == 0:
            continue

        fdf = distance_ring_suppression(fdf)
        if CONDENSATION_METHOD == 'dbscan':
            centroids = dbscan_condensation(fdf, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        else:
            centroids = snr_based_condensation(fdf)
        centroids = confidence_filter(centroids)
        total_condensed += len(centroids)
        if len(centroids) > 0:
            all_measurements[fid] = centroids

    if verbose:
        print(f'  Raw: {total_raw}, Filtered: {total_filtered}, Condensed: {total_condensed}')
        print(f'  Active frames: {len(all_measurements)}')

    # [4] 初始化
    if verbose: print('[4/8] Initializing PDAF-KF tracker...')
    tm = TrackManager(dt=nom_dt)

    # [5] 逐帧处理
    if verbose: print('[5/8] Frame-by-frame PDAF tracking...')
    for fid in sorted(all_measurements.keys()):
        tm.process_frame(all_measurements[fid])

    # [6] 提取 + RTS + 合并
    if verbose: print('[6/8] Track extraction + RTS smoothing + merging...')
    valid_tracks = tm.get_valid_tracks()

    # 收集合并前每条航迹的详细指标
    pre_merge_metrics = []
    for track in valid_tracks:
        vel_hist = getattr(track, '_velocity_history', [])
        vel_std = float(np.std(vel_hist[-min(20, len(vel_hist)):])) if len(vel_hist) >= 3 else 999.0
        pre_merge_metrics.append({
            'id': track.id,
            'age': track.age,
            'hits': track.hits,
            'hit_ratio': track.hits / max(track.age, 1),
            'misses': track.misses,
            'consecutive_misses': track.consecutive_misses,
            'quality_score': track.quality_score,
            'vel_std': vel_std,
            'range': float(track.x[0]),
            'velocity': float(track.x[1]),
            'azimuth': float(track.x[2]),
            'elevation': float(track.x[4]),
            'length_frames': len(track.history),
        })

    for track in valid_tracks:
        try:
            track.rts_smooth()
        except:
            pass

    valid_tracks_before_merge = len(valid_tracks)
    valid_tracks = merge_tracks(valid_tracks)
    # 重新RTS合并后的航迹
    for track in valid_tracks:
        if not hasattr(track, 'history_smoothed') or track.history_smoothed is None:
            try:
                track.rts_smooth()
            except:
                pass

    if verbose:
        print(f'  Valid tracks: {len(valid_tracks)}')
        for track in valid_tracks:
            r = track.x[0]
            ratio = track.hits / max(track.age, 1)
            print(f'    Track #{track.id}: {track.age} frames, hits={track.hits}, '
                  f'ratio={ratio:.2f}, r={r:.0f}m, q={track.quality_score:.2f}')

    # [7] 生成输出 (仅命中帧)
    if verbose: print('[7/8] Generating output (hit frames only)...')
    output_rows = []
    track_idx = 1

    # 收集合并后每条航迹的详细指标
    post_merge_metrics = []
    for track in valid_tracks:
        history = getattr(track, 'history_smoothed', track.history)
        hit_frames = getattr(track, '_hit_frames', [True] * len(history))
        time_hist = track.t_history if hasattr(track, 't_history') else [0] * len(history)
        n_total = len(history)
        track_has_output = False

        for k in range(n_total):
            if k < len(hit_frames) and not hit_frames[k]:
                continue
            state = history[k]
            fpga_t = int(time_hist[k]) if k < len(time_hist) else 0

            output_rows.append({
                '航迹批号': track_idx,
                'fpga时间': fpga_t,
                '距离': round(float(state[0]), 2),
                '方位': round(float(state[2]), 2),
                '俯仰': round(float(state[4]), 2),
            })
            track_has_output = True

        vel_hist = getattr(track, '_velocity_history', [])
        vel_std = float(np.std(vel_hist[-min(20, len(vel_hist)):])) if len(vel_hist) >= 3 else 999.0

        post_merge_metrics.append({
            'id': track.id,
            'age': track.age,
            'hits': track.hits,
            'hit_ratio': track.hits / max(track.age, 1),
            'misses': track.misses,
            'consecutive_misses': track.consecutive_misses,
            'quality_score': track.quality_score,
            'vel_std': vel_std,
            'range': float(track.x[0]),
            'velocity': float(track.x[1]),
            'azimuth': float(track.x[2]),
            'elevation': float(track.x[4]),
            'length_frames': n_total,
            'output_points': sum(1 for k in range(n_total)
                                 if k < len(hit_frames) and hit_frames[k]),
        })

        if track_has_output:
            track_idx += 1

    # 保底
    if len(output_rows) == 0:
        for fid in sorted(all_measurements.keys()):
            for pt in all_measurements[fid]:
                output_rows.append({
                    '航迹批号': 1,
                    'fpga时间': int(pt.get('time', 0)),
                    '距离': round(pt['range'], 2),
                    '方位': round(np.rad2deg(pt['azimuth']), 2),
                    '俯仰': round(np.rad2deg(pt['elevation']), 2),
                })

    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig', lineterminator='\n')
    n_tracks = result_df['航迹批号'].nunique() if len(result_df) > 0 else 0
    if verbose:
        print(f'\n  Final: {n_tracks} tracks, {len(result_df)} points')
        print(f'  Saved: {output_path}')
        print('[8/8] Done!')

    if return_metrics:
        metrics = {
            'n_tracks': n_tracks,
            'n_output_points': len(result_df),
            'total_raw': total_raw,
            'total_filtered': total_filtered,
            'total_condensed': total_condensed,
            'n_frames': len(frames),
            'active_frames': len(all_measurements),
            'nom_dt': nom_dt,
            'tracks_before_merge': valid_tracks_before_merge,
            'tracks_after_merge': len(valid_tracks),
            'pre_merge_metrics': pre_merge_metrics,
            'post_merge_metrics': post_merge_metrics,
        }
        return result_df, metrics
    return result_df


# ============================================================
# 8. 运行入口
# ============================================================
def run_v4_2():
    print('\n' + '='*60)
    print('Task B v4-2: PDAF-KF + UCM + DBSCAN Target Tracking')
    print('='*60)

    v42_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(v42_dir, '任务提交')
    os.makedirs(out_dir, exist_ok=True)

    out_a = os.path.join(out_dir, 'Track_A_submission.csv')
    out_b = os.path.join(out_dir, 'Track_B_submission.csv')

    result_a = process_task_b_v4_2(TRACK_A_DATA, out_a, 'A')
    result_b = process_task_b_v4_2(TRACK_B_DATA, out_b, 'B')

    print(f'\n{"="*60}')
    print(f'v4-2 输出目录: {out_dir}')
    return result_a, result_b


if __name__ == '__main__':
    run_v4_2()
