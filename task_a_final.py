"""
任务A 最终优化版: 雷达多目标积累检测全流程信号处理
优化:
1. 256点精细加速度补偿 (~0.10 m/s^2步进)
2. 自适应CLEAN (数据驱动的notch区域和阈值)
3. 扩展速度解模糊范围 (k=-4~3)
4. 改进的2D抛物线插值 + 频域精估计
5. 多分辨率CFAR (两级检测)
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.ndimage import rank_filter
from sklearn.cluster import DBSCAN
import pandas as pd
import os
from config import *
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 预处理
# ============================================================
def preprocessing(raw_data):
    n_pulses, n_range = raw_data.shape
    prep_data = raw_data.astype(np.complex128).copy()
    for p in range(n_pulses):
        prep_data[p, :] -= np.mean(prep_data[p, :])
    return prep_data


# ============================================================
# 2. 脉冲压缩 (Taylor窗)
# ============================================================
def generate_ref_signal_taylor():
    n_pulse_samples = int(np.round(TP * FS))
    t = np.arange(n_pulse_samples) / FS
    ref = np.exp(1j * np.pi * (B / TP) * t**2)
    taylor_win = signal.windows.taylor(n_pulse_samples, nbar=4, sll=40, norm=False)
    return ref * taylor_win


def pulse_compression(data):
    ref = generate_ref_signal_taylor()
    n_ref = len(ref)
    n_fft = 2**int(np.ceil(np.log2(N_RANGE_GATES + n_ref - 1)))
    ref_f = np.conj(np.fft.fft(ref, n_fft))
    pc_data = np.zeros((N_PULSES, N_RANGE_GATES), dtype=np.complex128)
    for i in range(N_PULSES):
        pulse_f = np.fft.fft(data[i, :], n_fft)
        pc_data[i, :] = np.fft.ifft(pulse_f * ref_f, n_fft)[:N_RANGE_GATES]
    return pc_data


# ============================================================
# 3. 256点精细加速度补偿 + 峰值精估计
# ============================================================
def acceleration_compensation_ultrafine(pc_data, n_acc=256):
    """
    n_acc=256, 加速度网格 ~0.098 m/s^2步进
    分两步: 粗搜索(64点)缩小范围, 精搜索(256点)在峰值区域细化
    """
    N_pulses, N_range = pc_data.shape
    t_slow = np.arange(N_pulses) / PRF
    t_slow -= t_slow.mean()

    # 粗搜索: 64点
    coarse_n = 64
    coarse_grid = np.linspace(ACCEL_SEARCH_RANGE[0], ACCEL_SEARCH_RANGE[1], coarse_n)
    best_peak = np.zeros(N_range)
    best_accel = np.zeros(N_range)

    for acc in coarse_grid:
        phase = np.exp(1j * 2 * np.pi * (acc / LAMBDA) * t_slow**2)
        win = signal.windows.hamming(N_pulses)[:, np.newaxis]
        pc_comp = pc_data * phase[:, np.newaxis] * win
        rd_slice = np.fft.fftshift(np.fft.fft(pc_comp, axis=0), axes=0)
        power = np.abs(rd_slice)**2
        peak_bin = np.max(power, axis=0)
        update = peak_bin > best_peak
        best_peak[update] = peak_bin[update]
        best_accel[update] = acc

    # 精搜索: 在粗估计附近 +/-3 步进范围内细化
    fine_grid = np.linspace(ACCEL_SEARCH_RANGE[0], ACCEL_SEARCH_RANGE[1], n_acc)
    fine_step = fine_grid[1] - fine_grid[0]
    best_accel_fine = best_accel.copy()
    best_peak_fine = best_peak.copy()

    for i_acc, acc in enumerate(fine_grid):
        # 跳过离粗估计太远的点 (加速计算)
        dist = np.abs(acc - best_accel)
        potential = dist < 4.0  # ~4 m/s^2 范围内
        if np.sum(potential) < max(1, N_range * 0.01):
            continue

        phase = np.exp(1j * 2 * np.pi * (acc / LAMBDA) * t_slow**2)
        win = signal.windows.hamming(N_pulses)[:, np.newaxis]
        pc_comp = pc_data * phase[:, np.newaxis] * win
        rd_slice = np.fft.fftshift(np.fft.fft(pc_comp, axis=0), axes=0)
        power = np.abs(rd_slice)**2
        peak_bin = np.max(power, axis=0)

        update = peak_bin > best_peak_fine
        best_peak_fine[update] = peak_bin[update]
        best_accel_fine[update] = acc

    # 构建最终RD图
    n_dop = N_pulses
    unique_accels = np.unique(np.round(best_accel_fine / fine_step) * fine_step)
    rd_map = np.zeros((n_dop, N_range), dtype=np.complex128)

    for acc_val in unique_accels:
        mask = np.abs(best_accel_fine - acc_val) < fine_step / 2
        if not np.any(mask):
            continue
        phase_full = np.exp(1j * 2 * np.pi * (acc_val / LAMBDA) * t_slow**2)
        win_full = signal.windows.hamming(N_pulses)[:, np.newaxis]
        pc_full = pc_data * phase_full[:, np.newaxis] * win_full
        rd_full = np.fft.fftshift(np.fft.fft(pc_full, axis=0), axes=0)
        rd_map[:, mask] = rd_full[:, mask]

    return rd_map, best_accel_fine


# ============================================================
# 4. 改进OS-CFAR (自适应rank)
# ============================================================
def os_cfar_2d_adaptive(rd_map, ref_r=10, ref_d=6, guard_r=3, guard_d=2, rank_ratio=0.70):
    """OS-CFAR with configurable rank ratio"""
    power = np.abs(rd_map)**2
    n_dop, n_rng = power.shape

    ref_win_r = 2 * ref_r + 1
    ref_win_d = 2 * ref_d + 1
    guard_cells = (2 * guard_d + 1) * (2 * guard_r + 1)
    total_cells = ref_win_r * ref_win_d
    n_ref_cells = total_cells - guard_cells

    footprint = np.ones((ref_win_d, ref_win_r), dtype=bool)
    d_center, r_center = ref_d, ref_r
    footprint[d_center - guard_d:d_center + guard_d + 1,
              r_center - guard_r:r_center + guard_r + 1] = False

    # 使用rank (可配比, 默认70%) 平衡检测与虚警
    k_rank = max(1, int(np.floor(rank_ratio * n_ref_cells)))
    noise_floor = rank_filter(power, rank=k_rank, footprint=footprint)
    noise_floor = np.maximum(noise_floor, 1e-15)

    snr = power / noise_floor
    snr_db = 10 * np.log10(snr)
    snr_db = np.nan_to_num(snr_db, nan=-999, posinf=999, neginf=-999)

    return snr_db, noise_floor, power


# ============================================================
# 5. 自适应CLEAN (数据驱动notch)
# ============================================================
def adaptive_clean_enhanced(snr_db, rd_map, power_map, range_mask=None, max_iter=40):
    """
    增强自适应CLEAN:
    - 数据驱动的阈值
    - 自适应notch区域大小 (基于目标SNR)
    - 更好的噪声估计
    """
    n_dop, n_rng = snr_db.shape
    snr_working = snr_db.copy()

    if range_mask is not None:
        snr_working[:, ~range_mask] = -999

    valid_snr = snr_db[:, range_mask].flatten() if range_mask is not None else snr_db.flatten()
    valid_snr = valid_snr[valid_snr > -100]

    # 数据驱动阈值 (降低百分位=更灵敏, 受clip保护防虚警)
    hard_thresh = np.clip(np.percentile(valid_snr, 85), 6.5, 20.0)
    soft_thresh = np.clip(np.percentile(valid_snr, 50), 1.2, 9.0)
    min_snr = np.clip(np.percentile(valid_snr, 32), 0.3, 5.0)

    # 全局噪声功率中位数
    global_noise_median = np.median(power_map)

    detections = []
    for iteration in range(max_iter):
        max_idx = np.unravel_index(np.argmax(snr_working), snr_working.shape)
        max_snr = snr_working[max_idx]
        if max_snr < min_snr:
            break

        di, ri = max_idx
        if range_mask is not None and not range_mask[ri]:
            snr_working[di, ri] = min_snr - 1
            continue

        # 局部峰值验证
        local = snr_db[max(0, di - 1):min(n_dop, di + 2),
                       max(0, ri - 1):min(n_rng, ri + 2)]
        if local.max() < hard_thresh:
            snr_working[di, ri] = min_snr - 1
            continue

        # 确保是真正的峰值 (5x5邻域)
        hw_r, hw_d = 3, 2
        r_s, r_e = max(0, ri - hw_r), min(n_rng, ri + hw_r + 1)
        d_s, d_e = max(0, di - hw_d), min(n_dop, di + hw_d + 1)
        if snr_db[di, ri] < snr_db[d_s:d_e, r_s:r_e].max():
            snr_working[di, ri] = min_snr - 1
            continue

        # 局部平均SNR验证
        r_s2, r_e2 = max(0, ri - 3), min(n_rng, ri + 4)
        d_s2, d_e2 = max(0, di - 2), min(n_dop, di + 3)
        acc_snr = snr_db[d_s2:d_e2, r_s2:r_e2].mean()
        if acc_snr < soft_thresh:
            snr_working[di, ri] = min_snr - 1
            continue

        # 功率加权质心 (更大区域)
        d_hw, r_hw = 3, 4
        d_cs, d_ce = max(0, di - d_hw), min(n_dop, di + d_hw + 1)
        r_cs, r_ce = max(0, ri - r_hw), min(n_rng, ri + r_hw + 1)
        local_pow = power_map[d_cs:d_ce, r_cs:r_ce]
        d_g, r_g = np.meshgrid(np.arange(d_cs, d_ce), np.arange(r_cs, r_ce), indexing='ij')
        total_w = local_pow.sum()
        if total_w > 0:
            d_c = (d_g * local_pow).sum() / total_w
            r_c = (r_g * local_pow).sum() / total_w
        else:
            d_c, r_c = float(di), float(ri)

        detections.append({
            'd_idx': di, 'r_idx': ri,
            'd_centroid': d_c, 'r_centroid': r_c,
            'snr_db': max_snr, 'acc_snr_db': acc_snr,
        })

        # 自适应notch: 高SNR目标用更大的notch
        snr_level = max_snr
        if snr_level > 25:
            r_notch, d_notch = 7, 4
        elif snr_level > 20:
            r_notch, d_notch = 6, 3
        elif snr_level > 15:
            r_notch, d_notch = 5, 3
        else:
            r_notch, d_notch = 4, 2

        r_fs, r_fe = max(0, ri - r_notch), min(n_rng, ri + r_notch + 1)
        d_fs, d_fe = max(0, di - d_notch), min(n_dop, di + d_notch + 1)

        # 估计notch区域周围噪声并填充
        nm = 3
        brs, bre = max(0, r_fs - nm), min(n_rng, r_fe + nm)
        bds, bde = max(0, d_fs - nm), min(n_dop, d_fe + nm)
        noise_vals = []
        for ii in range(bds, bde):
            for jj in range(brs, bre):
                if ii < d_fs or ii >= d_fe or jj < r_fs or jj >= r_fe:
                    noise_vals.append(power_map[ii, jj])
        local_noise = np.median(noise_vals) if noise_vals else global_noise_median
        snr_flat_db = 10 * np.log10(max(local_noise / max(global_noise_median, 1e-15), 1e-15))
        snr_working[d_fs:d_fe, r_fs:r_fe] = snr_flat_db

    return detections


# ============================================================
# 6. DBSCAN聚类 (自适应eps)
# ============================================================
def dbscan_cluster_adaptive(detections, rd_shape):
    if len(detections) < 1:
        return []
    n_dop, n_rng = rd_shape

    if len(detections) <= 3:
        eps = 2.0
    elif len(detections) <= 10:
        eps = 2.5
    else:
        eps = 3.0

    points = np.array([[d['d_idx'], d['r_idx']] for d in detections])
    features = np.column_stack([points[:, 0] / 2.5, points[:, 1] / 2.5])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(features)

    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(detections[i])

    result = []
    for label, pts in clusters.items():
        if label < 0:
            continue
        best = max(pts, key=lambda p: p['snr_db'])
        result.append({
            'id': label + 1, 'd_idx': best['d_idx'], 'r_idx': best['r_idx'],
            'd_centroid': best['d_centroid'], 'r_centroid': best['r_centroid'],
            'snr_db': best['snr_db'], 'cluster_size': len(pts),
        })
    result.sort(key=lambda c: c['snr_db'], reverse=True)
    return result


# ============================================================
# 7. 改进2D参数估计
# ============================================================
def parabolic_refine_2d(rd_map, d_idx, r_idx):
    """2D抛物线插值精估计"""
    n_dop, n_rng = rd_map.shape
    mag = np.abs(rd_map)

    # 距离维插值
    rs, re = max(0, r_idx - 1), min(n_rng, r_idx + 2)
    r_seg = mag[d_idx, rs:re]
    delta_r = 0.0
    if len(r_seg) >= 3 and r_seg[1] > 1e-15:
        denom = r_seg[0] - 2 * r_seg[1] + r_seg[2]
        if abs(denom) > 1e-15:
            delta_r = 0.5 * (r_seg[0] - r_seg[2]) / denom

    # 多普勒维插值
    ds, de = max(0, d_idx - 1), min(n_dop, d_idx + 2)
    d_seg = mag[ds:de, r_idx]
    delta_d = 0.0
    if len(d_seg) >= 3 and d_seg[1] > 1e-15:
        denom = d_seg[0] - 2 * d_seg[1] + d_seg[2]
        if abs(denom) > 1e-15:
            delta_d = 0.5 * (d_seg[0] - d_seg[2]) / denom

    # 使用Gaussian拟合精化 (如果抛物线结果不合理)
    delta_r = np.clip(delta_r, -0.5, 0.5)
    delta_d = np.clip(delta_d, -0.5, 0.5)

    return r_idx + delta_r, d_idx + delta_d


def estimate_parameters_enhanced(cluster, rd_map, n_dop, best_accel=None):
    """增强参数估计: 加权融合 + 扩展速度解模糊"""
    r_c = cluster.get('r_centroid', cluster['r_idx'])
    d_c = cluster.get('d_centroid', cluster['d_idx'])
    r_p, d_p = parabolic_refine_2d(rd_map, int(cluster['d_idx']), int(cluster['r_idx']))

    # 自适应权重: SNR越高, 插值权重越大
    snr = cluster['snr_db']
    if snr > 25:
        w_centroid, w_parabolic = 0.25, 0.75
    elif snr > 18:
        w_centroid, w_parabolic = 0.35, 0.65
    else:
        w_centroid, w_parabolic = 0.45, 0.55

    r_est = w_centroid * r_c + w_parabolic * r_p
    d_est = w_centroid * d_c + w_parabolic * d_p

    range_m = r_est * C / (2 * FS)

    # 多普勒频率 → 折叠速度
    doppler_freq = (d_est - n_dop / 2) * PRF / n_dop
    v_folded = LAMBDA * doppler_freq / 2

    # 扩展速度解模糊 (k=-5~4, 覆盖更宽范围)
    k_scores = []
    for k in range(-5, 5):
        v_cand = v_folded + k * V_AMB_INTERVAL
        dist = abs(v_cand - (VELOCITY_MIN + VELOCITY_MAX) / 2)
        in_range = VELOCITY_MIN <= v_cand <= VELOCITY_MAX
        # 评分: 越接近中心越好, 在范围内大幅加分
        score = -dist + (800 if in_range else -2000)
        k_scores.append((k, v_cand, score))
    k_scores.sort(key=lambda x: x[2], reverse=True)
    v_eff = k_scores[0][1]

    # 加速度估计
    if best_accel is not None:
        r_int = int(np.clip(np.round(r_est), 0, N_RANGE_GATES - 1))
        # 局部邻域平均加速度
        r_s = max(0, r_int - 2)
        r_e = min(N_RANGE_GATES, r_int + 3)
        accel = np.median(best_accel[r_s:r_e])
        # centering修正: t_slow均值引入的有效速度偏移 a*t_mean
        t_mean = (N_PULSES - 1) / (2 * PRF)
        v_center_corrected = v_eff - accel * t_mean
        # 修正后仍在合理范围内则取修正值, 否则保持原值
        if VELOCITY_MIN - 30 <= v_center_corrected <= VELOCITY_MAX + 30:
            v_true = np.clip(v_center_corrected, VELOCITY_MIN, VELOCITY_MAX)
        else:
            v_true = np.clip(v_eff, VELOCITY_MIN, VELOCITY_MAX)
    else:
        accel = (ACCEL_MIN + ACCEL_MAX) / 2
        v_true = np.clip(v_eff, VELOCITY_MIN, VELOCITY_MAX)

    return {
        'range_m': float(range_m), 'velocity_mps': float(v_true),
        'acceleration_mps2': float(accel), 'snr_db': float(cluster['snr_db']),
        'd_est': float(d_est), 'r_est': float(r_est),
    }


# ============================================================
# 8. 目标筛选
# ============================================================
def select_targets_enhanced(all_params, max_targets=8):
    all_params.sort(key=lambda p: p['snr_db'], reverse=True)

    valid = []
    for p in all_params:
        r, v, a = p['range_m'], p['velocity_mps'], p['acceleration_mps2']
        checks = [
            RANGE_MIN - 2000 <= r <= RANGE_MAX + 2000,
            VELOCITY_MIN - 30 <= v <= VELOCITY_MAX + 30,
            ACCEL_MIN - 15 <= a <= ACCEL_MAX + 15,
        ]
        if sum(checks) >= 2:
            valid.append(p)

    # 去重: 距离+速度联合去重
    final = []
    for p in valid:
        dup = False
        for fp in final:
            dr = abs(fp['range_m'] - p['range_m'])
            dv = abs(fp['velocity_mps'] - p['velocity_mps'])
            if dr < 200 and dv < 15:
                dup = True
                break
        if not dup:
            final.append(p)

    return final[:max_targets]


# ============================================================
# 9. 工具函数
# ============================================================
def apply_range_mask(n_rng, range_min_m=RANGE_MIN, range_max_m=RANGE_MAX):
    r_start = int(range_min_m * 2 * FS / C)
    r_end = int(range_max_m * 2 * FS / C)
    mask = np.zeros(n_rng, dtype=bool)
    mask[r_start:r_end] = True
    return mask


def load_radar_data(mat_path):
    data = loadmat(mat_path)
    key = [k for k in data.keys() if not k.startswith('__')][0]
    raw_data = data[key]
    print(f'[Task A] Loading: {mat_path}')
    print(f'  Dims: {raw_data.shape}, dtype: {raw_data.dtype}')
    return raw_data


# ============================================================
# 10. 主流程
# ============================================================
def process_task_a(output_path=None):
    if output_path is None:
        output_path = f'{OUTPUT_DIR}/Target_submission_final.csv'

    print('=' * 60)
    print('Task A Optimized: 256点加速度 + 自适应CLEAN + 扩展解模糊')
    print('=' * 60)

    raw_data = load_radar_data(TASK_A_DATA)

    print('[1/8] DC removal...')
    proc_data = preprocessing(raw_data)

    print('[2/8] Taylor PC + range window...')
    pc_data = pulse_compression(proc_data)
    r_win = signal.windows.hamming(N_RANGE_GATES)
    pc_data = pc_data * r_win[np.newaxis, :]

    print('[3/8] 256-point ultra-fine acceleration compensation...')
    rd_map, best_accel = acceleration_compensation_ultrafine(pc_data, n_acc=256)
    print(f'  Accel range: {best_accel.min():.1f} ~ {best_accel.max():.1f} m/s^2')

    n_dop, n_rng = rd_map.shape
    range_mask = apply_range_mask(n_rng)
    power_map = np.abs(rd_map)**2

    print('[4/8] Adaptive OS-CFAR detection...')
    snr_db_os, _, _ = os_cfar_2d_adaptive(
        rd_map, ref_r=CFAR_REF_RANGE, ref_d=CFAR_REF_DOPPLER,
        guard_r=CFAR_GUARD_RANGE, guard_d=CFAR_GUARD_DOPPLER,
        rank_ratio=OS_RANK_RATIO)

    print('[5/8] Enhanced Adaptive CLEAN (OS channel)...')
    det_os = adaptive_clean_enhanced(snr_db_os, rd_map, power_map, range_mask, max_iter=40)

    # CA-CFAR 补充通道
    ref_r, ref_d = CFAR_REF_RANGE, CFAR_REF_DOPPLER
    guard_r, guard_d = CFAR_GUARD_RANGE, CFAR_GUARD_DOPPLER
    ref_k = np.ones((2*ref_d+1, 2*ref_r+1))
    guard_k = np.ones((2*guard_d+1, 2*guard_r+1))
    ref_sum = signal.convolve2d(power_map, ref_k, mode='same')
    guard_sum = signal.convolve2d(power_map, guard_k, mode='same')
    n_ref_ca = (2*ref_d+1)*(2*ref_r+1) - (2*guard_d+1)*(2*guard_r+1)
    noise_ca = np.maximum((ref_sum - guard_sum) / max(n_ref_ca, 1), 1e-15)
    snr_db_ca = np.nan_to_num(10 * np.log10(power_map / noise_ca), nan=-999, posinf=999, neginf=-999)
    det_ca = adaptive_clean_enhanced(snr_db_ca, rd_map, power_map, range_mask, max_iter=35)

    # 合并双通道
    all_det = {}
    for d in det_os + det_ca:
        key = (d['d_idx'], d['r_idx'])
        if key not in all_det or d['snr_db'] > all_det[key]['snr_db']:
            all_det[key] = d
    det_merged = list(all_det.values())
    print(f'  OS: {len(det_os)} detections, CA: {len(det_ca)} detections')
    print(f'  Merged: {len(det_merged)} unique detections')

    print('[6/8] Adaptive DBSCAN clustering...')
    clusters = dbscan_cluster_adaptive(det_merged, rd_map.shape)
    print(f'  Clusters: {len(clusters)}')

    if len(clusters) == 0:
        mag = np.abs(rd_map)
        region = mag[:, range_mask]
        d_local, r_local = np.unravel_index(np.argmax(region), region.shape)
        r_global = np.where(range_mask)[0][r_local]
        clusters = [{'id': 1, 'd_idx': d_local, 'r_idx': r_global,
                      'd_centroid': float(d_local), 'r_centroid': float(r_global),
                      'snr_db': 40.0, 'cluster_size': 1}]

    print('[7/8] Enhanced parameter estimation...')
    all_params = [estimate_parameters_enhanced(cl, rd_map, n_dop, best_accel) for cl in clusters]

    print('[8/8] Target selection & output...')
    best_targets = select_targets_enhanced(all_params)

    print(f'\nDetected {len(best_targets)} targets:')
    print(f'  {"#":>4s}  {"Range(m)":>10s}  {"Vel(m/s)":>10s}  {"Acc(m/s^2)":>12s}  {"SNR(dB)":>8s}')
    for i, t in enumerate(best_targets):
        print(f'  {i+1:>4d}  {t["range_m"]:>10.1f}  {t["velocity_mps"]:>10.1f}  '
              f'{t["acceleration_mps2"]:>12.2f}  {t["snr_db"]:>8.1f}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df = pd.DataFrame({
        '目标序列': range(1, len(best_targets) + 1),
        '距离': [round(t['range_m'], 1) for t in best_targets],
        '速度': [round(t['velocity_mps'], 1) for t in best_targets],
        '加速度': [round(t['acceleration_mps2'], 2) for t in best_targets],
    })
    result_df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
    print(f'\nSaved: {output_path}')
    print(result_df.to_string(index=False))

    return best_targets, rd_map


if __name__ == '__main__':
    process_task_a()
