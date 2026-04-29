"""
任务C 最终优化版: 基于航迹特征的目标分类识别
优化:
1. 保留基线81维精炼特征 (经59样本验证)
2. Voting集成增强: GB(400)+RF(400)+ET(250), soft voting
3. 全LOO评估 + 逐类分析
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.metrics import classification_report
from scipy import stats as scipy_stats
from config import *

warnings.filterwarnings('ignore')

CLASS_NAMES = ['Bird', 'Fixed-wing UAV', 'Helicopter',
               'Passenger ship', 'Rotary drone', 'Speedboat']


# ============================================================
# 1. 特征提取 (基线81维 + 精选增量)
# ============================================================
def extract_track_features(df):
    """从单条航迹提取特征"""
    features = {}
    n_frames = len(df)
    features['n_frames'] = n_frames

    col_map = {}
    for col in TASKC_FEATURES:
        if col in df.columns:
            col_map[col] = df[col].values.astype(np.float64)

    # 基础统计 (5特征 × 12统计 = 60)
    for col_name, vals in col_map.items():
        if len(vals) == 0:
            continue
        features[f'{col_name}_mean'] = np.mean(vals)
        features[f'{col_name}_std'] = np.std(vals)
        features[f'{col_name}_min'] = np.min(vals)
        features[f'{col_name}_max'] = np.max(vals)
        features[f'{col_name}_median'] = np.median(vals)
        features[f'{col_name}_range'] = np.max(vals) - np.min(vals)
        if len(vals) >= 2:
            x = np.arange(len(vals))
            slope, _, _, _, _ = scipy_stats.linregress(x, vals)
            features[f'{col_name}_slope'] = slope
        else:
            features[f'{col_name}_slope'] = 0.0
        features[f'{col_name}_q25'] = np.percentile(vals, 25)
        features[f'{col_name}_q75'] = np.percentile(vals, 75)
        if abs(np.mean(vals)) > 1e-10:
            features[f'{col_name}_cv'] = np.std(vals) / abs(np.mean(vals))
        else:
            features[f'{col_name}_cv'] = 0.0
        if len(vals) >= 4:
            features[f'{col_name}_kurtosis'] = scipy_stats.kurtosis(vals)
            features[f'{col_name}_skew'] = scipy_stats.skew(vals)
        else:
            features[f'{col_name}_kurtosis'] = 0.0
            features[f'{col_name}_skew'] = 0.0

    # 物理特征 (速度相关)
    if 'V_m' in col_map:
        v = col_map['V_m']
        features['V_sign'] = np.sign(np.mean(v))
        features['V_abs_mean'] = np.mean(np.abs(v))
        features['V_abs_max'] = np.max(np.abs(v))
        features['V_abs_min'] = np.min(np.abs(v))
        features['V_pos_ratio'] = np.mean(v > 0)
        features['V_neg_ratio'] = np.mean(v < 0)

    # 仰角特征
    if 'E_m' in col_map:
        e = col_map['E_m']
        features['E_abs_mean'] = np.mean(np.abs(e))
        features['E_abs_max'] = np.max(np.abs(e))

    # 交互特征
    if 'SNR' in col_map and 'V_m' in col_map:
        snr, v = col_map['SNR'], col_map['V_m']
        v_abs = np.abs(v) + 1e-10
        features['SNR_over_V'] = np.mean(snr / v_abs)

    if 'R_m' in col_map and 'V_m' in col_map:
        r, v = col_map['R_m'], col_map['V_m']
        v_abs = np.abs(v) + 1e-10
        features['R_over_V'] = np.mean(r / v_abs)

    if 'SNR' in col_map and 'R_m' in col_map:
        snr, r = col_map['SNR'], col_map['R_m']
        features['SNR_over_R'] = np.mean(snr / (r + 1e-10))

    if 'A_m' in col_map and len(col_map['A_m']) >= 2:
        az = col_map['A_m']
        az_diff = np.diff(az)
        features['Az_total_change'] = np.sum(np.abs(az_diff))
        features['Az_change_std'] = np.std(az_diff)

    if 'V_m' in col_map and 'E_m' in col_map:
        v, e = col_map['V_m'], col_map['E_m']
        features['VxE_corr'] = np.corrcoef(v, e)[0, 1] if len(v) >= 3 else 0.0

    # 距离区域特征
    if 'R_m' in col_map:
        r = col_map['R_m']
        features['R_below_2km'] = np.mean(r < 2.0)
        features['R_2to5km'] = np.mean((r >= 2.0) & (r < 5.0))
        features['R_above_5km'] = np.mean(r >= 5.0)

    # SNR阈值特征
    if 'SNR' in col_map:
        snr = col_map['SNR']
        features['SNR_above_50'] = np.mean(snr > 50)
        features['SNR_above_60'] = np.mean(snr > 60)
        features['SNR_below_40'] = np.mean(snr < 40)

    # 精选新增特征 (低噪声, 高判别力)
    if 'V_m' in col_map and len(col_map['V_m']) >= 3:
        v = col_map['V_m']
        v_acc = np.abs(np.diff(v))
        features['V_acc_mean'] = np.mean(v_acc)
        features['V_acc_std'] = np.std(v_acc)
        # 悬停比
        features['V_hover_ratio'] = np.mean(np.abs(v) < 5.0)

    if 'A_m' in col_map and 'R_m' in col_map and len(col_map['A_m']) >= 2:
        az = col_map['A_m']
        az_diff = np.diff(az)
        az_diff = (az_diff + 180) % 360 - 180
        features['Az_change_mean'] = np.mean(np.abs(az_diff))

    if 'E_m' in col_map and len(col_map['E_m']) >= 3:
        e = col_map['E_m']
        features['E_roughness'] = np.std(np.diff(e))

    if 'SNR' in col_map and len(col_map['SNR']) >= 3:
        snr = col_map['SNR']
        features['SNR_stability'] = np.std(np.diff(snr))

    return features


# ============================================================
# 2. 数据加载
# ============================================================
def load_training_data(data_dir):
    X_list, y_list = [], []
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        csv_files = [f for f in os.listdir(cls_dir) if f.endswith('.csv')]
        for fname in csv_files:
            try:
                df = pd.read_csv(os.path.join(cls_dir, fname))
                feat = extract_track_features(df)
                X_list.append(feat)
                y_list.append(cls_name)
            except Exception as e:
                print(f'  [WARN] Failed loading {fname}: {e}')
    X_df = pd.DataFrame(X_list)
    print(f'\n[Task C] Loaded {len(X_df)} training samples, {len(X_df.columns)} features')
    for cls in CLASS_NAMES:
        print(f'    {cls}: {y_list.count(cls)}')
    return X_df, y_list


def load_test_data(data_dir):
    X_list, names_list = [], []
    csv_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.csv')],
        key=lambda x: int(x.replace('test_data_', '').replace('.csv', ''))
    )
    for fname in csv_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, fname))
            feat = extract_track_features(df)
            X_list.append(feat)
            names_list.append(fname.replace('.csv', ''))
        except Exception as e:
            print(f'  [WARN] Failed loading {fname}: {e}')
    X_df = pd.DataFrame(X_list)
    print(f'\n[Task C] Loaded {len(X_df)} test samples')
    return X_df, names_list


# ============================================================
# 3. 增强Voting集成 + LOO
# ============================================================
def train_ensemble_loo(X_train, y_train):
    """
    增强Voting集成:
    - GB(400) + RF(400) + ET(250) soft voting
    - 比基线更多的estimators
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    X_train = X_train.fillna(X_train.median())
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    def build_ensemble(random_state=42):
        gb = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.07,
            subsample=0.7, max_features=0.7, min_samples_leaf=2,
            random_state=random_state,
        )
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=2,
            random_state=random_state, class_weight='balanced',
        )
        et = ExtraTreesClassifier(
            n_estimators=250, max_depth=4, min_samples_leaf=2,
            random_state=random_state, class_weight='balanced',
        )
        return VotingClassifier(
            estimators=[('gb', gb), ('rf', rf), ('et', et)],
            voting='soft',
        )

    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        loo_model = build_ensemble(random_state=42)
        loo_model.fit(X_tr, y_tr)
        y_pred_all.append(loo_model.predict(X_te)[0])
        y_true_all.append(y_te[0])

    acc = np.mean(np.array(y_true_all) == np.array(y_pred_all))
    print(f'\n[Task C] LOO CV accuracy: {acc*100:.1f}% ({np.sum(np.array(y_true_all) == np.array(y_pred_all))}/{len(y_true_all)})')

    print('\n[Task C] LOO Classification Report:')
    print(classification_report(y_true_all, y_pred_all,
                                target_names=le.classes_, labels=range(len(le.classes_))))

    print('\n[Task C] Per-class LOO accuracy:')
    for i, cls in enumerate(le.classes_):
        mask = np.array(y_true_all) == i
        if mask.sum() > 0:
            ca = (np.array(y_true_all)[mask] == np.array(y_pred_all)[mask]).mean()
            correct = (np.array(y_true_all)[mask] == np.array(y_pred_all)[mask]).sum()
            print(f'  {cls}: {ca*100:.1f}% ({correct}/{mask.sum()})')

    print('\n[Task C] Confusion Matrix:')
    cm = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)
    for t, p in zip(y_true_all, y_pred_all):
        cm[t, p] += 1
    header = '        ' + '  '.join(f'{c[:6]:>6s}' for c in le.classes_)
    print(header)
    for i, cls in enumerate(le.classes_):
        row = f'{cls[:12]:>12s} ' + ' '.join(f'{cm[i, j]:>6d}' for j in range(len(le.classes_)))
        print(row)

    final_model = build_ensemble(random_state=42)
    final_model.fit(X_scaled, y_enc)

    # 记录训练集特征中位数, 用于测试时缺失列补缺
    train_medians = X_train.median()

    class ModelWrapper:
        def __init__(self, model, scaler, le, train_medians):
            self.model = model
            self.scaler = scaler
            self.le = le
            self.train_medians = train_medians

        def predict(self, X):
            X = X.fillna(X.median())
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_cols = list(self.scaler.feature_names_in_)
                for c in feature_cols:
                    if c not in X.columns:
                        # 用训练集中位数填充缺失特征, 经RobustScaler变换后为中性值
                        fallback = self.train_medians.get(c, 0.0) if hasattr(self, 'train_medians') else 0.0
                        X[c] = fallback
                X = X[feature_cols]
            X_scaled = self.scaler.transform(X)
            return self.le.inverse_transform(self.model.predict(X_scaled))

    return ModelWrapper(final_model, scaler, le, train_medians)


# ============================================================
# 4. 主流程
# ============================================================
def run_task_c(output_path=None):
    if output_path is None:
        output_path = TASKC_SUBMISSION

    train_dir = TASKC_TRAIN_DIR
    test_dir = TASKC_TEST_DIR

    if not os.path.isdir(train_dir):
        print(f'[ERROR] Train dir not found: {train_dir}')
        print('请将赛题训练数据放入 data/train/ 目录')
        return None
    if not os.path.isdir(test_dir):
        print(f'[ERROR] Test dir not found: {test_dir}')
        print('请将赛题测试数据放入 data/test/ 目录')
        return None

    print('=' * 60)
    print('  Task C Optimized: Enhanced Voting (GB400+RF400+ET250)')
    print('  Features: 81 base + 8 selected new = ~89 features')
    print('=' * 60)

    print('[1/4] Loading training data...')
    X_train, y_train = load_training_data(train_dir)
    if len(X_train) == 0:
        print('[ERROR] No training data!')
        return None

    print('[2/4] Enhanced Voting LOO training...')
    wrapper = train_ensemble_loo(X_train, y_train)

    print('[3/4] Loading test data...')
    X_test, test_names = load_test_data(test_dir)
    if len(X_test) == 0:
        print('[ERROR] No test data!')
        return None

    print('[4/4] Predicting & saving...')
    y_pred = wrapper.predict(X_test)

    result_df = pd.DataFrame({'name': test_names, 'label': y_pred})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
    print(f'\n[Task C] Saved: {output_path}')
    print(result_df.to_string(index=False))

    return result_df


if __name__ == '__main__':
    run_task_c()
