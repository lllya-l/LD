"""
v4-2 完整流水线运行脚本 (DBSCAN凝聚版)
Task A: 多目标积累检测 (使用final版本)
Task B: 目标跟踪与航迹生成 (v4-2: PDAF-KF+UCM + DBSCAN凝聚可选)
Task C: 目标分类识别 (使用final版本)

输出到 v4-2/任务提交/
"""
import time, os, sys, warnings
warnings.filterwarnings('ignore')

V42_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V42_DIR)
sys.path.insert(0, project_root)

from config import *
OUT_DIR = os.path.join(V42_DIR, '任务提交')
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

# ===== Task A =====
print('#' * 60)
print('# Task A: 多目标积累检测')
print('#' * 60)
ta_s = time.time()
from task_a_final import process_task_a
process_task_a(output_path=os.path.join(OUT_DIR, 'Target_submission.csv'))
print(f'Task A: {time.time()-ta_s:.1f}s\n')

# ===== Task B =====
print('#' * 60)
print('# Task B: 目标跟踪与航迹生成 (v4-2: PDAF-KF + UCM + DBSCAN)')
print('#' * 60)
tb_s = time.time()
from task_b_v4_2 import process_task_b_v4_2
process_task_b_v4_2(TRACK_A_DATA, os.path.join(OUT_DIR, 'Track_A_submission.csv'), 'A')
print()
process_task_b_v4_2(TRACK_B_DATA, os.path.join(OUT_DIR, 'Track_B_submission.csv'), 'B')
print(f'Task B: {time.time()-tb_s:.1f}s\n')

# ===== Task C =====
print('#' * 60)
print('# Task C: 目标分类识别')
print('#' * 60)
tc_s = time.time()
from task_c_final import run_task_c
run_task_c(output_path=os.path.join(OUT_DIR, 'predictions.csv'))
print(f'Task C: {time.time()-tc_s:.1f}s\n')

print('=' * 60)
print(f'  全部完成! 总耗时 {time.time()-t0:.1f}s')
print(f'  输出目录: {OUT_DIR}')
print('=' * 60)

# 验证输出文件
import pandas as pd
for fn in ['Target_submission.csv', 'Track_A_submission.csv',
           'Track_B_submission.csv', 'predictions.csv']:
    fp = os.path.join(OUT_DIR, fn)
    if os.path.exists(fp):
        df = pd.read_csv(fp, encoding='utf-8')
        n_tracks = df['航迹批号'].nunique() if '航迹批号' in df.columns else 0
        print(f'  [OK] {fn}: {len(df)}条, cols={list(df.columns)[:6]}')
    else:
        print(f'  [MISS] {fn}')
