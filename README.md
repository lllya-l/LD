# 雷达微弱多目标智能探测、跟踪与识别系统

> **睿创杯赛题四 — 雷达信号处理**
> 面向真实雷达场景的微弱多目标检测、航迹跟踪与目标分类一体化解决方案

---

## 目录

- [项目概述](#项目概述)
- [任务说明](#任务说明)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [文件清单](#文件清单)
- [数据准备](#数据准备)
- [运行方式](#运行方式)
- [输出说明](#输出说明)
- [技术架构](#技术架构)
- [版本历史](#版本历史)

---

## 项目概述

本项目针对雷达目标检测跟踪中的三大核心难题，构建了一套完整的信号处理 + 机器学习流水线：

| 任务 | 解决的核心问题 | 核心技术 |
|------|---------------|---------|
| **Task A** 多目标积累检测 | 高加速度（-85~-100 m/s²）下微弱目标能量扩散，常规FFT积累失效 | 256点超细加速度补偿 + OS/CA-CFAR双通道 + 自适应CLEAN |
| **Task B** 航迹跟踪与生成 | 密集杂波中低SNR点迹的航迹自动提取，虚假起批与漏检的矛盾 | PDAF-KF + DBSCAN马氏距离凝聚 + UCM无偏转换 + M/N起批 |
| **Task C** 目标分类识别 | 仅有5维点迹序列情况下区分6类目标（鸟、无人机、直升机、客船等） | 89维精炼特征 + GBDT/RF/ET三模型Soft Voting集成 |

---

## 任务说明

### Task A — 多目标积累检测

从原始雷达 I/Q 复数数据（300 脉冲 × 8000 距离门）中检测目标，输出目标的距离、速度、加速度。

- **输入**：`a)_Target_dataset.mat`（复数矩阵 300×8000）
- **输出**：`Target_submission.csv`（目标序列, 距离, 速度, 加速度）
- **信号流程**：直流去除 → 脉冲压缩(Taylor窗) → 256点超细加速度补偿 → OS-CFAR + CA-CFAR双通道 → 自适应CLEAN → DBSCAN聚类 → 2D抛物线精插值 → 扩展速度解模糊 → 目标筛选

### Task B — 航迹跟踪与生成

从点迹级雷达数据中自动提取目标航迹，处理密集杂波环境下的航迹关联问题。

- **输入**：`b)_Track_A_dataset.csv`、`b)_Track_B_dataset.csv`
- **输出**：`Track_A_submission.csv`、`Track_B_submission.csv`
- **信号流程**：帧分组 → 物理过滤 → 距离环抑制 → DBSCAN/SNR凝聚 → 置信度判决 → PDAF-KF跟踪 → M/N起批 → RTS后向平滑 → 航迹合并

### Task C — 目标分类识别

从雷达点迹序列中提取特征，对目标进行6分类识别。

- **训练数据**：`data/train/`（按6个类别分目录）
- **测试数据**：`data/test/`（18个测试样本CSV）
- **输出**：`predictions.csv`（name, label）
- **分类流程**：89维特征提取 → RobustScaler归一化 → GBDT(400)+RF(400)+ET(250) Soft Voting → LOO交叉验证

---

## 环境要求

- **Python**: 3.8+
- **操作系统**: Windows / Linux / macOS
- **依赖库**:

| 库 | 版本要求 | 用途 |
|---|---------|------|
| numpy | ≥1.21 | 数值计算 / 信号处理 |
| pandas | ≥1.3 | 数据读取 / CSV输出 |
| scipy | ≥1.7 | 信号处理 / 统计 / 线性代数 |
| scikit-learn | ≥1.0 | DBSCAN聚类 / 分类器 / 数据预处理 |

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/radar-detection-tracking.git
cd radar-detection-tracking

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据（见下方数据准备章节）

# 4. 一键运行全部任务
python run_v4_2.py
```

---

## 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `run_v4_2.py` | ~2K | 流水线入口：顺序调用A/B/C三个任务 |
| `task_a_final.py` | ~20K | Task A：多目标积累检测完整信号处理 |
| `task_b_v4_2.py` | ~43K | Task B：PDAF-KF + DBSCAN航迹跟踪（核心） |
| `task_c_final.py` | ~13K | Task C：特征提取 + 集成分类器 |
| `config.py` | ~3K | 全局配置：雷达参数 / CFAR参数 / 跟踪参数 / 路径 |
| `requirements.txt` | ~0.1K | Python依赖清单 |
| `README.md` | 本文件 | 项目说明文档 |
| `data/` | — | 数据目录（需手动放置，见下方） |

---

## 数据准备

由于赛题数据版权限制，数据文件**不包含在仓库中**。请按以下结构手动放置：

```
data/
├── a)_Target_dataset.mat      # Task A：雷达原始I/Q数据
├── b)_Track_A_dataset.csv     # Task B：航迹数据集A（点迹级）
├── b)_Track_B_dataset.csv     # Task B：航迹数据集B（点迹级）
├── train/                     # Task C：训练数据（按类别分目录）
│   ├── Bird/
│   ├── Fixed-wing UAV/
│   ├── Helicopter/
│   ├── Passenger ship/
│   ├── Rotary drone/
│   └── Speedboat/
└── test/                      # Task C：测试数据（18个CSV文件）
    ├── test_data_1.csv
    ├── test_data_2.csv
    └── ...
```

> `config.py` 中的数据路径默认指向 `./data/` 目录，如数据放置位置不同，请修改 `config.py` 中的路径配置。

---

## 运行方式

### 一键运行（推荐）

```bash
python run_v4_2.py
```

依次执行 Task A → Task B → Task C，自动生成所有提交文件。

### 单独运行各任务

```bash
# 仅运行 Task A
python -c "from task_a_final import process_task_a; process_task_a()"

# 仅运行 Task B
python -c "from task_b_v4_2 import run_v4_2; run_v4_2()"

# 仅运行 Task C
python -c "from task_c_final import run_task_c; run_task_c()"
```

### 手动指定输出路径

```python
from task_b_v4_2 import process_task_b_v4_2
process_task_b_v4_2(
    filepath='data/b)_Track_A_dataset.csv',
    output_path='output/Track_A_submission.csv',
    dataset_name='A'
)
```

---

## 输出说明

运行后 `output/` 目录下生成4个文件：

| 文件 | 对应任务 | 格式 | 说明 |
|------|---------|------|------|
| `Target_submission.csv` | Task A | `目标序列,距离,速度,加速度` | 检测到的目标参数 |
| `Track_A_submission.csv` | Task B | `航迹批号,fpga时间,距离,方位,俯仰` | 数据集A的航迹点 |
| `Track_B_submission.csv` | Task B | `航迹批号,fpga时间,距离,方位,俯仰` | 数据集B的航迹点 |
| `predictions.csv` | Task C | `name,label` | 18个测试样本的分类结果 |

---

## 技术架构

### 系统总览

```
             ┌──────────────────────────────────────────────┐
             │                 数据层                         │
             │  Task A: I/Q复数据(300×8000)                  │
             │  Task B: 点迹级CSV(时间/距离/方位/俯仰/速度)    │
             │  Task C: 航景点迹CSV(A_m/E_m/R_m/SNR/V_m)     │
             └──────────┬────────────────┬──────────────────┘
                        │                │
        ┌───────────────┘      ┌─────────┘
        ▼                      ▼
┌──────────────────┐  ┌──────────────────────┐
│  Task A          │  │  Task B              │
│  脉冲压缩         │  │  帧分组+物理过滤      │
│  256点加速度补偿  │  │  距离环抑制           │
│  OS/CA-CFAR双通道 │  │  DBSCAN/SNR凝聚      │
│  自适应CLEAN      │  │  PDAF-KF跟踪         │
│  DBSCAN聚类       │  │  M/N起批确认         │
│  参数精估计       │  │  RTS平滑+航迹合并     │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         ▼                       ▼
    Target_submission.csv    Track_*_submission.csv
                                      │
                                      ▼
                              ┌──────────────────┐
                              │  Task C          │
                              │  89维特征提取     │
                              │  RobustScaler    │
                              │  Voting集成分类   │
                              │  (GB+RF+ET)      │
                              └───────┬──────────┘
                                      ▼
                              predictions.csv
```

### Task A 核心创新

- **256点超细加速度补偿网格**：分粗搜索（64点）→精搜索（256点）两步，步进 ~0.098 m/s²
- **双通道CFAR检测**：OS-CFAR（自适应rank=70%）+ CA-CFAR 互补，提高检测概率
- **自适应CLEAN**：数据驱动阈值（SNR百分位）+ 自适应Notch尺寸（SNR越高Notch越大）
- **扩展速度解模糊**：k = -5~4 覆盖完整目标速度区间，加速度修正去除时间中心偏移

### Task B 核心创新

- **DBSCAN马氏距离凝聚**：特征空间 `[r/σ_r, cross_az/σ_cross, cross_el/σ_ph, v/σ_v]`，σ_cross = r·sin(0.25°) 自适应缩放
- **PDAF-KF 6D跟踪**：状态量 `[r, vr, az, v_az, el, v_el]`，概率数据关联处理波门内多量测
- **M/N远场起批**：远场2/6、近场3/4的差异化起批逻辑，兼顾远场低检测率和近场高虚警
- **RTS后向平滑 + 三维合并**：平滑后航迹更平滑，合并条件含首尾衔接/同时段重叠/全程时空重叠

### Task C 核心创新

- **89维精炼特征体系**：5维度（A/E/R/SNR/V）×12统计量 + 物理特征（悬停比、变化率、相关性）+ 交互特征
- **三模型Soft Voting**：GBDT 400棵 + RF 400棵 + ET 250棵，抗过拟合
- **LOO全交叉验证**：Leave-One-Out评估，逐类分析准确率与混淆矩阵

---

## 版本历史

| 版本 | 亮点 | Task B分数 |
|------|------|-----------|
| v3-33分 | 基线PDAF-KF + SNR凝聚 | ~33分 |
| v4-35分 | 增强预处理 + 起批逻辑优化 | ~35分 |
| **v4-2** | **DBSCAN凝聚切换 + RTS平滑 + 三维时空合并** | **当前版本** |
| v5 | 仿真驱动调参 | — |
| v6-32分 | 简化起批 + 参数松弛 | ~32分 |
| v7 | 特征工程 + 后处理升级 | 迭代中 |

---

## 许可证

本项目仅用于学术竞赛交流。

---

> **⚠️ 注意**：本仓库仅包含算法代码，不包含赛题原始数据。数据需从竞赛主办方获取或使用仿真数据代替。
