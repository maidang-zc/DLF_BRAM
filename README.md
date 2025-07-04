---

# 🎭 Detecting Deceptive Behavior via Learning Relation-Aware Visual Representations

---

## 📁 项目结构

```
DLF_BRAM/
├── main.py                  # 训练与测试主程序
├── models/
│   └── DLF_BRAM.py          # 模型结构定义
├── datasets/
│   └── mydataloader.py      # 数据集加载器
├── checkpoints/            # 模型保存路径
├── data/                   # 数据根目录（frames、annotations 组织）
├── log/                    # 训练日志保存路径
├── requirements.txt        # 依赖项（推荐）
└── README.md               # 本文件
```

---

## 🔧 安装环境

建议使用 Python 3.7+ 与以下主要依赖：

```bash
pip install -r requirements.txt
```

如未提供 `requirements.txt`，请确保以下依赖已安装：

```bash
pip install torch torchvision scikit-learn numpy
```

---

## 🚀 快速开始

### 1. 数据准备

确保你的数据目录结构如下（以 Dolos 为例）：

```
data/
└── Dolos/
    ├── frames/                # 每个视频一个子文件夹，内含图像帧
    └── keyblock7head/         # 对应关键区域注释 .json 文件夹（5/7块模式）
```

### 2. 训练模型

```bash
python main.py \
    --data_name Dolos \
    --train_flag 1 \
    --len 4 \
    --blocks 5 \
    --depth 4 \
    --size 96 \
    --batch_size 32 \
    --lr 1e-6 \
    --num_epochs 100 \
    --device cuda:0
```

你可以自定义以下参数：

| 参数             | 含义               | 默认值       |
| -------------- | ---------------- | --------- |
| `--data_name`  | 使用数据集名称（如 Dolos） | `'Dolos'` |
| `--train_flag` | 数据划分编号（1/2/3）    | `1`       |
| `--len`        | 每个样本帧数           | `4`       |
| `--blocks`     | 块数（5或7）          | `5`       |
| `--depth`      | Transformer 层数   | `4`       |
| `--size`       | 输入图像块尺寸          | `96`      |
| `--batch_size` | 批次大小             | `32`      |
| `--num_epochs` | 训练轮数             | `100`     |
| `--lr`         | 初始学习率            | `1e-6`    |
| `--device`     | 使用GPU设备          | `cuda:0`  |

### 3. 测试模型

```bash
python main.py \
    --data_name Dolos \
    --train_flag 1 \
    --len 4 \
    --blocks 5 \
    --depth 4 \
    --size 96 \
    --batch_size 32 \
    --test \
    --pretrained_path ./checkpoints/Dolos_1_DLF_BRAM_4_5/bestepoch.pth
```

---

## 🧠 模型介绍

DLF-BRAM 包括以下关键设计：

* ✅ **多分支结构**：三个分支分别建模不同空间-块信息。
* ✅ **块级区域建模**：基于 AU 选择的区域进行局部建模。
* ✅ **动态权重更新**：训练时根据可识别性分支性能自适应融合。
* ✅ **支持5块/7块设置**：可选不同区域划分粒度。
* ✅ **交叉验证**：通过 `--train_flag` 参数切换数据划分。

---

## 📊 日志与结果

训练日志保存在：

```
log/2025-07-04_Dolos_1_DLF_BRAM_4_5.txt
```

模型权重保存在：

```
checkpoints/Dolos_1_DLF_BRAM_4_5/bestepoch.pth
```


---

## 💡 引用方式

如果你在研究中使用了本代码，请引用：

> *Zhu et al., "DLF-BRAM: Dual-Level Block Relation-Aware Modeling for Deception Detection", TIFS 2025 (Under Review).*

---

## 📬 联系方式

如有问题或合作意向，请联系项目负责人：

```
朱俊伟 - 浙江工商大学
邮箱: zc123***@mail.zjgsu.edu.cn
```

---

如果你需要我帮你根据现有目录自动生成 `.gitignore` 或 `requirements.txt`，我也可以一并生成。是否需要？
