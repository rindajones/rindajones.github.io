[Home](../../) | [English](./)

# 6D Pose Estimation from Monocular RGB

単眼 RGB 画像から、既知形状のマグカップの 6D Pose（位置・姿勢）を推定する。

# Contents

- [1. Overview](#1-overview)
- [2. 環境](#2-環境)
- [3. 学習データ作成](#3-学習データ作成)
- [4. 物体検知モデル構築](#4-物体検知モデル構築)
- [5. キーポイントモデル構築](#5-キーポイントモデル構築)
- [6. 姿勢推定](#6-姿勢推定)
- [7. Raspberry Pi との性能比較](#7-raspberry-pi-との性能比較)
- [8. アノテーション改善](#8-アノテーション改善)

# 1. Overview

本プロジェクトでは、単眼 RGB 画像から、既知形状のマグカップの 6D Pose（位置・姿勢）を推定する。

まず Blender 上でカメラおよびマグカップの位置・姿勢をランダムに変化させ、3,000 枚の合成画像と Ground Truth を生成する。この Sim データを使用して、マグカップを画像から検出する物体検知モデルを構築する。

次に、物体検知モデルによって得られた BBox を基準に画像をクロップし、マグカップの 3D モデル上にあらかじめ定義したキーポイントを検出するキーポイントモデルを学習する。

実写画像の推論では、以下の Pipeline を使用する。

**RGB Image → Object Detection → BBox Crop → Keypoint Detection → PnP → 6D Pose**

PnP では、画像から検出した 2D キーポイントと、対応する既知の 3D キーポイント座標から、カメラ座標系におけるマグカップの位置・姿勢を推定する。

さらに、Sim データのみで学習したキーポイントモデルを 173 枚の実写アノテーションデータで fine-tuning する。最終的に、961 枚の実写画像に対して、

- **Sim Only**：Sim データのみで学習したキーポイントモデル
- **Real Fine-tuning**：Sim モデルを実写データで fine-tuning したモデル

の2モデルを同じ Pipeline で評価し、キーポイント confidence、PnP inlier 数、reprojection error、`solvePnPRansac` の失敗数を比較する。

さらに、実写キーポイントアノテーションの幾何学的一貫性を改善し、PnP姿勢推定の安定性への影響も確認する。

なお、実写画像には 6D Pose の Ground Truth が存在しないため、本資料では推定 Pose の絶対精度は評価しない。代わりに、キーポイント confidence、PnP inlier 数、reprojection error、RANSAC failure 数、およびフレーム間の姿勢推定の安定性を指標として比較する。

# 2. 環境

```
tool-detection-pose/
├── blender
├── cam_calib
├── detection
├── docs
├── keypoint
├── README.md
└── real_dataset
```

```bash
$ cd keypoint/
$ source .venv/bin/activate
(.venv) $ python --version
Python 3.11.2
```

# 3. 学習データ作成

カメラを動かして、カップのランダムな位置と向きの学習データを作成。

```
├── blender
│   ├── 0550.txt
│   ├── 0551.txt
│   ├── 0552.txt
│   ├── datagen_random_camera_mug.py
│   ├── synthetic_dataset_0550_randomcam
│   ├── synthetic_dataset_0551_randomcam
│   ├── synthetic_dataset_0552_randomcam
│   └── tool_pose.blend
```

```
blender -b tool_pose.blend -P datagen_random_camera_mug.py -- 0550.txt
```

`0550.txt` `0551.txt` `0552.txt` は、カメラとカップの位置と向き。
```
IMG_0550

#1 
Camera
  location = (1.10000002, -0.18405022, 1.08624732)
  rotation_deg = (75.080479, 1.535211, 92.172959)

Mug_Root
  location = (0.43371654, -0.24633931, 0.78921199)
  rotation_deg = (90.049760, -331.921048, -66.042895)
...  
```

学習データ総数 `3,000` 件：
```
synthetic_dataset_0550_randomcam
├── gt
│   ├── s0550_0000.json
...
│   └── s0550_0999.json
└── images
    ├── s0550_0000.png
...
    └── s0550_0999.png
```
![0550](images/s0550_0018.png)

```
synthetic_dataset_0551_randomcam
├── gt
│   ├── s0551_0000.json
...
│   └── s0551_0999.json
└── images
    ├── s0551_0000.png
...
    └── s0551_0999.png
```
![0551](images/s0551_0019.png)

```
synthetic_dataset_0552_randomcam
├── gt
│   ├── s0552_0000.json
...
│   └── s0552_0999.json
└── images
    ├── s0552_0000.png
...
    └── s0552_0999.png
```
![0552](./images/s0552_0031.png)


# 4. 物体検知モデル構築

## 4.1 Model

物体検知には PyTorch / TorchVision の `ssdlite320_mobilenet_v3_large` を使用する。

- Source image resolution: 1920 × 1080
- Detector: SSDLite320 MobileNetV3-Large
- Detector internal input size: 320 × 320
- Classes: Background / Combination Wrench / Mug
- Output: Class, confidence score, BBox

本モデルは後段のキーポイント検知のために、マグカップ領域を検出・クロップする目的で使用する。

`train.py` `dataset.py` を使用、`models/` へ

```bash
(.venv) detection$ python train.py 
Device: cuda
Train samples: 3000
Epoch 001/100 train_loss=3.324669
Epoch 002/100 train_loss=1.868230
Epoch 003/100 train_loss=1.370852
Epoch 004/100 train_loss=1.174247
Epoch 005/100 train_loss=1.005588
  saved: models/ssdlite320_sim0550_0552_epoch_005.pth
Epoch 006/100 train_loss=0.940396
...
```

50 epoch 学習し、5 epoch 毎に重みファイルを保存：
```bash
(.venv) rinda@P1G7 detection$ ls models/
ssdlite320_sim0550_0552_epoch_005.pth  ssdlite320_sim0550_0552_epoch_020.pth  ssdlite320_sim0550_0552_epoch_035.pth  ssdlite320_sim0550_0552_epoch_050.pth
ssdlite320_sim0550_0552_epoch_010.pth  ssdlite320_sim0550_0552_epoch_025.pth  ssdlite320_sim0550_0552_epoch_040.pth
ssdlite320_sim0550_0552_epoch_015.pth  ssdlite320_sim0550_0552_epoch_030.pth  ssdlite320_sim0550_0552_epoch_045.pth
```

各重みファイルで実写データを推論：
```bash
(.venv) detection$ python infer_real_multi_pth.py 
```

推論結果：
```bash
results_real_randomcam_multi_pth
├── all_epochs_scene_summary.csv
├── epoch_010
│   ├── frame_results.csv
│   ├── scene_summary.csv
│   └── vis
├── epoch_020
│   ├── frame_results.csv
│   ├── scene_summary.csv
│   └── vis
├── epoch_030
│   ├── frame_results.csv
│   ├── scene_summary.csv
│   └── vis
├── epoch_040
│   ├── frame_results.csv
│   ├── scene_summary.csv
│   └── vis
└── epoch_050
    ├── frame_results.csv
    ├── scene_summary.csv
    └── vis
```

`epoch_030` の重みファイルを検知モデルとして採用。以下 `epoch_030/scene_summary.csv` から：

| Object | Frames | Success | Low Confidence | Total Detected | Detection Rate | Miss |
|---|---:|---:|---:|---:|---:|---:|
| Mug | 961 | **961** | 0 | **961** | **100.0%** | 0 |
| Combination Wrench | 961 | 449 | 208 | **657** | **68.4%** | 304 |

`Mug` の検知は `100%` で、今回の実写データはシミュレーションモデルで検知可能と判断。

`Combination Wrench` の検知率は低い。原因として、シミュレーション上の Combination Wrench と実物の外観差が大きいことが考えられる。

**以降では 6D Pose 推定の対象である Mug のみを扱い、Combination Wrench については言及しない。**

[Synthetic-to-Real Object Detection | Blender + PyTorch on YouTube](https://youtube.com/shorts/THVDTcLsF0w)



# 5. キーポイントモデル構築

## 5.1 Model

キーポイント検知には Heatmap Regression ベースのモデルを使用する。

- Input: 420 × 420 cropped RGB image
- Keypoints: 11 points
- Output: 11 keypoint heatmaps
- Heatmap size: 112 × 112
- Keypoint position: Heatmap peak position
- Training: Synthetic data → Real fine-tuning

各キーポイントはマグカップの既知 3D モデル上の点と対応しており、検出した 2D 座標と既知の 3D 座標を PnP に使用する。

```
keypoint/
├── dataset
├── model
├── output
├── training
```

`dataset`フォルダ：

```
dataset/
├── real
│   ├── detection
│   ├── gt
│   ├── gt_visuals
│   ├── images
│   └── metadata.csv
├── real_anno
│   ├── IMG_0538_00008.jpg
│   ├── IMG_0538_00008.json
...
│   ├── IMG_0552_00088.jpg
│   └── IMG_0552_00088.json
└── sim
    ├── detection
    ├── gt
    ├── gt_visuals
    ├── images
    └── metadata.csv
```

`real_anno`：実写データをアノテーションしたデータ。
`sim`：Sim データ、`datagen_keypoint_sim.py` で作成。
`real`：Sim モデルを fine-tuning する際の実写データ、`datagen_keypoint_real.py` で作成。


## 5.2 Sim モデル構築

### 5.2.1 学習データ作成

`datagen_keypoint_sim.py` 

```python
SOURCE_DIRS = [
    Path("../blender/synthetic_dataset_0550_randomcam"),
    Path("../blender/synthetic_dataset_0551_randomcam"),
    Path("../blender/synthetic_dataset_0552_randomcam"),
]

MODEL_PATH = Path(
    "../detection/models/"
    "ssdlite320_sim0550_0552_epoch_030.pth"
)

CROP_SIZE = 420
OUTPUT_SIZE = 420
```

Sim データ `SOURCE_DIRS` を検知モデル `MODEL_PATH` で推論した結果を元に、BBox からクロップした画像を `420x420` として出力（BBox を十分に囲むサイズ）。

出力：
```
dataset/sim
├── detection
├── gt
├── gt_visuals
├── images
└── metadata.csv
```

### 5.2.2 モデル構築
`train.py` `model.py` `dataset.py` を使用して構築：

```bash
(.venv) keypoint$ python training/train.py 
Device: cuda
Train samples: 2971
Epoch 001 | train loss=0.003636 mean=147.95px max=497.90px
  [BEST] epoch=1 train_mean=147.95px
Epoch 002 | train loss=0.000607
Epoch 003 | train loss=0.000579
Epoch 004 | train loss=0.000573
Epoch 005 | train loss=0.000569
Epoch 006 | train loss=0.000567
Epoch 007 | train loss=0.000567
Epoch 008 | train loss=0.000565
Epoch 009 | train loss=0.000562
Epoch 010 | train loss=0.000558 mean=129.44px max=429.86px
  [BEST] epoch=10 train_mean=129.44px
Epoch 011 | train loss=0.000550
...
Epoch 200 | train loss=0.000007 mean=  1.57px max=104.14px

============================================================
TRAINING COMPLETE
============================================================
Best epoch       : 190
Best train error : 1.56px
Best train mean  : 1.56px

Train per-keypoint error:
  K0: mean=  1.56px max=  3.89px
  K1: mean=  1.57px max=  4.30px
  K2: mean=  1.56px max=  4.53px
  K3: mean=  1.57px max=  4.70px
  T0: mean=  1.53px max=  4.03px
  T1: mean=  1.59px max=  4.23px
  T2: mean=  1.57px max=  4.27px
  T3: mean=  1.61px max= 71.67px
  K4: mean=  1.52px max=  4.06px
  K5: mean=  1.56px max=  4.45px
  K6: mean=  1.53px max= 39.06px

Best model : model/sim/best_model.pth
History    : model/sim/training_history.csv
Visuals    : model/sim/train_visuals

```
### 5.2.3 Large-error samples

Best model (epoch 190) の学習データ上の平均キーポイント誤差は 1.56 px だった。

- T3: max 71.67 px
- K6: max 39.06 px
- その他のキーポイント誤差: max 5.01 px

該当画像を確認すると、T3 / K6 はいずれも他の物体によってキーポイント位置が遮蔽されていた。

![T3 occlusion](./images/s0552_0744.png)
![K6 occlusion](./images/s0552_0054.png)

このため、大きな誤差は通常の可視キーポイントの検出失敗ではなく、occlusion により直接観測できないキーポイントの推定で発生している。

### 5.2.4 Best Model

200 epoch 学習し、10 epoch 毎に重みファイルを保存：

```bash
(.venv) keypoint$ ls model/sim
best_model.pth  epoch_020.pth  epoch_050.pth  epoch_080.pth  epoch_110.pth  epoch_140.pth  epoch_170.pth  epoch_200.pth
epoch_001.pth   epoch_030.pth  epoch_060.pth  epoch_090.pth  epoch_120.pth  epoch_150.pth  epoch_180.pth  training_history.csv
epoch_010.pth   epoch_040.pth  epoch_070.pth  epoch_100.pth  epoch_130.pth  epoch_160.pth  epoch_190.pth  train_visuals
```

`best_model.pth` は学習中に train mean error が最小となった checkpoint の model weights を保存したもので、今回の学習では epoch 190 の weights が選択された。


## 5.3 Real モデル構築

### 5.3.1 実写データのアノテーション

**961** 枚の実写データの **173** 枚にキーポイントアノテーションを GT として定義。
![実写アノテーション](./images/annotation.png)

### 5.3.2 学習データ作成

`datagen_keypoint_real.py` 

```python
DEFAULT_SOURCE_DIR = Path("dataset/real_anno")
DEFAULT_OUTPUT_DIR = Path("dataset/real")

DEFAULT_MODEL_PATH = Path(
    "../detection/models/"
    "ssdlite320_sim0550_0552_epoch_030.pth"
)

NUM_CLASSES = 3
MUG_LABEL = 2
SCORE_THRESHOLD = 0.5

CROP_SIZE = 420
OUTPUT_SIZE = 420

KEYPOINT_NAMES = [
    "K0",
    "K1",
    "K2",
    "K3",
    "T0",
    "T1",
    "T2",
    "T3",
    "K4",
    "K5",
    "K6",
]
```

実写をアノテーションしたデータ `DEFAULT_SOURCE_DIR` から、fine-tuning のための学習データ `DEFAULT_OUTPUT_DIR` を作成。物体検知モデル `DEFAULT_MODEL_PATH` を使用、サイズ `420x420` の画像を作成。`DEFAULT_SOURCE_DIR` から `KEYPOINT_NAMES` 等の `GT` を作成。

出力：
```
dataset/real
├── detection
├── gt
├── gt_visuals
├── images
└── metadata.csv
```

### 5.3.3 モデル構築

`train_real_fit.py` `model.py` `dataset_real.py` を使用して構築。

`dataset_real.py`

```python
DATASET_ROOT = (
    "dataset/real"
)

BASE_MODEL = Path(
    "model/sim/best_model.pth"
)

OUTPUT_DIR = Path(
    "model/real"
)

EPOCHS = 300
```
学習データ `DATASET_ROOT`、fine-tuningするモデル `BASE_MODEL`、出力先 `OUTPUT_DIR`、epoch `300`


```bash
(.venv)  keypoint$ python training/train_real_fit.py 
Device: cuda
Train samples: 173
Skipped: {'missing_image': 0, 'no_keypoint': 0, 'invalid_gt': 0}
Base model: model/sim/best_model.pth
Base epoch: 190
Base synthetic train mean: 1.56px

Before fine-tuning | mean=126.23px max=425.33px valid=1414

Before fine-tuning per-keypoint error:
  K0: n=109 mean=156.66px max=299.01px
  K1: n=108 mean=185.39px max=404.57px
  K2: n=125 mean=136.62px max=348.14px
  K3: n= 91 mean=155.22px max=276.81px
  T0: n=145 mean= 85.96px max=359.21px
  T1: n=128 mean=116.61px max=411.70px
  T2: n=135 mean=161.65px max=390.46px
  T3: n=166 mean=145.10px max=385.73px
  K4: n=118 mean=108.26px max=344.40px
  K5: n=116 mean= 81.79px max=286.29px
  K6: n=173 mean= 84.57px max=425.33px

Epoch 001 | train loss=0.000556 mean=116.90px max=422.06px valid=1414
  [BEST] epoch=1 train_mean=116.90px
Epoch 002 | train loss=0.000529
Epoch 003 | train loss=0.000506
Epoch 004 | train loss=0.000481
Epoch 005 | train loss=0.000453
Epoch 006 | train loss=0.000420
Epoch 007 | train loss=0.000386
Epoch 008 | train loss=0.000353
Epoch 009 | train loss=0.000325
Epoch 010 | train loss=0.000300 mean= 33.57px max=420.29px valid=1414
  [BEST] epoch=10 train_mean=33.57px
Epoch 011 | train loss=0.000278

...

Epoch 299 | train loss=0.000003
Epoch 300 | train loss=0.000003 mean=  1.47px max=  3.00px valid=1414
  [BEST] epoch=300 train_mean=1.47px

============================================================
REAL FINE-TUNING COMPLETE
============================================================
Best epoch       : 300
Best train error : 1.47px
Best train mean  : 1.47px
Best train max   : 3.00px

Best real-fit per-keypoint error:
  K0: n=109 mean=  1.42px max=  2.81px
  K1: n=108 mean=  1.48px max=  2.54px
  K2: n=125 mean=  1.45px max=  2.82px
  K3: n= 91 mean=  1.41px max=  2.60px
  T0: n=145 mean=  1.54px max=  2.65px
  T1: n=128 mean=  1.44px max=  3.00px
  T2: n=135 mean=  1.43px max=  2.58px
  T3: n=166 mean=  1.49px max=  2.73px
  K4: n=118 mean=  1.55px max=  2.85px
  K5: n=116 mean=  1.54px max=  2.56px
  K6: n=173 mean=  1.45px max=  2.85px

Best model : model/real/best_model.pth
History    : model/real/training_history.csv
Visuals    : model/real/train_visuals
(.venv) rinda@P1G7 keypoint$ df -k
Filesystem     1K-blocks      Used Available Use% Mounted on
tmpfs            3234836      3136   3231700   1% /run
/dev/nvme0n1p2 490048472 368155536  96926332  80% /
tmpfs           16174168     71580  16102588   1% /dev/shm
tmpfs               5120         8      5112   1% /run/lock
efivarfs             172        80        88  48% /sys/firmware/efi/efivars
/dev/nvme0n1p1   1098632      6288   1092344   1% /boot/efi
tmpfs            3234832       148   3234684   1% /run/user/1000
```

epoch 300 における学習データ上の平均キーポイント誤差は `1.47` px となった。

# 6. 姿勢推定

`run_pipeline_real.py`
```python
DEFAULT_FRAMES_ROOT = Path(
    "../real_dataset/frames"
)

DEFAULT_DETECTOR_PATH = Path(
    "../detection/models/ssdlite320_sim0550_0552_epoch_030.pth"
)

DEFAULT_KEYPOINT_MODEL_PATH = Path(
    "model/sim/best_model.pth"
    # "model/real/best_model.pth"
)

DEFAULT_OUTPUT_DIR = Path(
    "output/pipeline_keypoint_real_simonly"
    # "output/pipeline_keypoint_real"
)

RANSAC_REPROJ_ERROR = 12.0 #8.0 #5.0
FINAL_INLIER_ERROR = 12.0 #8.0 #5.0
```
`DEFAULT_DETECTOR_PATH`：検知モデル、Sim モデルを使用。
`DEFAULT_KEYPOINT_MODEL_PATH`：キーポイント検知モデル。
`DEFAULT_OUTPUT_DIR`：姿勢推定結果。

実写アノテーションを用いた検証では数 px 程度のアノテーション誤差が確認されたため、複数の閾値を試した結果、本評価では `12.0` px を採用した。

なお PnP では、3Dモデル上に定義した11点のキーポイントと、Keypoint model が推定した画像上の2D点を対応させる。3D keypoints は以下の座標を使用した。単位は meter。
```
| Point | X | Y | Z |
|---|---:|---:|---:|
| K0 | 0.036510 | -0.000531 | 0.001519 |
| K1 | -0.036406 | -0.000531 | 0.001519 |
| K2 | -0.000755 | -0.035998 | 0.001519 |
| K3 | -0.000755 | 0.036682 | 0.001519 |
| T0 | 0.036510 | -0.000531 | 0.088068 |
| T1 | -0.036406 | -0.000531 | 0.088068 |
| T2 | -0.000755 | -0.035998 | 0.088068 |
| T3 | -0.000755 | 0.036682 | 0.088068 |
| K4 | 0.037754 | -0.000025 | 0.078986 |
| K5 | 0.037754 | -0.000025 | 0.024736 |
| K6 | 0.071656 | 0.000064 | 0.061815 |
```

## 6.1 Sim モデル

`model/sim/best_model.pth` を使用。

```bash
(.venv) rinda@P1G7 keypoint$ python training/run_pipeline_real.py 
Device         : cuda
Frames root    : ../real_dataset/frames
Detector       : ../detection/models/ssdlite320_sim0550_0552_epoch_030.pth
Keypoint model : model/sim/best_model.pth
Canonical crop : 420x420
KP input       : 420x420
KP heatmap     : 112x112
KP checkpoint  : epoch=190
Processed: 1/961

...

Processed: 950/961

============================================================
UNIFIED REAL PIPELINE COMPLETE
============================================================
Images       : 961
Processed    : 961
No detection : 0
Output       : output/pipeline_keypoint_real_simonly
CSV log      : output/pipeline_keypoint_real_simonly/pipeline_log.csv
```

```
pipeline_keypoint_real_simonly
├── IMG_0538
│   ├── IMG_0538_00001_crop.png
│   ├── IMG_0538_00001_detection.png
│   ├── IMG_0538_00001_keypoints_crop.png
│   ├── IMG_0538_00001_keypoints_original.png
│   ├── IMG_0538_00001_pnp.png
│   ├── IMG_0538_00002_crop.png
│   ├── IMG_0538_00002_detection.png
│   ├── IMG_0538_00002_keypoints_crop.png
│   ├── IMG_0538_00002_keypoints_original.png
│   ├── IMG_0538_00002_pnp.png
...
├── IMG_0539
├── IMG_0540
├── IMG_0541
├── IMG_0542
├── IMG_0543
├── IMG_0544
├── IMG_0546
├── IMG_0547
├── IMG_0548
├── IMG_0549
├── IMG_0550
├── IMG_0551
├── IMG_0552
└── pipeline_log.csv
```

## 6.2 Real モデル（fine-tuning モデル）

`model/real/best_model.pth` Real モデルを使用。


```bash
(.venv) keypoint$ python training/run_pipeline_real.py 
Device         : cuda
Frames root    : ../real_dataset/frames
Detector       : ../detection/models/ssdlite320_sim0550_0552_epoch_030.pth
Keypoint model : model/real/best_model.pth
Canonical crop : 420x420
KP input       : 420x420
KP heatmap     : 112x112
KP checkpoint  : epoch=300
Processed: 1/961
Processed: 50/961

...

Processed: 900/961
Processed: 950/961

============================================================
UNIFIED REAL PIPELINE COMPLETE
============================================================
Images       : 961
Processed    : 961
No detection : 0
Output       : output/pipeline_keypoint_real
CSV log      : output/pipeline_keypoint_real/pipeline_log.csv
```

```
pipeline_keypoint_real
├── IMG_0538
│   ├── IMG_0538_00001_crop.png
│   ├── IMG_0538_00001_detection.png
│   ├── IMG_0538_00001_keypoints_crop.png
│   ├── IMG_0538_00001_keypoints_original.png
│   ├── IMG_0538_00001_pnp.png
│   ├── IMG_0538_00002_crop.png
│   ├── IMG_0538_00002_detection.png
│   ├── IMG_0538_00002_keypoints_crop.png
│   ├── IMG_0538_00002_keypoints_original.png
│   ├── IMG_0538_00002_pnp.png
...
├── IMG_0539
├── IMG_0540
├── IMG_0541
├── IMG_0542
├── IMG_0543
├── IMG_0544
├── IMG_0546
├── IMG_0547
├── IMG_0548
├── IMG_0549
├── IMG_0550
├── IMG_0551
├── IMG_0552
└── pipeline_log.csv
```

`IMG_0538_00001_detection.png`
![IMG_0538_00001_detection](./images/real/IMG_0538_00001_detection.png)
`IMG_0538_00001_crop.png`
![IMG_0538_00001_crop](./images/real/IMG_0538_00001_crop.png)
`IMG_0538_00001_keypoints_crop.png`
![IMG_0538_00001_keypoints_crop](./images/real/IMG_0538_00001_keypoints_crop.png)
`IMG_0538_00001_keypoints_original.png`
![IMG_0538_00001_detection](./images/real/IMG_0538_00001_keypoints_original.png)
`IMG_0538_00001_pnp.png`
![IMG_0538_00001_pnp](./images/real/IMG_0538_00001_pnp.png)

## 6.3 Sim / Real モデル比較

実写データ 961 枚に対して、Sim モデルと Real fine-tuning モデルを使用して
キーポイント検知および PnP を実行した結果を比較する。

| 指標 | Real Fine-tuning | Sim Only |
|---|---:|---:|
| Frames | 961 | 961 |
| Mean Keypoint Confidence | **0.604** | 0.104 |
| Mean PnP Final Inliers | **7.10** | 2.22 |
| Mean PnP All RMSE | **59.3 px** | 134.3 px |
| solvePnPRansac Failed | **21** | 538 |

Real fine-tuning モデルでは、Sim Only モデルと比較してキーポイントの confidence が高く、PnP の final inlier 数も増加した。特に `solvePnPRansac` の失敗は、Sim Only の 538 件に対して Real Fine-tuning では 21 件まで減少した。

この結果から、Real Fine-tuning により実写画像に対するキーポイント推定の confidence と幾何的一貫性が改善し、その結果として PnP の安定性が大きく向上したと判断する。ただし、実写画像には 6D Pose の Ground Truth が存在しないため、ここで評価しているのは姿勢推定の絶対精度ではなく、推定処理の安定性である。

[6D Pose Estimation – Before Real-World Fine-Tuning (Sim Only) on YouTube](https://youtu.be/XR3__jy2JHs)
[6D Pose Estimation – After Real-World Fine-Tuning on YouTube](https://youtu.be/3KiCMTUQuoI)


# 7. Raspberry Pi との性能比較

学習および評価に使用したUbuntu + CUDA環境と、Raspberry Pi上で同一の推論パイプラインを実行し、推論結果と処理速度を比較した。

## 7.1 Raspberry Pi

### 7.1.1 実行環境

| 項目 | 内容 |
|---|---|
| Device | Raspberry Pi 5 Model B Rev 1.0 |
| CPU | ARM Cortex-A76, 4 cores, max 2.4 GHz |
| Memory | 4 GB |
| OS | Debian GNU/Linux 12 (bookworm) |
| ONNX Runtime | 1.29.0 |
| Execution Provider | CPUExecutionProvider |

学習済みの物体検知モデルとキーポイント推定モデルをONNX形式に変換し、Raspberry Pi上で推論パイプラインを実行した。

使用したモデルを以下に示す。

| モデル | ONNXモデル |
|---|---|
| 物体検知 | `ssdlite320_sim0550_0552_epoch_030.onnx` |
| キーポイント推定 | `best_model.onnx` |

推論パイプラインは以下の構成とした。

**物体検知 → クロップ → キーポイント推定 → PnP姿勢推定**

## 7.2 Ubuntu + CUDA

### 7.2.1 実行環境

| 項目 | 内容 |
|---|---|
| CPU | Intel Core Ultra 7 155H, 16 cores / 22 threads, max 4.8 GHz |
| GPU | NVIDIA RTX 2000 Ada Generation, 8 GB |
| Memory | 30 GB |
| OS | Ubuntu 24.04.4 LTS |
| NVIDIA Driver | 580.126.09 |
| CUDA | 13.0 |
| Inference | PyTorch / CUDA |


Raspberry Piと同一の入力画像を使用し、同一構成の推論パイプラインを実行した。


## 7.3 性能比較

同一の961枚の実写画像を使用し、Raspberry Pi 5とUbuntu + CUDA環境で推論結果および処理速度を比較した。

| Platform | Detection | Keypoint | PnP | Total | FPS |
|---|---:|---:|---:|---:|---:|
| Raspberry Pi 5 / CPU | 70.36 ms | 56.06 ms | 5.34 ms | 132.12 ms | 7.60 |
| Ubuntu / RTX 2000 Ada / CUDA | 60.25 ms | 21.87 ms | 5.18 ms | 87.71 ms | 11.60 |

Ubuntu + CUDA環境では、パイプライン全体の処理速度はRaspberry Pi 5の約1.5倍となった。特にキーポイント推定では、56.06 msから21.87 msとなり、約2.6倍高速であった。

一方、推論結果は両環境でほぼ一致した。物体検知スコアは全961画像で一致し、BBox座標差も最大0.001 px未満であった。また、11点のキーポイント座標は全961画像で一致した。

PnPの判定結果も全体としてほぼ一致したが、RANSACの非決定性により、961画像中2画像でRANSAC_FAILとなるフレームに違いが見られた。

[6D Pose Estimation: CUDA vs Raspberry Pi 5 on YouTube](https://youtu.be/zFwZ-Rls9bs)

# 8. アノテーション改善

## 8.1 背景

手動アノテーションでは、画像上で正確なキーポイント位置を定義することが難しい。特に今回のマグカップのように、画像上から3D形状上の対応点を正確に判断しにくい物体では、アノテーションそのものに誤差や幾何学的な不整合が生じる。

その結果、fine-tuning 後のキーポイント推論が一見良好であっても、後段の PnP による姿勢推定では推定座標軸の向きや位置が不安定になるケースが確認された。

## 8.2 手法

実写画像を Blender の背景画像として配置し、対応する3Dオブジェクトを実写画像上の物体に重ね合わせる。これにより、3Dモデル上で定義したキーポイントを画像平面へ投影し、**3D / 2D の対応関係が一貫したアノテーション**を作成する。

### 8.2.1 実写物体と3Dオブジェクトの位置合わせ

実写の物体と3Dオブジェクトが一致するように調整する。

![](./images/annon_01.png)

### 8.2.2 アノテーション対象画像を背景画像に設定

![](./images/annon_02.png)

### 8.2.3 3Dオブジェクトを実写画像に重ねる

3Dオブジェクトを `Wireframe` 表示し、実写画像上の物体と一致するように位置・姿勢を調整する。

![](./images/annon_03.png)

位置合わせ後、`.blend` ファイルを保存して以下のスクリプトを実行する。

```bash
/Applications/Blender5_1_2.app/Contents/MacOS/Blender \
    -b tool_pose.blend -P save_source_gt.py
```

**Note**: 本アノテーション作りは **Mac で実施** 。


出力：`source_GT.jsonl` 実写映像に対する、対象物の3D/2D情報を保存。

## 8.3 学習データ作成

### 8.3.1 3D / 2D キーポイントの投影

スクリプト：`posegt_project_source.py` 
入力　　　：`posegt_source_GT.jsonl`
出力　　　：`posegt_projected_GT.jsonl`
```bash
blender$ blender5 -b tool_pose.blend -P posegt_project_source.py
```

### 8.3.2 Real dataset生成

スクリプト：`posegt_datagen_real.py`
```python
DEFAULT_SOURCE_GT = Path(
    "../blender/posegt_projected_GT.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "dataset/real_posegt"
)

REAL_IMAGE_DIR = Path(
    "dataset/real_anno"
)
```

`DEFAULT_SOURCE_GT` `REAL_IMAGE_DIR` を入力として学習データを作成：
```bash
(.venv) keypoint$ python training/posegt_datagen_real.py
```

### 8.3.3 Fine-tuning

学習スクリプト：`train_real_fit.py`
```python
DATASET_ROOT = (
    # "dataset/real"
    "dataset/real_posegt"
)

BASE_MODEL = Path(
    "model/sim/best_model.pth"
)

OUTPUT_DIR = Path(
    # "model/real"
    "model/real_posegt"
)
```

```bash
(.venv) keypoint$ python training/train_real_fit.py 
```

### 8.3.4 推論

推論スクリプト：`run_pipeline_real.py`
```python
DEFAULT_KEYPOINT_MODEL_PATH = Path(
    # "model/sim/best_model.pth"
    # "model/real/best_model.pth"
    "model/real_posegt/best_model.pth"
)

DEFAULT_OUTPUT_DIR = Path(
    # "output/pipeline_keypoint_real_simonly"
    "output/pipeline_keypoint_posegt"
)
```
```bash
(.venv) keypoint$ python training/run_pipeline_real.py 
```

## 8.4 結果

今回は、推論結果の動画で使用した以下のシーンの **59** 枚を再アノテーション：
```
    "IMG_0540",
    "IMG_0541",
    "IMG_0543",
    "IMG_0546",
    "IMG_0548",
    "IMG_0549",
    "IMG_0550",
    "IMG_0552",
```

アノテーション改善後は、改善前に見られた 推定座標軸の大きな反転や急激な変動が減少し、フレーム間でより一貫した姿勢推定結果が得られた。定量的な Pose GT は存在しないため、ここでは主に可視化結果から時間方向の安定性を確認している。

[Watch the Before/After video on YouTube](https://youtu.be/rC1AJGKOdmE)
