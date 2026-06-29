# UAV Ring Mosaic

这是一个面向多无人机俯视场景的环绕式拼接原型工程，重点解决“建筑周边地面区域”的统一观察问题，而不是生成严格测绘意义上的正射图。

项目的基本假设是：建筑屋顶、地面、树木、车辆并不共面，因此不能简单依赖一个对整幅图都完美成立的全局变换。当前实现优先保证建筑周边地面的连续性、可观察性，以及中间过程的可控和可调试。

## 项目特点

- 支持多路无人机图像拼接
- 支持基于固定单应的多路视频拼接
- 提供自动初始化流程：相邻视角特征匹配 + Lowe ratio test + RANSAC
- 提供人工控制点标注工具，作为自动初始化失败时的回退方案
- 支持建筑区域 `mask`，用于特征匹配抑制和融合阶段弱化
- 支持全局 Homography 和 APAP 局部形变两种变换模型
- 保留调试输出，便于分析匹配质量、权重图和中间结果

## 为什么不用黑盒 Stitcher

本仓库不使用 OpenCV 黑盒 `Stitcher`。原因很直接：当前场景不是“单平面、低视差、弱遮挡”的标准全景拼接问题。若直接采用黑盒方案，往往会出现：

- 对齐建筑时，地面错位
- 对齐地面时，建筑边缘错位
- 中间过程难以解释，调参成本高

因此这里采用显式流程：

1. 基于相邻视角估计变换
2. 用建筑 `mask` 限制不稳定区域的影响
3. 统一投影到全局画布
4. 通过可配置的融合策略生成结果

## 目录结构

```text
uav_ring_mosaic/
├── configs/        # 示例配置与相机参数
├── data/           # 示例图像、mask、可选视频
├── docs/           # 说明文档
├── guideline/      # 内部开发资料（不会同步到公开仓库）
├── outputs/        # 示例输出与调试结果
├── src/            # 标注、匹配、变换、融合、主流程代码
├── .gitignore
├── README.md
└── requirements.txt
```

## 环境依赖

- Python 3.10+
- opencv-python
- numpy
- pyyaml
- tqdm

安装方式：

```bash
pip install -r requirements.txt
```

## 快速开始

仓库当前已包含示例图像与部分输出，默认配置文件为 [`configs/demo_config.yaml`](configs/demo_config.yaml)。

### 1. 标注或修正控制点

```bash
python src/annotate_points.py --config configs/demo_config.yaml
```

适用场景：

- 自动初始化效果不稳定
- 需要人工指定地面控制点
- 需要检查相邻视角之间的对应关系

### 2. 标注或修正建筑区域 Mask

```bash
python src/annotate_mask.py --config configs/demo_config.yaml
```

建筑 `mask` 主要用于：

- 在特征匹配阶段尽量避开非地面区域
- 在融合阶段降低建筑区域对地面连续性的干扰

### 3. 运行图像拼接

```bash
python src/main_image_mosaic.py --config configs/demo_config.yaml
```

### 4. 运行视频拼接

```bash
python src/main_video_mosaic.py --config configs/demo_config.yaml
```

## 配置说明

默认配置见 [`configs/demo_config.yaml`](configs/demo_config.yaml)。

关键字段包括：

- `input.mode`: 输入模式，`image` 或 `video`
- `input.image_paths`: 各路无人机图像路径
- `input.video_paths`: 各路无人机视频路径
- `masks.building_masks`: 各路建筑 `mask` 路径
- `homography.mode`: `auto` 或 `manual`
- `homography.feature_method`: 自动初始化所用特征方法
- `homography.warp_model`: `global` 或 `apap`
- `blending.method`: 融合方法
- `video.output_path`: 视频拼接输出路径

## 推荐使用流程

对于新数据，建议按下面顺序使用：

1. 先准备多路图像，确认相邻视角之间有足够重叠
2. 先尝试自动初始化
3. 若自动匹配不稳定，再使用控制点工具人工修正
4. 标注建筑 `mask`
5. 运行图像拼接验证几何效果
6. 最后切换到视频模式复用同一组变换

## 输出结果

典型输出位置：

- `outputs/homographies/control_points.yaml`
- `outputs/homographies/homographies.yaml`
- `outputs/mosaics/mosaic_result.jpg`
- `outputs/mosaics/mosaic_preview.jpg`
- `outputs/mosaics/debug/`
- `outputs/homographies/debug/`
- `outputs/videos/mosaic_output.mp4`

## 当前局限

当前仓库是一个偏工程验证性质的原型，暂时不覆盖以下能力：

- 完整三维建模
- 严格测绘级正射校正
- seam finding / graph cut / multi-band blending 等更高级融合
- 大视差场景下的强鲁棒全自动拼接
- 长时视频中的全局时序优化

## 后续可扩展方向

- 更稳健的地面区域特征筛选
- 多相机拓扑关系自动推断
- 更强的局部形变与配准优化
- 漂移修正与时间平滑
- 更精细的拼接缝处理

## 补充文档

- 使用说明：[`docs/usage.md`](docs/usage.md)
- 方法说明：[`docs/method_notes.md`](docs/method_notes.md)

如果你要将本仓库迁移到新的无人机布局或新场景，优先修改配置文件中的输入路径、相机顺序、`mask` 路径以及 `homography` 相关参数。
