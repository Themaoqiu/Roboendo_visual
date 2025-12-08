# Roboendo Visual Pipeline

## 1. 环境准备
1. 依赖安装
    ```bash
    conda create -n roboendo python=3.11
    cd Roboendo_visual
    pip install -e .
    pip install ultralytics opencv-python
    ```
2. 准备模型权重：
	- `models/yolov8x.pt`
	- `models/sam2.1_hiera_large.pt`

3. 在main.py中配置:
    ```python
    YOLO_CHECKPOINT = '/models/yolov8x.pt'
    SAM2_CHECKPOINT = '/models/sam2.1_hiera_large.pt'
    SAM2_CFG = "sam2.1_hiera_l.yaml"
    ```

## 2. 运行步骤

- 使用现成脚本运行：
  ```bash
  bash sam2/run_detect.sh
  ```
- 直接运行主脚本并可传递视频或摄像头路径：
  ```bash
  python sam2/main.py
  # 指定录像文件
  python sam2/main.py --camera_id "/home/ubuntu/dcai/roboendo/videos/test2.mp4"
  ```

执行后会弹出窗口显示分割、追踪与箭头提示，按 `q` 退出、`r` 重置。确保 `CUDA_VISIBLE_DEVICES` 设置正确（可在 `run_detect.sh` 中调整）。
