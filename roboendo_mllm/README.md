# Roboendo 实时模块

## 环境搭建

1. 先创建虚拟环境并安装依赖：

```bash
conda create -n roboendo python=3.11
pip install vllm
```

2. 若需要实时麦克风采集，请确保 `sounddevice` 与 `soundfile` 可用。Linux 还需安装 PortAudio：

```bash
sudo apt install libportaudio2
```

3. 修改 `config.MODEL_NAME` 指向模型路径，保持模型与 `vLLM` 版本兼容。

4. 修改 `config.py` 调整运行参数：

```python
VIDEO_MODE = "file"               # "realtime" 或 "file"
VIDEO_SOURCE = 0                   # 摄像头索引（整数）或视频文件路径
AUDIO_MODE = "file"               # "mic" 或 "file"
AUDIO_SOURCE = "testvoice.wav"    # 音频源（麦克风或 wav 文件）
TARGET_FPS = 5                      # 主循环帧率（默认为 5）
VISION_INFERENCE_INTERVAL = 1      # 每 1 帧做一次视觉推理（可改为 6）
FOG_THRESHOLD = 0.5
CONTAMINATION_THRESHOLD = 0.5
SAMPLING_TEMPERATURE = 0.2
SAMPLING_MAX_TOKENS = 256
```

* `VIDEO_SOURCE` 直接输入摄像头 ID（如 `0`）即可打开摄像头；输入视频路径会读取该文件。
* `TARGET_FPS` 控制读取帧率；`VISION_INFERENCE_INTERVAL` 决定推理频率（例如 5Hz = `TARGET_FPS=5` + `VISION_INFERENCE_INTERVAL=1`）。
* `FOG_THRESHOLD` / `CONTAMINATION_THRESHOLD` 决定何时输出 suction/clean。

## 如何运行

在 `realtime/` 目录下：

```bash
# 运行实时版本（按照 config 输入摄像头或视频）
python main.py

# 运行测试版本（预设音频/视频）
python test.py
```

程序会根据 `VISION_INFERENCE_INTERVAL` 采样帧，并将图像/音频块提交到 `vLLM`。视觉推理结果会通过 `generate_command` 直接打印 `clean/suction` 指令；语音命令检测到时会打印移动指令。

## 使用建议

* 确保 `config.VISION_PROMPT_TEMPLATE` / `config.AUDIO_PROMPT_TEMPLATE` 指向合法的 JSON 结构，便于 `vLLM` 输出可解析的结果。
* 如果语音推理不需要，可将 `AUDIO_MODE` 设为 `file` 并设置短音频文件以避免静默检测误触。

