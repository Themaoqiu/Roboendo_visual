MODEL_NAME = "model/Qwen2.5-Omni-3B"
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.4

VIDEO_MODE = "file" # "realtime" 或 "file"
VIDEO_SOURCE = "Roboendo/2e70a22a7a8ee3a8dbe1df9dd84f19d3.mp4"  # 摄像头ID(实时模式) 或 "path/to/video.mp4"(文件模式)

AUDIO_MODE = "mic" # "mic"(实时模式) 或 "file"(文件模式)
AUDIO_SOURCE = "Roboendo/testvoice.wav"  # "mic"(实时模式) 或 "path/to/audio.wav"(文件模式)
AUDIO_SAMPLE_RATE = 16000

TARGET_FPS = 30
CYCLE_PERIOD = 1.0 / TARGET_FPS  

# 视觉分析频率
VISION_INFERENCE_INTERVAL = 6

# 音频缓冲时长（秒）
AUDIO_BUFFER_DURATION = 3.0

VAD_ENERGY_THRESHOLD = 0.01  # 能量阈值
VAD_MIN_SPEECH_FRAMES = 3  # 最少连续语音帧数（确认开始说话）
VAD_MIN_SILENCE_FRAMES = 5  # 最少连续静音帧数（确认说话结束）

EMA_ALPHA = 0.3 

FOG_THRESHOLD = 0.5  # 雾气触发阈值
CONTAMINATION_THRESHOLD = 0.1  # 污染触发阈值
STATE_CONFIRM_FRAMES = 3  # 状态确认需要的连续帧数

SEVERITY_MAP = {
    "none": 0.0,
    "mild": 0.3,
    "moderate": 0.6,
    "severe": 1.0
}

DEFAULT_MOVE_DISTANCE = 2

DIRECTION_MAP = {
    "left": "x=-{distance}mm",
    "right": "x={distance}mm",
    "up": "y={distance}mm",
    "down": "y=-{distance}mm",
    "forward": "z={distance}mm",
    "backward": "z=-{distance}mm",
}

VOICE_CONFIDENCE_THRESHOLD = 0.7

VOICE_COMMAND_TIMEOUT = 0.3

SYSTEM_PROMPT = "You are Qwen, a virtual assistant for endoscopic surgery, capable of analyzing images and understanding voice commands."

VISION_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
<|vision_bos|><|IMAGE|><|vision_eos|>

Analyze this endoscopic image and assess:

Fog: White hazy layer covering entire image. When level is severe, the whole view appears almost white; when none, the image just has a little white light reflections.

Contamination: Red/black spots or areas that clearly do not belong to real animal tissue.

1. Fog level: none/mild/moderate/severe
2. Contamination level: none/mild/moderate/severe
3. Image clarity score: 1-10


only output in JSON format: 
{{"fog_level": "none/mild/moderate/severe", "contamination_level": "none/mild/moderate/severe", "clarity_score": N}}<|im_end|>
<|im_start|>assistant
"""

AUDIO_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
<|audio_bos|><|AUDIO|><|audio_eos|>
You are listening to a surgeon giving movement commands to a surgical robot during an operation.

The audio contains Chinese or English speech. Listen carefully for direction words:

**Chinese directions:**
- 左 / 左边 / 往左 = LEFT
- 右 / 右边 / 往右 = RIGHT
- 上 / 上面 / 往上 = UP
- 下 / 下面 / 往下 = DOWN
- 前 / 深一点 / 进去 = FORWARD
- 后 / 退 / 浅一点 / 出来 = BACKWARD

Common phrases:
- "左一点" = move left a bit
- "往右移" = move right
- "上去一点" = move up
- "深入一点" = move forward

Distance modifiers:
- Default: 2mm
- If you hear "多一点/more/大" → 5mm
- If you hear "少一点/little/小" → 1mm

Your task:
If you hear ANY direction word → set has_command=true and specify the direction
If no direction word → has_command=false, direction="none"

Only Output in this JSON format:
{{"has_command": true/false, "direction": "left/right/up/down/forward/backward/none", "distance_mm": N}}<|im_end|>
<|im_start|>assistant
"""

SAMPLING_TEMPERATURE = 0.1
SAMPLING_MAX_TOKENS = 128