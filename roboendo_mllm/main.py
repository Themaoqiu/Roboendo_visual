import cv2
import json
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque
import re
import threading
from typing import Optional, Tuple, Dict
from PIL import Image

from vllm import LLM, SamplingParams

import config


fog_smooth = 0.0
contam_smooth = 0.0
frame_count = 0


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        pass
    
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match. group())
        except:
            pass
    
    return {}


def severity_to_score(severity: str) -> float:
    return config.SEVERITY_MAP. get(severity. lower(), 0.0)


def update_ema(old_val: float, new_val: float, alpha: float) -> float:
    return alpha * new_val + (1 - alpha) * old_val


class SimpleVAD:
    def __init__(self):
        self. is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_buffer = []
    
    def process(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        energy = np.mean(audio_chunk ** 2)
        
        if energy > config.VAD_ENERGY_THRESHOLD:
            self.silence_frames = 0
            self. speech_frames += 1
            
            if self.speech_frames >= config.VAD_MIN_SPEECH_FRAMES:
                self.is_speaking = True
            
            if self.is_speaking:
                self.speech_buffer. append(audio_chunk)
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_buffer.append(audio_chunk)
                
                if self. silence_frames >= config.VAD_MIN_SILENCE_FRAMES:
                    if len(self.speech_buffer) > 0:
                        utterance = np.concatenate(self.speech_buffer)
                        self._reset()
                        return utterance
            else:
                self.speech_frames = 0
        
        return None
    
    def _reset(self):
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_buffer = []


class AudioCapture:
    def __init__(self):
        self. mode = config.AUDIO_MODE
        self.sample_rate = config. AUDIO_SAMPLE_RATE
        self.vad = SimpleVAD()
        self.audio_queue = deque()
        self.lock = threading.Lock()
        
        self.file_audio = None
        self. file_position = 0
        self.stream = None
        
        if self.mode == "file" and config. AUDIO_SOURCE != "mic":
            self.file_audio, sr = sf.read(config. AUDIO_SOURCE)
            if sr != self.sample_rate:
                print(f"Warning: 音频文件采样率 {sr}Hz != 目标采样率 {self.sample_rate}Hz")
            print(f"加载音频文件: {config.AUDIO_SOURCE}, 时长: {len(self.file_audio)/sr:.1f}s")
    
    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        with self.lock:
            self. audio_queue.append(indata[:, 0]. copy())
    
    def start(self):
        if self. mode == "realtime":
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._callback,
                blocksize=int(self.sample_rate * config.CYCLE_PERIOD)
            )
            self.stream.start()
            print("音频采集已启动（实时模式）")
        else:
            print("音频采集已启动（文件模式）")
    
    def get_chunk(self) -> Optional[np.ndarray]:
        if self.mode == "realtime":
            with self.lock:
                if len(self.audio_queue) > 0:
                    return self.audio_queue.popleft()
            return None
        else:
            if self.file_audio is None:
                return None
            
            chunk_size = int(self.sample_rate * config.CYCLE_PERIOD)
            
            if self.file_position >= len(self.file_audio):
                self.file_position = 0
                print("[音频] 文件播放完毕，重新开始")
            
            end_pos = min(self.file_position + chunk_size, len(self.file_audio))
            chunk = self.file_audio[self.file_position:end_pos]
            
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            self.file_position = end_pos
            return chunk
    
    def stop(self):
        if self. mode == "realtime" and self.stream:
            self.stream.stop()
            self.stream.close()
            print("音频采集已停止")


def init_model() -> LLM:
    print(f"正在加载模型: {config.MODEL_NAME}")
    llm = LLM(
        model=config.MODEL_NAME,
        max_model_len=config.MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=config. GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt={"image": 1, "audio": 1}
    )
    print("模型加载完成")
    return llm


def analyze_image(llm: LLM, frame: np.ndarray) -> dict:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    prompt = config.VISION_PROMPT_TEMPLATE.format(system=config.SYSTEM_PROMPT)
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }
    
    sampling_params = SamplingParams(
        temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config. SAMPLING_MAX_TOKENS
    )
    
    try:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        text = outputs[0].outputs[0].text
        result = parse_json(text)
        return result
    except Exception as e:
        print(f"视觉推理错误: {e}")
        return {"fog_level": "none", "contamination_level": "none", "clarity_score": 5}


def analyze_audio(llm: LLM, audio: np.ndarray, sample_rate: int) -> dict:
    prompt = config.AUDIO_PROMPT_TEMPLATE.format(system=config.SYSTEM_PROMPT)
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"audio": (audio, sample_rate)}
    }
    
    sampling_params = SamplingParams(
        temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config. SAMPLING_MAX_TOKENS
    )
    
    try:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        text = outputs[0].outputs[0].text
        result = parse_json(text)
        return result
    except Exception as e:
        print(f"语音推理错误: {e}")
        return {"has_command": False, "direction": "none", "distance_mm": 0}


def parse_move_command(audio_result: dict) -> Optional[str]:
    direction = audio_result.get("direction", "none").lower()
    distance = audio_result. get("distance_mm", config. DEFAULT_MOVE_DISTANCE)
    
    if direction in config.DIRECTION_MAP:
        return config.DIRECTION_MAP[direction].format(distance=distance)
    
    return None


def generate_command(fog_smooth: float, contam_smooth: float) -> dict:
    command = {
        "suction": "no",
        "clean": "no",
        "move": "no"
    }
    
    need_clean = contam_smooth > config.CONTAMINATION_THRESHOLD
    need_suction = fog_smooth > config. FOG_THRESHOLD
    
    if need_clean:
        command["clean"] = "yes"
    elif need_suction:
        command["suction"] = "yes"
    
    return command


def main():
    global fog_smooth, contam_smooth, frame_count
    
    print("=" * 60)
    print("内窥镜手术机器人控制系统启动")
    print("=" * 60)
    
    llm = init_model()
    
    print(f"\n初始化视频采集: 模式={config.VIDEO_MODE}, 源={config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return

    video_fps = cap. get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"初始化音频采集: 模式={config.AUDIO_MODE}, 源={config. AUDIO_SOURCE}")
    audio_capture = AudioCapture()
    audio_capture.start()
    
    vad = SimpleVAD()
    
    print("\n系统就绪，开始主循环...")
    print("=" * 60)
    
    try:
        while True:
            cycle_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                if config.VIDEO_MODE == "file":
                    print("视频文件播放完毕")
                    break
                else:
                    print("警告: 读取帧失败")
                    continue
            
            frame_count += 1
            
            if frame_count % config.VISION_INFERENCE_INTERVAL == 0:
                vision_start = time.time()
                vision_result = analyze_image(llm, frame)
                vision_time = time.time() - vision_start
                
                fog_score = severity_to_score(vision_result. get("fog_level", "none"))
                contam_score = severity_to_score(vision_result.get("contamination_level", "none"))
                
                fog_smooth = update_ema(fog_smooth, fog_score, config.EMA_ALPHA)
                contam_smooth = update_ema(contam_smooth, contam_score, config.EMA_ALPHA)
                
                video_time = frame_count / video_fps
                print(f"[视觉@{video_time:.2f}s] fog={vision_result.get('fog_level')} ({fog_smooth:.2f}), "
                    f"contam={vision_result.get('contamination_level')} ({contam_smooth:.2f})")
                
                command = generate_command(fog_smooth, contam_smooth)
                print(f"[输出] {json.dumps(command)}")
                print("-" * 60)
            
            audio_chunk = audio_capture.get_chunk()
            if audio_chunk is not None:
                utterance = vad.process(audio_chunk)
                
                if utterance is not None:
                    print(f"[音频] 检测到完整语句，时长={len(utterance)/config.AUDIO_SAMPLE_RATE:.1f}s")
                    
                    audio_start = time.time()
                    audio_result = analyze_audio(llm, utterance, config.AUDIO_SAMPLE_RATE)
                    audio_time = time.time() - audio_start
                    
                    print(f"[音频] 解析结果: {audio_result}, 推理耗时={audio_time*1000:.1f}ms")
                    
                    if audio_result.get("has_command"):
                        voice_cmd = parse_move_command(audio_result)
                        command = {
                            "suction": "no",
                            "clean": "no",
                            "move": voice_cmd if voice_cmd else "no"
                        }
                        print(f"[输出] {json. dumps(command)}")
                        print("-" * 60)
            
            elapsed = time.time() - cycle_start
    
    except KeyboardInterrupt:
        print("\n\n收到中断信号")
    
    finally:
        print("\n清理资源...")
        cap.release()
        audio_capture.stop()
        print("系统已关闭")


if __name__ == "__main__":
    main()