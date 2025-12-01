import argparse
import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.ops import nms
from skimage.measure import ransac, LineModelND

from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2.build_sam import build_sam2_camera_predictor
# =============================================================
# 从 detect1.py 整合的代码
# =============================================================
class ObjectDetector:
    def __init__(self, YOLO_checkpoint: str, score_thres: float = 0.05, iou_thres: float = 0.5):
        """
        Initialize ObjectDetector with YOLOv8 model and parameters for detection.
        
        Parameters:
        - YOLO_checkpoint: Path to the YOLOv8 checkpoint file
        - score_thres: Confidence threshold for keeping boxes
        - iou_thres: IoU threshold for non-maximal suppression
        """
        self.model = YOLO(YOLO_checkpoint)
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.frame_with_bboxes = None
        self.detection = None 
    
    def _calculate_aspect_ratio(self, bbox: np.ndarray) -> float:
        """计算边界框的长宽比"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(width, height) / (min(width, height) + 1e-6)
    
    def _is_from_edge(self, bbox: np.ndarray, frame_shape: tuple, edge_threshold: int = 50) -> tuple:
        """
        检查边界框是否从图像边缘伸入
        
        Returns:
        - (is_from_edge, edge_type): edge_type可以是'left', 'right', 'top', 'bottom'或None
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # 检查是否接触各个边缘
        from_left = x1 < edge_threshold
        from_right = x2 > (width - edge_threshold)
        from_top = y1 < edge_threshold
        from_bottom = y2 > (height - edge_threshold)
        
        edges = []
        if from_left: edges.append('left')
        if from_right: edges.append('right')
        if from_top: edges.append('top')
        if from_bottom: edges.append('bottom')
        
        return len(edges) > 0, edges

    def _get_surgical_instrument_bbox(self, pred_boxes: np.ndarray, scores: np.ndarray, 
                                      labels: np.ndarray, frame_shape: tuple) -> dict:
        """
        优先选择从侧边伸入的细长物体（手术刀特征）
        
        筛选策略：
        1. 从边缘伸入的物体
        2. 长宽比大于2.0的细长物体
        3. 置信度最高的候选
        
        Parameters:
        - pred_boxes: 边界框数组 [x1, y1, x2, y2]
        - scores: 置信度分数
        - labels: 标签
        - frame_shape: 帧的形状 (height, width)
        
        Returns:
        - 最佳手术器械的检测结果字典
        """
        if len(pred_boxes) == 0:
            return None
        
        # 计算每个bbox的特征
        candidates = []
        for i, bbox in enumerate(pred_boxes):
            aspect_ratio = self._calculate_aspect_ratio(bbox)
            is_from_edge, edge_types = self._is_from_edge(bbox, frame_shape, edge_threshold=100)
            
            # 评分系统
            score_weight = 0.0
            
            # 1. 从侧边（左或右）伸入 +3分
            if 'left' in edge_types or 'right' in edge_types:
                score_weight += 3.0
            # 从上下伸入 +1分（不太常见但也可能）
            elif 'top' in edge_types or 'bottom' in edge_types:
                score_weight += 1.0
            
            # 2. 细长物体（长宽比 > 2.0）+2分，比例越大分数越高
            if aspect_ratio > 2.0:
                score_weight += min(2.0 + (aspect_ratio - 2.0) * 0.3, 5.0)
            
            # 3. 检测置信度加权
            score_weight += scores[i] * 2.0
            
            # 4. 排除过大的物体（可能是整个手术台）
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_shape[0] * frame_shape[1]
            area_ratio = bbox_area / frame_area
            
            if area_ratio > 0.5:  # 如果物体占据超过50%的画面，可能不是手术刀
                score_weight -= 5.0
            
            candidates.append({
                'index': i,
                'score_weight': score_weight,
                'aspect_ratio': aspect_ratio,
                'is_from_edge': is_from_edge,
                'edge_types': edge_types,
                'confidence': scores[i],
                'area_ratio': area_ratio
            })
        
        # 按评分排序
        candidates.sort(key=lambda x: x['score_weight'], reverse=True)
        
        # 选择最佳候选
        best_candidate = candidates[0]
        best_idx = best_candidate['index']
        
        print(f"[Detection] Selected bbox - Aspect ratio: {best_candidate['aspect_ratio']:.2f}, "
              f"From edge: {best_candidate['edge_types']}, "
              f"Confidence: {best_candidate['confidence']:.2f}, "
              f"Score weight: {best_candidate['score_weight']:.2f}, "
              f"Area ratio: {best_candidate['area_ratio']:.2f}")
        
        return {
            "bboxes": pred_boxes[best_idx].astype(np.int32).tolist(),
            "det_scores": scores.tolist()[best_idx],
            "labels": labels.tolist()[best_idx]
        }

    def _process_detection(self, frame: np.ndarray) -> dict:
        """Detect objects in a single frame using YOLOv8 and return the surgical instrument bbox."""
        pil_image = Image.fromarray(frame)
        result = self.model(pil_image)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        # Apply Non-Maximal Suppression (NMS)
        keep_idx = nms(torch.tensor(pred_boxes), torch.tensor(scores), iou_threshold=self.iou_thres)
        pred_boxes, scores, labels = pred_boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        # Filter by confidence score threshold
        keep_idx = scores > self.score_thres
        pred_boxes, scores, labels = pred_boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        # 使用改进的手术器械检测策略
        instrument_detection = self._get_surgical_instrument_bbox(pred_boxes, scores, labels, frame.shape)
        
        return instrument_detection

    def _draw_bboxes_on_frame(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw bounding boxes on the frame."""
        if detections is None:
            return frame
            
        bbox, score, label = detections['bboxes'], detections['det_scores'], detections['labels']
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # Green box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = f"Instrument {label}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return the surgical instrument bounding box.
        
        Parameters:
        - frame: A single image/frame (numpy array)
        
        Returns:
        - detections: A dictionary containing the bounding box, label, and score
        """
        # Process the frame for object detection
        self.detection = self._process_detection(frame)
        
        if self.detection:
            print(f"Detection for the Surgical Instrument: {self.detection}")
            # Draw the bounding box on the frame
            self.frame_with_bboxes = self._draw_bboxes_on_frame(frame.copy(), self.detection)
        else:
            print("No surgical instrument detected!")
            self.frame_with_bboxes = frame.copy()
        
        return self.detection

class CameraTrackingPipeline:
    def __init__(self, camera, yolo_checkpoint: str, sam2_checkpoint: str, sam2_cfg: str):
        self.detector = ObjectDetector(yolo_checkpoint)
        self.segmentor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint)
        self.if_init = False
        self.cap = camera

    def _apply_mask_visualization(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """改进的遮罩可视化"""
        # 确保输入是uint8类型
        frame = frame.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        # 1. 绿色半透明叠加
        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = mask  # 绿色通道
        frame_with_mask = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        
        # 2. 绘制黄色轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 255), 3)
        
        # 3. 增强遮罩内部颜色
        mask_bool = mask > 127
        frame_with_mask[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 0.6,
            np.full_like(frame[mask_bool], [0, 255, 0]), 0.4, 0
        )
        
        # 确保输出是uint8并且是连续内存
        frame_with_mask = np.ascontiguousarray(frame_with_mask, dtype=np.uint8)
        
        return frame_with_mask

    def run(self, output_video_path: str, mask_output_dir: str):

        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)
        
        # 创建帧保存目录
        frames_output_dir = mask_output_dir.rstrip('/') + "_visualization"
        if not os.path.exists(frames_output_dir):
            os.makedirs(frames_output_dir)

        # 读取第一帧获取视频参数
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error: Cannot read video")
            return
        
        # 获取视频参数
        height, width = first_frame.shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 30.0
        
        print(f"Video info - Width: {width}, Height: {height}, FPS: {fps}")
        
        # 重置视频到开始位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"End of video at frame {frame_idx}")
                    break
                
                # 确保frame是正确的格式
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

                if not self.if_init:
                    # Process first frame
                    detections = self.detector.process_frame(frame)
                    
                    if detections is None:
                        print("No instrument detected in first frame, skipping initialization")
                        frame_idx += 1
                        continue
                    
                    self.segmentor.load_first_frame(frame)
                    self.if_init = True

                    _, _, out_mask_logits = self.segmentor.add_new_prompt(
                        frame_idx=0, 
                        obj_id=1, 
                        bbox=np.array(detections["bboxes"], dtype=np.float32).reshape(2,2)
                    )
                else:
                    # Process subsequent frames
                    _, out_mask_logits = self.segmentor.track(frame)

                # Apply the mask to the frame
                mask = out_mask_logits.sigmoid().cpu().numpy()
                mask = mask[0, 0]  # Assuming batch size is 1
                mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold the mask

                # 保存遮罩图像
                mask_filename = os.path.join(mask_output_dir, f"mask_{frame_idx:06d}.png")
                cv2.imwrite(mask_filename, mask)

                # 使用改进的可视化方法
                frame_with_mask = self._apply_mask_visualization(frame, mask)
                
                # 保存可视化帧（使用高质量JPEG）
                vis_filename = os.path.join(frames_output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(vis_filename, frame_with_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Print progress
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx} frames ({frame_idx/fps:.2f} seconds)")

            self.cap.release()
            
            print(f"\n{'='*60}")
            print(f"Total frames processed: {frame_idx}")
            print(f"Mask images saved to: {mask_output_dir}")
            print(f"Visualization frames saved to: {frames_output_dir}")
            print(f"{'='*60}\n")
            
            # 使用FFmpeg合成视频
            if frame_idx > 0:
                print("Creating video from frames using FFmpeg...")
                
                ffmpeg_cmd = (
                    f'ffmpeg -framerate {fps} '
                    f'-i {frames_output_dir}/frame_%06d.jpg '
                    f'-c:v libx264 -pix_fmt yuv420p -crf 18 '
                    f'-y {output_video_path}'
                )
                
                print(f"Running: {ffmpeg_cmd}\n")
                exit_code = os.system(ffmpeg_cmd)
                
                if exit_code == 0:
                    # 检查输出文件
                    if os.path.exists(output_video_path):
                        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
                        print(f"\n{'='*60}")
                        print(f"✓ Video created successfully!")
                        print(f"  Path: {output_video_path}")
                        print(f"  Size: {file_size:.2f} MB")
                        print(f"  Frames: {frame_idx}")
                        print(f"  Duration: {frame_idx/fps:.2f} seconds")
                        print(f"{'='*60}")
                    else:
                        print("Error: Video file was not created")
                else:
                    print(f"Error: FFmpeg failed with exit code {exit_code}")
                    print("You can manually create the video by running:")
                    print(ffmpeg_cmd)
            else:
                print("No frames processed, skipping video creation")

# =============================================================
# 从 mask2.py 整合的代码
# =============================================================
def median_blur_edges(image_path, kernel_size=9):
    """
    使用中值模糊平滑图像边缘
    :param image_path: 输入图像路径
    :param kernel_size: 模糊核大小，必须是大于1的奇数，例如 5
    :return: 平滑后的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像文件未找到或无法读取")

    blurred_image = cv2.medianBlur(img, kernel_size)
    return blurred_image

# =============================================================
# 从 Angle3.py 整合的代码
# =============================================================
def visualize_points(image, points, color=(0, 0, 255), thickness=1):
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), 3, color, thickness)

def find_intersection(line1, line2):
    p1, v1 = line1.params
    p2, v2 = line2.params
    a1 = v1[1]
    b1 = -v1[0]
    c1 = v1[0] * p1[1] - v1[1] * p1[0]
    a2 = v2[1]
    b2 = -v2[0]
    c2 = v2[0] * p2[1] - v2[1] * p2[0]
    D = a1 * b2 - a2 * b1
    if D == 0:
        return None
    x = (b1 * c2 - b2 * c1) / D
    y = (a2 * c1 - a1 * c2) / D
    return np.array([x, y])

def find_tip_point(points, C_point):
    if len(points) == 0:
        return None
    C_array = np.array(C_point)
    distances = np.linalg.norm(points - C_array, axis=1)
    max_idx = np.argmax(distances)
    return points[max_idx]

def draw_line(img, modeline, color, length=1000):
    v = modeline.params[1]
    t = np.linspace(-length, length, 2)
    line_points = modeline.params[0] + t[:, np.newaxis] * v
    cv2.line(img, tuple(line_points[0].astype(int)),tuple(line_points[1].astype(int)),color, 1)

def compute_main(mask_path, visualize=False, min_length=28):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Image not found")

    h, w = mask.shape[:2]  # 修改这行，添加[:2]确保只取前两个维度
    upper_edges = []
    lower_edges = []
    segments_mode = False

    # 判断位置
    white_pixels = np.column_stack(np.where(mask == 255))
    if len(white_pixels) == 0:
        return 0.0, 1.0, None, None, None
    centroid_x = np.mean(white_pixels[:, 1])
    # 决定扫描方向并输出
    if centroid_x <= w / 2:
        scan_direction = "left-to-right"
        scan_order = range(w)
    else:
        scan_direction = "right-to-left"
        scan_order = reversed(range(w))
    print(f"[compute_main] scan direction: {scan_direction}")

    for x in scan_order:
        column = mask[:, x]
        white_pixels = np.where(column == 255)[0]
        if len(white_pixels) == 0:
            continue
        diffs = np.diff(white_pixels)
        breaks = np.where(diffs > 1)[0] + 1
        segments_indices = np.split(white_pixels, breaks)
        segments = []

        for seg in segments_indices:
            if len(seg) >= min_length:
                start_y = seg[0]
                end_y = seg[-1]
                segments.append((start_y, end_y))
        if len(segments) == 2:
            segments_mode = True
        if segments_mode and len(segments) >= 1:
            upper_edges.append((x, segments[0][1]))
            try:
                lower_edges.append((x, segments[1][0]))
            except IndexError:
                pass

    if not upper_edges or not lower_edges:
        return 0.0, 1.0, None, None, None

    upper_edges_np = np.array(upper_edges)
    lower_edges_np = np.array(lower_edges)

    try:
        model_upper, _ = ransac(upper_edges_np, LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
        model_lower, _ = ransac(lower_edges_np, LineModelND, min_samples=2, residual_threshold=1, max_trials=100)

    except ValueError:
        return 0.0, 1.0, None, None, None

    C = find_intersection(model_upper, model_lower)
    if C is None:
        return 0.0, 1.0, None, None, None

    A = find_tip_point(upper_edges_np, C)
    B = find_tip_point(lower_edges_np, C)
    if A is None or B is None:
        return 0.0, 1.0, A, B, C

    CA = np.array(A) - np.array(C)
    CB = np.array(B) - np.array(C)
    if np.linalg.norm(CA) == 0 or np.linalg.norm(CB) == 0:
        return 0.0, 1.0, A, B, C

    cosine = np.dot(CA, CB) / (np.linalg.norm(CA) * np.linalg.norm(CB))
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    ratio = np.linalg.norm(CB) / np.linalg.norm(CA) if np.linalg.norm(CA) != 0 else 1.0

    if visualize:
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        visualize_points(vis_img, upper_edges_np, (0, 255, 0))
        visualize_points(vis_img, lower_edges_np, (255, 0, 0))

        draw_line(vis_img, model_upper, (0, 180, 0))
        draw_line(vis_img, model_lower, (0, 0, 180))

        cv2.circle(vis_img, tuple(np.array(C).astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(vis_img, tuple(A.astype(int)), 5, (0, 255, 255), -1)
        cv2.circle(vis_img, tuple(B.astype(int)), 5, (255, 255, 0), -1)
        
        cv2.imshow("Visualization", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return angle, ratio, A, B, C

# =============================================================
# 从 plot4.py 整合的代码
# =============================================================
def compute_fft(signal, sample_rate):
    """
    返回：
    freqs : ndarray
        FFT结果
    magnitude : ndarray
        FFT结果的幅度值
    """
    n = len(signal)
    fft_result = fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    magnitude = np.abs(fft_result)
    return freqs, magnitude

# 找变大变小全局下标
def find_points(angle, coverage, count, sigma = 1.0):
    # 条件1: 覆盖面积变小 (导数 < 0)
    # 条件2: 开口角度变小 (导数 < 0)
    # 条件3: 变化幅度超过噪声阈值
    open_deriv_smo, area_deriv_smo = gaussian_filter1d(angle, sigma=sigma), gaussian_filter1d(coverage, sigma=sigma)
    open_deriv, area_deriv = np.gradient(open_deriv_smo), np.gradient(area_deriv_smo)
    res = []
    for i in range(len(open_deriv)):
        threshold = 0.1
        if (area_deriv[i] < -threshold) or (area_deriv[i] > threshold) or (open_deriv[i] < -threshold) or (open_deriv[i] > threshold):
            win_start = step_size * count
            global_idx = win_start + i
            res.append(global_idx)
    return res
    
# 计算窗口fft突变
def find_fft_points(data, sigma=1.0):
    threshold = 6 # 亲测阈值为7会全部消失，6会少最后一个，似乎没什么用
    data = np.array(data)
    smooth = gaussian_filter1d(data, sigma=sigma)
    fir_deriv = np.gradient(smooth)
    abs_deriv = np.abs(fir_deriv)
    max_index = np.argmax(abs_deriv)
    if abs_deriv[max_index] > threshold:
        return int(max_index)
    else:
        return

# 获取ftt突变全局下标
def sliding_window(data, count):
    win_data, _ = compute_fft(data, sample_rate)
    local_idx = find_fft_points(win_data)
    win_start = step_size * count
    win_end = win_start + window_size
    if local_idx != None:
        global_idx = win_start + local_idx
    else:
        return []
    return [global_idx]

# 输入全部角度和覆盖面积数据
def find_main(open_angle, area_coverage):
    reverse_res = []
    mut_res = []
    count = 0
    for i in range(0, len(open_angle), step_size):
        angle = open_angle[i:i + sample_rate]
        coverage = area_coverage[i:i + sample_rate]
        mut_res = mut_res + sliding_window(angle, count)
        reverse_res = reverse_res + find_points(angle, coverage, count)
        count += 1
    return mut_res, reverse_res

# =============================================================
# 从 drawarrow5.py 整合的代码
# =============================================================
def annotate_video_with_arrow(input_video_path, results_json_path, points_json_path, output_video_path):
    # 加载 results.json 文件获取 low 坐标
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results_json_data = json.load(f)
    low_list = results_json_data.get("up", [])

    # 加载 points.json 文件获取 reverse_res 帧索引
    with open(points_json_path, 'r', encoding='utf-8') as f:
        points_json_data = json.load(f)
    reverse_res = points_json_data.get("reverse_res", [])
    # 将 reverse_res 展平为一维列表，并去重
    reverse_frames_indices = sorted(list(set([idx for idx in reverse_res if isinstance(idx, int)])))


    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开输入视频文件")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    border_threshold = 300  # 距离边界小于5像素视为异常值
    valid_x_range = (border_threshold, width - border_threshold)
    valid_y_range = (0, height)
    frame_index = 0

    def valid_point(point):
        if point is None:
            return False
        x, y = map(int, point)
        return (valid_x_range[0] < x < valid_x_range[1] and valid_y_range[0] < y < valid_y_range[1])
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的 low 坐标
        up = low_list[frame_index] if frame_index < len(low_list) else None
        if valid_point(up):
            # 检查当前帧是否在 reverse_res 列表中
            if frame_index in reverse_frames_indices:
                # 计算画面中心点
                center_point = (width // 2, height // 2)

                # 绘制绿色箭头
                if up is not None:
                    low_point = tuple(map(int, up))
                    arrow_color = (0, 255, 0)  # 绿色
                    arrow_thickness = 10  # 箭头粗细

                    cv2.arrowedLine(frame, center_point, low_point, arrow_color, arrow_thickness)

        out.write(frame)
        print(f"处理进度: {frame_index+1}/{total_frames} 帧", end='\r')
        frame_index += 1

    cap.release()
    out.release()
    print(f"\n视频处理完成,保存至: {os.path.abspath(output_video_path)}")

# =============================================================
# 从 draw6.py 整合的代码
# =============================================================
def annotate_video_with_json(input_video_path, json_path, output_video_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    angle_list = json_data.get("opening_angle", [])
    ratio_list = json_data.get("block_ratio", [])
    up_list = json_data.get("up", [])
    low_list = json_data.get("low", [])
    nodal_list = json_data.get("nodal", [])

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开输入视频文件")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0
    border_threshold = 300  # 距离边界小于5像素视为异常值
    valid_x_range = (border_threshold, width - border_threshold)
    valid_y_range = (0, height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        angle = angle_list[frame_index] if frame_index < len(angle_list) else None
        ratio = ratio_list[frame_index] if frame_index < len(ratio_list) else 0
        up = up_list[frame_index] if frame_index < len(up_list) else None
        low = low_list[frame_index] if frame_index < len(low_list) else None
        nodal = nodal_list[frame_index] if frame_index < len(nodal_list) else None

        def valid_point(point):
            if point is None:
                return False
            x, y = map(int, point)
            return (valid_x_range[0] < x < valid_x_range[1] and valid_y_range[0] < y < valid_y_range[1])

        # 检查各点有效性
        valid_up = valid_point(up)
        valid_low = valid_point(low)
        valid_nodal = valid_point(nodal)

        point_radius = 10
        colors = {
            'up': (0, 255, 0),    # 绿色
            'low': (255, 0, 0),   # 蓝色
            'nodal': (0, 0, 255)  # 红色
        }
        # 绘制上端点（仅当有效时）
        if valid_up:
            cv2.circle(frame, tuple(map(int, up)), point_radius, colors['up'], -1)
        
        # 绘制下端点（仅当有效时）
        if valid_low:
            cv2.circle(frame, tuple(map(int, low)), point_radius, colors['low'], -1)
        
        # 绘制交点（仅当有效时）
        if valid_nodal:
            cv2.circle(frame, tuple(map(int, nodal)), point_radius, colors['nodal'], -1)
        
        # 创建显示文本
        angle_text = f"Angle: {angle:.2f}" if angle is not None else "Angle: N/A"
        ratio_text = f"Block Ratio: {ratio:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3
        color = (0, 255, 0)
        y_offset = 100

        cv2.putText(frame, angle_text, (50, y_offset), font, font_scale, color, font_thickness)
        cv2.putText(frame, ratio_text, (50, y_offset + 60), font, font_scale, color, font_thickness)
        out.write(frame)
        print(f"处理进度: {frame_index+1}/{total_frames} 帧", end='\r')
        frame_index += 1
    cap.release()
    out.release()
    print(f"\n视频处理完成,保存至: {os.path.abspath(output_video_path)}")

# =============================================================
# VideoProcessingPipeline 类和 main 函数（修改后的版本）
# =============================================================
class VideoProcessingPipeline:
    def __init__(self, input_video_path, output_dir=None):
        self.input_video_path = input_video_path
        
        # 确定输出目录
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(input_video_path), "output")
        else:
            self.output_dir = output_dir
        
        # 创建必要的输出目录
        self.mask_dir = os.path.join(self.output_dir, "mask_frames")
        self.middle_dir = os.path.join(self.output_dir, "middle_blur")
        self.result_dir = os.path.join(self.output_dir, "results")
        
        # 确保目录存在
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.middle_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 输出文件路径
        self.output_video_path = os.path.join(self.output_dir, "out.mp4")
        self.results_json_path = os.path.join(self.result_dir, "results.json")
        self.points_json_path = os.path.join(self.result_dir, "points.json")
        self.arrow_video_path = os.path.join(self.output_dir, "arrow.mp4")
        self.final_video_path = os.path.join(self.output_dir, "final_annotated_video.mp4")
        
        # SAM2和YOLO配置
        self.yolo_checkpoint = 'E:\\segment\\yolov8.pt'
        self.sam2_checkpoint = 'E:\\segment\\segment2\\checkpoints\\sam2.1_hiera_large.pt'
        self.sam2_cfg = "sam2.1_hiera_l.yaml"
        
        # 平滑参数
        self.median_blur_kernel = 9
        
        # 突变点检测参数
        self.window_size = 30
        self.step_size = 15

    def run_detection_segmentation(self):
        """步骤1: 执行视频跟踪与分割"""
        print("\n----------第一步: 视频跟踪与分割---------------")
        try:
            # 创建视频捕获对象
            cap = cv2.VideoCapture(self.input_video_path)
            
            # 创建管道对象
            pipe = CameraTrackingPipeline(
                camera=cap,
                yolo_checkpoint=self.yolo_checkpoint,
                sam2_cfg=self.sam2_cfg,
                sam2_checkpoint=self.sam2_checkpoint
            )
            
            # 运行管道
            pipe.run(
                output_video_path=self.output_video_path,
                mask_output_dir=self.mask_dir
            )
            
            print("步骤1完成: 视频跟踪与分割已完成")
            return True
        except Exception as e:
            print(f"步骤1出错: {e}")
            return False
    
    def run_edge_smoothing(self):
        """步骤2: 执行平滑mask边缘"""
        print("\n-----------第二步: 平滑mask边缘-----------")
        try:
            # 处理所有mask图像
            image_files = [f for f in os.listdir(self.mask_dir) if f.endswith(("jpg", "JPG", "png", "PNG"))]
            count = 0
            
            for image_file in image_files:
                image_path = os.path.join(self.mask_dir, image_file)
                smoothed_image = median_blur_edges(image_path, self.median_blur_kernel)
                res_name = f"smooth_{count:04d}.png"
                res_file = os.path.join(self.middle_dir, res_name)
                cv2.imwrite(res_file, smoothed_image)
                print(f"保存帧: {count}, 路径: {res_file}", end='\r')
                count += 1
            
            print("\n步骤2完成: 平滑mask边缘已完成")
            return True
        except Exception as e:
            print(f"步骤2出错: {e}")
            return False
    
    def run_angle_detection(self):
        """步骤3: 执行检测角度"""
        print("\n-----------第三步: 检测角度-----------")
        try:
            # 批量处理图像
            data = {
                "opening_angle": [],
                "block_ratio": [],
                "up": [],
                "low": [],
                "nodal": []
            }
            
            image_files = [f for f in os.listdir(self.middle_dir) if f.endswith((".jpg", ".JPG", ".png", ".PNG"))]
            count = 0
            
            for image_file in image_files:
                image_path = os.path.join(self.middle_dir, image_file)
                angle, ratio, A, B, C = compute_main(image_path, False)
                data["opening_angle"].append(round(angle, 2))
                data["block_ratio"].append(round(ratio, 2))
                data["up"].append(A.tolist() if A is not None else None)
                data["low"].append(B.tolist() if B is not None else None)
                data["nodal"].append(C.tolist() if C is not None else None)
                print(f"{image_file}: 角度={angle:.2f}° 比例={ratio:.2f}", end="\r")
                count += 1
            
            # 保存结果
            with open(self.results_json_path, "w") as f:
                json.dump(data, f)
            
            print("\n步骤3完成: 检测角度已完成")
            return True
        except Exception as e:
            print(f"步骤3出错: {e}")
            return False
    
    def run_change_detection(self):
        """步骤4: 执行检测突变点"""
        print("\n-----------第四步: 检测突变点-----------")
        try:
            # 读取结果文件
            with open(self.results_json_path, 'r') as json_file:
                data = json.load(json_file)
            
            # 创建时间序列
            t = np.arange(len(data['opening_angle']))
            open_angle = np.array(data['opening_angle'])
            area_coverage = np.array(data['block_ratio'])
            
            # 设置全局变量，供plot4.py中的函数使用
            global window_size, step_size, sample_rate
            sample_rate = 30
            window_size = self.window_size
            step_size = self.step_size
            
            # 检测突变点
            mut_res, reverse_res = find_main(open_angle, area_coverage)
            
            # 保存突变点结果
            points_data = {
                "mut_res": mut_res,
                "reverse_res": reverse_res
            }
            with open(self.points_json_path, "w") as f:
                json.dump(points_data, f, indent=4)
            
            print("步骤4完成: 检测突变点已完成")
            return True
        except Exception as e:
            print(f"步骤4出错: {e}")
            return False
    
    def run_arrow_annotation(self):
        """步骤5: 执行绘制箭头"""
        print("\n-----------第五步: 绘制箭头-----------")
        try:
            # 执行箭头标注
            annotate_video_with_arrow(
                self.input_video_path,
                self.results_json_path,
                self.points_json_path,
                self.arrow_video_path
            )
            
            print("步骤5完成: 绘制箭头已完成")
            return True
        except Exception as e:
            print(f"步骤5出错: {e}")
            return False
    
    def run_text_annotation(self):
        """步骤6: 执行绘制文字"""
        print("\n-----------第六步: 绘制文字-----------")
        try:
            # 执行文字标注
            annotate_video_with_json(
                self.arrow_video_path,
                self.results_json_path,
                self.final_video_path
            )
            
            print("步骤6完成: 绘制文字已完成")
            return True
        except Exception as e:
            print(f"步骤6出错: {e}")
            return False
    
    def run(self):
        """运行完整的处理管道"""
        print(f"开始处理视频: {self.input_video_path}")
        print(f"输出目录: {self.output_dir}")
        
        # 按顺序执行各个步骤
        steps = [
            self.run_detection_segmentation,
            self.run_edge_smoothing,
            self.run_angle_detection,
            self.run_change_detection,
            self.run_arrow_annotation,
            self.run_text_annotation
        ]
        
        for i, step in enumerate(steps):
            if not step():
                print(f"处理流程在第{i+1}步失败")
                return False
        
        print("\n所有步骤已完成! 处理管道执行成功。")
        print(f"最终输出视频: {self.final_video_path}")
        return True

def main():
    # 创建一个模拟的args对象，包含默认值
    class Args:
        def __init__(self):
            # 设置默认的视频路径（请根据实际情况修改）
            self.input_video = "E:\\segment\\segment2\\video\\test4.mp4"
            # 设置默认的输出目录（请根据实际情况修改）
            self.output_dir = "E:\\segment\\segment2\\video\\output"
            # 设置默认的参数值
            self.window_size = 30
            self.step_size = 15
    
    args = Args()
    
    # 检查输入视频文件是否存在
    if not os.path.exists(args.input_video):
        print(f"错误: 视频文件 '{args.input_video}' 不存在")
        # 添加一个输入框让用户可以手动选择视频文件
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov")]
        )
        if file_path:
            args.input_video = file_path
        else:
            print("未选择视频文件，程序退出")
            return
    
    # 如果输出目录不存在，则创建
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建处理管道对象
    pipeline = VideoProcessingPipeline(args.input_video, args.output_dir)
    
    # 设置参数
    if args.window_size:
        pipeline.window_size = args.window_size
    if args.step_size:
        pipeline.step_size = args.step_size
    
    # 运行处理管道
    success = pipeline.run()
    
    if success:
        print("\n程序执行成功完成!")
        # 让窗口保持打开状态，以便用户查看结果
        input("按Enter键退出...")
    else:
        print("\n程序执行过程中出现错误!")
        input("按Enter键退出...")

# 设置全局变量，供plot4.py中的函数使用
global window_size, step_size, sample_rate
sample_rate = 30
window_size = 30
step_size = 15

if __name__ == "__main__":
    main()