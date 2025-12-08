"""main_plus.py - 实时手术器械检测系统
基于 main.py 改造，去掉视频保存，改为实时摄像头处理

保留功能：
1. YOLO目标检测
2. SAM2实例分割
3. 角度和比例计算（实时内存计算）
4. 突变点检测（滑动窗口）
5. 实时可视化（圆点 + 箭头 + 文字）

移除功能：
- 视频文件保存
- 图像帧保存
- JSON文件保存
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.ops import nms
from skimage.measure import ransac, LineModelND
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2.build_sam import build_sam2_camera_predictor

# =============================================================
# ObjectDetector 类（从 main.py 复制，保持不变）
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
        self.model = YOLO(YOLO_checkpoint, verbose=False)
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

        return {
            "bboxes": pred_boxes[best_idx].astype(np.int32).tolist(),
            "det_scores": scores.tolist()[best_idx],
            "labels": labels.tolist()[best_idx]
        }

    def _process_detection(self, frame: np.ndarray) -> dict:
        """Detect objects in a single frame using YOLOv8 and return the surgical instrument bbox."""
        pil_image = Image.fromarray(frame)
        result = self.model(pil_image, verbose=False)[0]
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

        return self.detection

# =============================================================
# 角度检测函数（从 main.py 的 Angle3.py 部分复制）
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

def compute_main(mask, min_length=28):
    """
    实时计算mask的角度和比例（从 main.py 复制）

    Returns:
    - angle: 开口角度
    - ratio: 长度比例
    - A: 上端点
    - B: 下端点
    - C: 交点
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask is None:
        return 0.0, 1.0, None, None, None

    w = mask.shape[1]
    upper_edges = []
    lower_edges = []
    segments_mode = False

    # 判断位置
    white_pixels = np.column_stack(np.where(mask == 255))
    if len(white_pixels) == 0:
        return 0.0, 1.0, None, None, None
    centroid_x = np.mean(white_pixels[:, 1])

    # 决定扫描方向
    if centroid_x <= w / 2:
        scan_order = range(w)
    else:
        scan_order = reversed(range(w))

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

    return angle, ratio, A, B, C

# =============================================================
# 突变点检测函数（从 main.py 的 plot4.py 部分复制）
# =============================================================
def compute_fft(signal, sample_rate):
    """计算FFT"""
    n = len(signal)
    fft_result = fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    magnitude = np.abs(fft_result)
    return freqs, magnitude

def find_points(angle, coverage, count, step_size, sigma=1.0):
    """找变大变小全局下标"""
    # 检查数据长度是否足够
    if len(angle) < 2 or len(coverage) < 2:
        return []

    open_deriv_smo = gaussian_filter1d(angle, sigma=sigma)
    area_deriv_smo = gaussian_filter1d(coverage, sigma=sigma)
    open_deriv = np.gradient(open_deriv_smo)
    area_deriv = np.gradient(area_deriv_smo)
    res = []
    for i in range(len(open_deriv)):
        threshold = 0.1
        if (area_deriv[i] < -threshold) or (area_deriv[i] > threshold) or (open_deriv[i] < -threshold) or (open_deriv[i] > threshold):
            win_start = step_size * count
            global_idx = win_start + i
            res.append(global_idx)
    return res

def find_fft_points(data, sigma=1.0):
    """计算窗口fft突变"""
    threshold = 6
    data = np.array(data)

    # 检查数据长度，至少需要2个元素才能计算梯度
    if len(data) < 2:
        return None

    smooth = gaussian_filter1d(data, sigma=sigma)
    fir_deriv = np.gradient(smooth)
    abs_deriv = np.abs(fir_deriv)
    max_index = np.argmax(abs_deriv)
    if abs_deriv[max_index] > threshold:
        return int(max_index)
    else:
        return None

def sliding_window(data, count, step_size, sample_rate):
    """获取fft突变全局下标"""
    # 检查数据长度是否足够
    if len(data) < 2:
        return []

    win_data, _ = compute_fft(data, sample_rate)
    local_idx = find_fft_points(win_data)
    win_start = step_size * count
    if local_idx is not None:
        global_idx = win_start + local_idx
        return [global_idx]
    else:
        return []

def find_main(open_angle, area_coverage, window_size, step_size, sample_rate):
    reverse_res = []
    mut_res = []
    count = 0
    for i in range(0, len(open_angle), step_size):
        angle = open_angle[i:i + sample_rate]
        coverage = area_coverage[i:i + sample_rate]
        mut_res = mut_res + sliding_window(angle, count, step_size, sample_rate)
        reverse_res = reverse_res + find_points(angle, coverage, count, step_size)
        count += 1
    return mut_res, reverse_res

# =============================================================
# 实时处理管道（基于 main.py 的架构）
# =============================================================
class RealtimeTrackingPipeline:
    def __init__(self, yolo_checkpoint: str, sam2_checkpoint: str, sam2_cfg: str,
                 window_size: int = 30, step_size: int = 15):
        """
        初始化实时跟踪管道

        Parameters:
        - yolo_checkpoint: YOLO模型路径
        - sam2_checkpoint: SAM2模型路径
        - sam2_cfg: SAM2配置文件
        - window_size: 突变检测窗口大小
        - step_size: 突变检测步长
        """
        # 初始化检测器和分割器
        self.detector = ObjectDetector(yolo_checkpoint)
        try:
            self.segmentor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint)
        except Exception as e:
            print(f"[错误] 初始化 SAM2 分割器时出错: {e}")
            print(f"配置文件: {sam2_cfg}")
            print(f"模型路径: {sam2_checkpoint}")
            import traceback
            traceback.print_exc()
            raise
        self.if_init = False

        # 保存配置用于重置
        self.sam2_cfg = sam2_cfg
        self.sam2_checkpoint = sam2_checkpoint

        # 突变检测参数
        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = 30

        # 实时数据存储（内存数组）
        self.angles = []
        self.ratios = []
        self.points_up = []
        self.points_low = []
        self.points_nodal = []

        # 当前帧索引
        self.frame_idx = 0

    def _apply_mask_visualization(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """改进的遮罩可视化（从 main.py 复制）"""
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
        if np.any(mask_bool):
            # 只有当存在mask像素时才处理
            try:
                mask_pixels = frame[mask_bool]
                green_overlay = np.full_like(mask_pixels, [0, 255, 0])
                blended = cv2.addWeighted(mask_pixels, 0.6, green_overlay, 0.4, 0)
                frame_with_mask[mask_bool] = blended
            except:
                # 如果出错，跳过这一步
                pass

        # 确保输出是uint8并且是连续内存
        frame_with_mask = np.ascontiguousarray(frame_with_mask, dtype=np.uint8)

        return frame_with_mask

    def _draw_annotations(self, frame: np.ndarray, angle, ratio, up, low, nodal,
                          reverse_frames_indices) -> np.ndarray:
        """
        绘制箭头和标注文字（整合 draw6.py 和 drawarrow5.py 的功能）

        Parameters:
        - frame: 输入帧
        - angle: 开口角度
        - ratio: 长度比例
        - up: 上端点A
        - low: 下端点B
        - nodal: 交点C
        - reverse_frames_indices: 突变点帧索引列表

        Returns:
        - 标注后的帧
        """
        height, width = frame.shape[:2]
        border_threshold = 300
        valid_x_range = (border_threshold, width - border_threshold)
        valid_y_range = (0, height)

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

        # 绘制绿色箭头（在突变点）
        if valid_up and self.frame_idx in reverse_frames_indices:
            # 计算画面中心点
            center_point = (width // 2, height // 2)
            low_point = tuple(map(int, up))
            arrow_color = (0, 255, 0)  # 绿色
            arrow_thickness = 10
            cv2.arrowedLine(frame, center_point, low_point, arrow_color, arrow_thickness)

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

        return frame

    def run(self, camera_id=1):
        """
        运行实时跟踪管道

        Parameters:
        - camera_id: 摄像头ID（0为默认摄像头，1为外接摄像头）
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        print(f"已打开摄像头 {camera_id}")
        print(f"按键说明：")
        print(f"  'q' - 退出程序")
        print(f"  'r' - 重新初始化")
        print(f"=" * 60)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"无法读取帧 {self.frame_idx}")
                    break

                # 确保frame是正确的格式
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

                # ========== 步骤1: YOLO检测 + SAM2跟踪 ==========
                if not self.if_init:
                    # 处理第一帧
                    detections = self.detector.process_frame(frame)

                    if detections is None:
                        cv2.imshow("Realtime Tracking", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    self.segmentor.load_first_frame(frame)
                    self.if_init = True

                    _, _, out_mask_logits = self.segmentor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,
                        bbox=np.array(detections["bboxes"], dtype=np.float32).reshape(2,2)
                    )
                else:
                    # 处理后续帧
                    _, out_mask_logits = self.segmentor.track(frame)

                # ========== 步骤2: 获取mask ==========
                mask = out_mask_logits.sigmoid().cpu().numpy()
                mask = mask[0, 0]
                mask = (mask > 0.5).astype(np.uint8) * 255

                # ========== 步骤3: 实时计算角度（内存计算，不存JSON）==========
                angle, ratio, A, B, C = compute_main(mask)

                # 存入数组
                self.angles.append(angle)
                self.ratios.append(ratio)
                self.points_up.append(A.tolist() if A is not None else None)
                self.points_low.append(B.tolist() if B is not None else None)
                self.points_nodal.append(C.tolist() if C is not None else None)

                # ========== 步骤4: 检测突变点 ==========
                reverse_frames_indices = []
                if len(self.angles) >= self.window_size:
                    mut_res, reverse_res = find_main(
                        self.angles,
                        self.ratios,
                        self.window_size,
                        self.step_size,
                        self.sample_rate
                    )
                    # 去重并排序
                    reverse_frames_indices = sorted(list(set([idx for idx in reverse_res if isinstance(idx, int)])))

                # ========== 步骤5: 绘制可视化 ==========
                # 先绘制mask
                frame_with_mask = self._apply_mask_visualization(frame, mask)

                # 再绘制标注（箭头 + 文字）
                display_frame = self._draw_annotations(
                    frame_with_mask,
                    angle,
                    ratio,
                    A,
                    B,
                    C,
                    reverse_frames_indices
                )

                # 添加状态提示
                cv2.putText(display_frame, "Press 'r' to manual reset | 'q' to quit",
                          (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 显示帧
                cv2.imshow("Realtime Tracking", display_frame)

                # 打印进度
                if self.frame_idx % 30 == 0:
                    print(f"已处理 {self.frame_idx} 帧 | 角度: {angle:.2f}° | 比例: {ratio:.2f}")

                self.frame_idx += 1

                # 按键控制（仅用于手动退出或重置）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\n[手动重置] 重新初始化跟踪器...")
                    print("清空历史数据，准备重新检测...")

                    # 重置所有状态
                    self.if_init = False
                    self.frame_idx = 0
                    self.angles.clear()
                    self.ratios.clear()
                    self.points_up.clear()
                    self.points_low.clear()
                    self.points_nodal.clear()

                    # 重新创建分割器（确保参数正确）
                    try:
                        # 释放旧的 segmentor（如果有的话）
                        if hasattr(self.segmentor, 'condition_state'):
                            self.segmentor.condition_state.clear()
                        
                        # 重新创建分割器（使用与初始化时相同的参数格式）
                        self.segmentor = build_sam2_camera_predictor(
                            self.sam2_cfg,
                            self.sam2_checkpoint
                        )
                        print("[手动重置] 完成！等待重新检测...\n")
                    except Exception as e:
                        print(f"[错误] 重置时出错: {e}")
                        print(f"配置文件: {self.sam2_cfg}")
                        print(f"模型路径: {self.sam2_checkpoint}")
                        print("请检查 SAM2 配置文件和模型路径是否正确")
                        import traceback
                        traceback.print_exc()

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"总帧数: {self.frame_idx}")
        if self.angles:
            print(f"平均角度: {np.mean(self.angles):.2f}°")
            print(f"平均比例: {np.mean(self.ratios):.2f}")
        print(f"{'='*60}\n")

# =============================================================
# 主函数
# =============================================================
def main():
    # 配置参数
    YOLO_CHECKPOINT = '/home/ubuntu/dcai/roboendo/models/yolov8x.pt'
    SAM2_CHECKPOINT = '/home/ubuntu/dcai/roboendo/models/sam2.1_hiera_large.pt'
    SAM2_CFG = "sam2.1_hiera_l.yaml"


    # 创建实时跟踪管道
    pipeline = RealtimeTrackingPipeline(
        yolo_checkpoint=YOLO_CHECKPOINT,
        sam2_checkpoint=SAM2_CHECKPOINT,
        sam2_cfg=SAM2_CFG,
        window_size=30,  # 可调整窗口大小
        step_size=15     # 可调整步长
    )

    # 运行（使用摄像头1）
    pipeline.run(camera_id="/home/ubuntu/dcai/roboendo/videos/test2.mp4")

if __name__ == "__main__":
    main()
