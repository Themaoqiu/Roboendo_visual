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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2.build_sam import build_sam2_camera_predictor

# =============================================================
# ObjectDetector 类
# =============================================================
class ObjectDetector:
    def __init__(self, YOLO_checkpoint: str, score_thres: float = 0.05, iou_thres: float = 0.5):
        self.model = YOLO(YOLO_checkpoint, verbose=False)
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.frame_with_bboxes = None
        self.detection = None

    def _calculate_aspect_ratio(self, bbox: np.ndarray) -> float:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(width, height) / (min(width, height) + 1e-6)

    def _is_from_edge(self, bbox: np.ndarray, frame_shape: tuple, edge_threshold: int = 50) -> tuple:
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]

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

    def _get_surgical_instrument_bbox(self, pred_boxes: np.ndarray, scores: np.ndarray,labels: np.ndarray, frame_shape: tuple) -> dict:
        if len(pred_boxes) == 0:
            return None

        candidates = []
        for i, bbox in enumerate(pred_boxes):
            aspect_ratio = self._calculate_aspect_ratio(bbox)
            is_from_edge, edge_types = self._is_from_edge(bbox, frame_shape, edge_threshold=100)

            score_weight = 0.0

            if 'left' in edge_types or 'right' in edge_types:
                score_weight += 3.0
            elif 'top' in edge_types or 'bottom' in edge_types:
                score_weight += 1.0

            if aspect_ratio > 2.0:
                score_weight += min(2.0 + (aspect_ratio - 2.0) * 0.3, 5.0)

            score_weight += scores[i] * 2.0

            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_shape[0] * frame_shape[1]
            area_ratio = bbox_area / frame_area

            if area_ratio > 0.5:
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

        candidates.sort(key=lambda x: x['score_weight'], reverse=True)

        best_candidate = candidates[0]
        best_idx = best_candidate['index']

        return {
            "bboxes": pred_boxes[best_idx].astype(np.int32).tolist(),
            "det_scores": scores.tolist()[best_idx],
            "labels": labels.tolist()[best_idx]
        }

    def _process_detection(self, frame: np.ndarray) -> dict:
        pil_image = Image.fromarray(frame)
        result = self.model(pil_image, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        keep_idx = nms(torch.tensor(pred_boxes), torch.tensor(scores), iou_threshold=self.iou_thres)
        pred_boxes, scores, labels = pred_boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        keep_idx = scores > self.score_thres
        pred_boxes, scores, labels = pred_boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        instrument_detection = self._get_surgical_instrument_bbox(pred_boxes, scores, labels, frame.shape)

        return instrument_detection

    def process_frame(self, frame: np.ndarray) -> dict:
        self.detection = self._process_detection(frame)

        return self.detection

# =============================================================
# 角度检测函数
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
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask is None:
        return 0.0, 1.0, None, None, None

    w = mask.shape[1]
    upper_edges = []
    lower_edges = []
    segments_mode = False

    white_pixels = np.column_stack(np.where(mask == 255))
    if len(white_pixels) == 0:
        return 0.0, 1.0, None, None, None
    centroid_x = np.mean(white_pixels[:, 1])

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
# 突变点检测函数
# =============================================================
def compute_fft(signal, sample_rate):
    n = len(signal)
    fft_result = fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    magnitude = np.abs(fft_result)
    return freqs, magnitude

def find_points(angle, coverage, count, step_size, sigma=1.0):
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
    threshold = 6
    data = np.array(data)

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
# 镜头移动控制函数
# =============================================================
def compute_camera_movement(frame_center: tuple, tip_point: tuple, sensitivity: float = 0.5) -> dict:
    if frame_center is None or tip_point is None:
        return {"dx": 0, "dy": 0}
    dx = int((tip_point[0] - frame_center[0]) * sensitivity)
    dy = int((frame_center[1] - tip_point[1]) * sensitivity)
    return {"dx": dx, "dy": dy}

def compute_mask_length_ratio(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    
    frame_h, frame_w = mask.shape[:2]
    points = np.column_stack(np.where(mask > 0))
    if len(points) < 2:
        return 0.0
    
    # 使用最小外接矩形来计算 mask 的整体长度
    # points 格式: (row, col) -> 转换为 (x, y) 格式给 cv2
    points_xy = points[:, ::-1].astype(np.float32)  # (col, row) = (x, y)
    rect = cv2.minAreaRect(points_xy)
    (cx, cy), (w, h), angle = rect
    
    mask_length = max(w, h)

    ratio = mask_length / frame_w if frame_w > 0 else 0.0

    return ratio

def compute_depth_estimation(handle_length_ratio: float, 
                             target_ratio: float = 0.5,
                             z_threshold: float = 0.1,
                             max_depth_step: int = 50) -> dict:
    if handle_length_ratio is None or handle_length_ratio <= 0:
        return {
            'dz': 0,
            'ratio_error': 0,
            'action': '保持',
            'confidence': 0.0
        }
    
    ratio_error = handle_length_ratio - target_ratio
    
    dz = -int(ratio_error * max_depth_step)
    
    if ratio_error > z_threshold:
        action = '后退'
        confidence = min(abs(ratio_error) / 0.4, 1.0)
    elif ratio_error < -z_threshold:
        action = '前进'
        confidence = min(abs(ratio_error) / 0.4, 1.0)
    else:
        action = '保持'
        dz = 0
        confidence = 1.0
    
    if dz <= 10:
        dz = 0.0

    return {
        'dz': dz,
        'ratio_error': float(ratio_error),
        'action': action,
        'confidence': float(confidence)
    }


class CommandStabilizer:
    def __init__(self,
                 ema_alpha: float = 0.2,
                 deadband_xy: float = 5,
                 deadband_z: float = 5,
                 max_step_xy: int = 20,
                 max_step_z: int = 10,
                 cooldown_frames: int = 10,
                 trigger_on_events: bool = True,
                 accumulate_threshold_xy: float = 25,
                 accumulate_threshold_z: float = 15):
        self.ema_alpha = ema_alpha
        self.deadband_xy = deadband_xy
        self.deadband_z = deadband_z
        self.max_step_xy = max_step_xy
        self.max_step_z = max_step_z
        self.cooldown_frames = cooldown_frames
        self.trigger_on_events = trigger_on_events
        self.accumulate_threshold_xy = accumulate_threshold_xy
        self.accumulate_threshold_z = accumulate_threshold_z

        self._ema = np.zeros(3, dtype=np.float32)
        self._last_output = np.zeros(3, dtype=np.float32)
        self._accum = np.zeros(3, dtype=np.float32)
        self._last_emit_frame = -9999

    def _apply_ema(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.array(xyz, dtype=np.float32)
        self._ema = self.ema_alpha * xyz + (1.0 - self.ema_alpha) * self._ema
        return self._ema

    def _apply_deadband(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz
        if abs(x) < self.deadband_xy:
            x = 0.0
        if abs(y) < self.deadband_xy:
            y = 0.0
        if abs(z) < self.deadband_z:
            z = 0.0
        return np.array([x, y, z], dtype=np.float32)

    def _limit_step(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        delta = curr - prev
        delta_xy = np.clip(delta[:2], -self.max_step_xy, self.max_step_xy)
        delta_z = np.clip(delta[2], -self.max_step_z, self.max_step_z)
        return prev + np.array([delta_xy[0], delta_xy[1], delta_z], dtype=np.float32)

    def update(self, frame_idx: int, xyz: list[float], is_event: bool = False) -> tuple[bool, list[int]]:
        smoothed = self._apply_ema(np.array(xyz, dtype=np.float32))
        smoothed = self._apply_deadband(smoothed)
        limited = self._limit_step(self._last_output, smoothed)

        delta_output = limited - self._last_output
        self._accum += np.abs(delta_output)
        accum_xy = self._accum[0] + self._accum[1]
        accum_z = self._accum[2]

        in_cooldown = (self._last_emit_frame != -9999) and ((frame_idx - self._last_emit_frame) < self.cooldown_frames)

        should_emit = False

        if in_cooldown:
            should_emit = False
        elif is_event and self.trigger_on_events:
            should_emit = True
        elif accum_xy >= self.accumulate_threshold_xy or accum_z >= self.accumulate_threshold_z:
            should_emit = True

        if should_emit:
            self._last_emit_frame = frame_idx
            self._accum[:] = 0.0
            self._last_output = limited
            emit_xyz = [int(round(limited[0])), int(round(limited[1])), int(round(limited[2]))]
            return True, emit_xyz

        return False, [0, 0, 0]

# =============================================================
# 实时处理管道
# =============================================================
class RealtimeTrackingPipeline:
    def __init__(self, yolo_checkpoint: str, sam2_checkpoint: str, sam2_cfg: str,
                 window_size: int = 30, step_size: int = 15):
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

        self.sam2_cfg = sam2_cfg
        self.sam2_checkpoint = sam2_checkpoint

        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = 30

        self.angles = []
        self.ratios = []
        self.points_up = []
        self.points_low = []
        self.points_nodal = []

        self.frame_idx = 0
        
        self.movement_data = []
        self.stabilizer = CommandStabilizer(
            ema_alpha=0.2,
            deadband_xy=5,
            deadband_z=3,
            max_step_xy=20,
            max_step_z=10,
            cooldown_frames=30,
            trigger_on_events=True,
            accumulate_threshold_xy=25,
            accumulate_threshold_z=15
        )

    def _apply_mask_visualization(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.uint8)
        mask = mask.astype(np.uint8)

        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = mask 
        frame_with_mask = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 255), 3)

        mask_bool = mask > 127
        if np.any(mask_bool):
            try:
                mask_pixels = frame[mask_bool]
                green_overlay = np.full_like(mask_pixels, [0, 255, 0])
                blended = cv2.addWeighted(mask_pixels, 0.6, green_overlay, 0.4, 0)
                frame_with_mask[mask_bool] = blended
            except:
                pass

        frame_with_mask = np.ascontiguousarray(frame_with_mask, dtype=np.uint8)

        return frame_with_mask

    def _compute_camera_control(self, frame: np.ndarray, up: tuple, mask: np.ndarray) -> dict:
        height, width = frame.shape[:2]
        frame_center = (width // 2, height // 2)
        
        xy_movement = compute_camera_movement(frame_center, up, sensitivity=1.0)
        
        mask_length_ratio = compute_mask_length_ratio(mask)
        z_movement = compute_depth_estimation(mask_length_ratio, target_ratio=0.5, max_depth_step=50)
        
        dx, dy = xy_movement['dx'], xy_movement['dy']
        dz = z_movement['dz']
        
        camera_command = f"{dx} {dy} {dz}"
        
        return {
            'x_y_movement': xy_movement,
            'z_movement': z_movement,
            'camera_command': camera_command,
            'xyz_array': [dx, dy, dz]
        }

    def _save_movement_data(self, output_file: str = None):
        if output_file is None:
            output_file = "movement_data.txt"
        
        try:
            with open(output_file, 'w') as f:
                f.write("# 镜头移动数据\n")
                f.write("# 格式: frame_idx dx dy dz\n")
                f.write("# dx, dy, dz 单位：像素 / 相对单位\n")
                f.write("=" * 50 + "\n")
                
                for data in self.movement_data:
                    f.write(f"{data[0]} {data[1]} {data[2]} {data[3]}\n")
            
            print(f"\n[保存成功] 镜头移动数据已保存到: {output_file}")
            print(f"[总数据行数] {len(self.movement_data)} 帧")
        except Exception as e:
            print(f"\n[保存失败] 无法保存到 {output_file}: {e}")

    def _draw_annotations(self, frame: np.ndarray, angle, ratio, up, low, nodal,
                          reverse_frames_indices) -> np.ndarray:
        height, width = frame.shape[:2]
        border_threshold = 300
        valid_x_range = (border_threshold, width - border_threshold)
        valid_y_range = (0, height)

        def valid_point(point):
            if point is None:
                return False
            x, y = map(int, point)
            return (valid_x_range[0] < x < valid_x_range[1] and valid_y_range[0] < y < valid_y_range[1])

        valid_up = valid_point(up)
        valid_low = valid_point(low)
        valid_nodal = valid_point(nodal)

        point_radius = 10
        colors = {
            'up': (0, 255, 0),    # 绿色
            'low': (255, 0, 0),   # 蓝色
            'nodal': (0, 0, 255)  # 红色
        }

        if valid_up:
            cv2.circle(frame, tuple(map(int, up)), point_radius, colors['up'], -1)

        if valid_low:
            cv2.circle(frame, tuple(map(int, low)), point_radius, colors['low'], -1)

        if valid_nodal:
            cv2.circle(frame, tuple(map(int, nodal)), point_radius, colors['nodal'], -1)

        if valid_up and self.frame_idx in reverse_frames_indices:
            center_point = (width // 2, height // 2)
            low_point = tuple(map(int, up))
            arrow_color = (0, 255, 0)  # 绿色
            arrow_thickness = 10
            cv2.arrowedLine(frame, center_point, low_point, arrow_color, arrow_thickness)

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

                frame = np.ascontiguousarray(frame, dtype=np.uint8)

                # ========== 步骤1: YOLO检测 + SAM2跟踪 ==========
                if not self.if_init:
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
                    _, out_mask_logits = self.segmentor.track(frame)

                # ========== 步骤2: 获取mask ==========
                mask = out_mask_logits.sigmoid().cpu().numpy()
                mask = mask[0, 0]
                mask = (mask > 0.5).astype(np.uint8) * 255

                # ========== 步骤3: 实时计算角度（内存计算，不存JSON）==========
                angle, ratio, A, B, C = compute_main(mask)

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
                    reverse_frames_indices = sorted(list(set([idx for idx in reverse_res if isinstance(idx, int)])))
                    

                # ========== 步骤5: 绘制可视化 ==========
                frame_with_mask = self._apply_mask_visualization(frame, mask)

                self._current_camera_control = self._compute_camera_control(frame, A, mask)
                
                xyz_array = self._current_camera_control.get('xyz_array', [0, 0, 0])
                is_event = self.frame_idx in reverse_frames_indices
                should_emit, stable_xyz = self.stabilizer.update(self.frame_idx, xyz_array, is_event=is_event)
                
                if should_emit:
                    if abs(stable_xyz[2]) > 0: 
                        z_only_cmd = [0, 0, stable_xyz[2]]
                        self.movement_data.append([self.frame_idx, z_only_cmd[0], z_only_cmd[1], z_only_cmd[2]])
                        self._current_camera_control['camera_command'] = f"{z_only_cmd[0]} {z_only_cmd[1]} {z_only_cmd[2]}"
                        self._current_camera_control['xyz_array'] = z_only_cmd
                    else:
                        self.movement_data.append([self.frame_idx, stable_xyz[0], stable_xyz[1], stable_xyz[2]])
                        self._current_camera_control['camera_command'] = f"{stable_xyz[0]} {stable_xyz[1]} {stable_xyz[2]}"
                        self._current_camera_control['xyz_array'] = stable_xyz
                else:
                    self._current_camera_control['camera_command'] = "0 0 0"
                    self._current_camera_control['xyz_array'] = [0, 0, 0]

                
                display_frame = self._draw_annotations(
                    frame_with_mask,
                    angle,
                    ratio,
                    A,
                    B,
                    C,
                    reverse_frames_indices
                )

                status_text = "Press 'r' to manual reset | 'q' to quit"
                cv2.putText(display_frame, status_text,
                            (10, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Realtime Tracking", display_frame)

                arrow_active = (A is not None and
                                self.frame_idx in reverse_frames_indices and
                                self._is_point_within_annotation_bounds(A, frame))
                
                if arrow_active and should_emit:
                    camera_cmd = self._current_camera_control.get('camera_command', '0 0 0')
                    print(f"帧 {self.frame_idx} | 角度: {angle:.2f}° | 比例: {ratio:.2f} | XYZ: {camera_cmd}")


                self.frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\n[手动重置] 重新初始化跟踪器...")
                    print("清空历史数据，准备重新检测...")

                    self.if_init = False
                    self.frame_idx = 0
                    self.angles.clear()
                    self.ratios.clear()
                    self.points_up.clear()
                    self.points_low.clear()
                    self.points_nodal.clear()

                    try:
                        if hasattr(self.segmentor, 'condition_state'):
                            self.segmentor.condition_state.clear()
                        
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
        
        self._save_movement_data()

    def _is_point_within_annotation_bounds(self, point: tuple, frame: np.ndarray) -> bool:
        if point is None:
            return False
        height, width = frame.shape[:2]
        border_threshold = 300
        valid_x_range = (border_threshold, width - border_threshold)
        valid_y_range = (0, height)
        x, y = map(int, point)
        return (valid_x_range[0] < x < valid_x_range[1]) and (valid_y_range[0] < y < valid_y_range[1])

# =============================================================
# 主函数
# =============================================================
def main():
    YOLO_CHECKPOINT = '/home/ubuntu/dcai/roboendo/models/yolov8x.pt'
    SAM2_CHECKPOINT = '/home/ubuntu/dcai/roboendo/models/sam2.1_hiera_large.pt'
    SAM2_CFG = "sam2.1_hiera_l.yaml"

    pipeline = RealtimeTrackingPipeline(
        yolo_checkpoint=YOLO_CHECKPOINT,
        sam2_checkpoint=SAM2_CHECKPOINT,
        sam2_cfg=SAM2_CFG,
        window_size=30,  # 可调整窗口大小
        step_size=15     # 可调整步长
    )

    pipeline.run(camera_id="/home/ubuntu/dcai/roboendo/videos/test2.mp4")

if __name__ == "__main__":
    main()