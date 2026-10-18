"""
YOLO Icon Detector using ONNX Runtime
Uses OmniParser icon detection model for UI element recognition
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import cv2


class YOLODetector:
    def __init__(self, model_path: str = "models/onnx/model.onnx"):
        """
        Initialize YOLO detector with ONNX model
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (640, 640)  # Standard YOLO input size
        
    def load(self):
        """Load the ONNX model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ YOLO model loaded from {self.model_path}")
        
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for YOLO inference
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed numpy array (1, 3, height, width)
        """
        # Resize maintaining aspect ratio
        img = np.array(image)
        orig_h, orig_w = img.shape[:2]
        
        # Calculate scale and padding
        scale = min(self.input_size[0] / orig_w, self.input_size[1] / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to input_size
        pad_w = self.input_size[0] - new_w
        pad_h = self.input_size[1] - new_h
        img_padded = cv2.copyMakeBorder(
            img_resized, 
            0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)
        )
        
        # Convert RGB to BGR (if needed) and normalize
        img_padded = cv2.cvtColor(img_padded, cv2.COLOR_RGB2BGR)
        img_padded = img_padded.astype(np.float32) / 255.0
        
        # HWC to CHW
        img_padded = img_padded.transpose(2, 0, 1)
        
        # Add batch dimension
        img_padded = np.expand_dims(img_padded, axis=0)
        
        return img_padded, scale, (pad_w, pad_h)
    
    def postprocess(self, output: np.ndarray, orig_shape: Tuple[int, int], 
                    scale: float, padding: Tuple[int, int], 
                    conf_threshold: float = 0.5, iou_threshold: float = 0.5) -> List[Dict]:
        """
        Postprocess YOLO output to bounding boxes
        
        Output format from OmniParser icon_detect: (batch, 5, 8400)
        - 5 channels: x, y, w, h, confidence (no class info for icon detection)
        - 8400: total anchor points
        
        Args:
            output: Raw model output
            orig_shape: Original image shape (height, width)
            scale: Scale factor used in preprocessing
            padding: Padding used in preprocessing (pad_w, pad_h)
            conf_threshold: Confidence threshold for filtering
            
        Returns:
            List of detections with bbox, confidence, and class
        """
        detections = []
        
        try:
            # Output shape: (1, 5, 8400)
            # Transpose to (8400, 5) for easier processing
            predictions = output[0].transpose(1, 0)  # (8400, 5)
            
            # Extract boxes (first 4) and confidence (last 1)
            boxes = predictions[:, :4]  # (8400, 4) - x, y, w, h
            confidences = predictions[:, 4]  # (8400,) - confidence
            
            # Filter by confidence threshold
            mask = confidences > conf_threshold
            boxes = boxes[mask]
            confidences = confidences[mask]
            
            if len(boxes) == 0:
                return detections
            
            # Sort by confidence (descending)
            sorted_indices = np.argsort(confidences)[::-1]
            boxes = boxes[sorted_indices]
            confidences = confidences[sorted_indices]
            
            # Apply Non-Maximum Suppression to remove duplicates
            keep_indices = self.nms(boxes, confidences, iou_threshold)
            boxes = boxes[keep_indices]
            confidences = confidences[keep_indices]
            
            # Process each detection
            for box, conf in zip(boxes, confidences):
                x, y, w, h = box
                
                # Convert from center/width-height to top-left/width-height
                x1 = x - w / 2
                y1 = y - h / 2
                
                # Remove padding and scale back to original image
                x1 = (x1 - padding[0] / 2) / scale
                y1 = (y1 - padding[1] / 2) / scale
                w = w / scale
                h = h / scale
                
                # Clip to image bounds
                x1 = max(0, min(x1, orig_shape[1]))
                y1 = max(0, min(y1, orig_shape[0]))
                w = min(w, orig_shape[1] - x1)
                h = min(h, orig_shape[0] - y1)
                
                # Skip if box is too small
                if w < 5 or h < 5:
                    continue
                
                # Calculate center as percentage
                center_x = (x1 + w / 2) / orig_shape[1]
                center_y = (y1 + h / 2) / orig_shape[0]
                
                detections.append({
                    'bbox': {
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(w),
                        'height': int(h)
                    },
                    'center_pct': {
                        'x': center_x * 100,
                        'y': center_y * 100
                    },
                    'confidence': float(conf),
                    'class': 'icon'  # All detections are icons
                })
                
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            import traceback
            traceback.print_exc()
            
        return detections
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Apply Non-Maximum Suppression
        
        Args:
            boxes: Array of boxes in (x, y, w, h) format
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        # Convert to (x1, y1, x2, y2) format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Sort by score (descending) - already sorted when called
        indices = np.arange(len(boxes))
        
        keep = []
        while len(indices) > 0:
            # Keep the box with highest score
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            iou = self.calculate_iou(
                np.array([x1[i], y1[i], x2[i], y2[i]]),
                np.stack([x1[indices[1:]], y1[indices[1:]], x2[indices[1:]], y2[indices[1:]]], axis=1)
            )
            
            # Keep boxes with IoU below threshold
            mask = iou < iou_threshold
            indices = indices[1:][mask]
        
        return np.array(keep)
    
    def calculate_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between a box and multiple boxes
        
        Args:
            box1: Single box (x1, y1, x2, y2)
            boxes2: Multiple boxes (N, 4) in (x1, y1, x2, y2) format
            
        Returns:
            Array of IoU values
        """
        # Calculate intersection
        x1_inter = np.maximum(box1[0], boxes2[:, 0])
        y1_inter = np.maximum(box1[1], boxes2[:, 1])
        x2_inter = np.minimum(box1[2], boxes2[:, 2])
        y2_inter = np.minimum(box1[3], boxes2[:, 3])
        
        inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union_area = box1_area + boxes2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def detect(self, image_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect icons in image
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections
        """
        if self.session is None:
            self.load()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        orig_shape = image.size[::-1]  # (height, width)
        
        # Preprocess
        img_input, scale, padding = self.preprocess(image)
        
        # Run inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: img_input}
        )[0]
        
        # Postprocess
        detections = self.postprocess(
            output, orig_shape, scale, padding, conf_threshold
        )
        
        return detections
    
    def find_icon_by_position(self, image_path: str, icon_type: str = 'top_left') -> Dict:
        """
        Find an icon by expected position (heuristic for UI elements)
        
        Args:
            image_path: Path to image
            icon_type: Type of icon to find ('top_left', 'top_right', 'top_bar', 'small')
            
        Returns:
            Detection result or None
        """
        detections = self.detect(image_path, conf_threshold=0.2)
        
        if not detections:
            return None
        
        image = Image.open(image_path)
        img_w, img_h = image.size
        
        # Filter detections based on expected position AND size
        # Prefer smaller icons (buttons) over large UI elements (address bars, menus)
        filtered = []
        for det in detections:
            center_x = det['bbox']['x'] + det['bbox']['width'] / 2
            center_y = det['bbox']['y'] + det['bbox']['height'] / 2
            size = det['bbox']['width'] * det['bbox']['height']
            
            # Skip very large detections (likely UI containers, not icons)
            # Max icon size: 200x200 pixels (40000 sq pixels)
            if size > 40000:
                continue
            
            if icon_type == 'top_left':
                # Top-left corner (back, menu, etc.)
                if center_x < img_w * 0.2 and center_y < img_h * 0.1:
                    filtered.append(det)
            elif icon_type == 'top_right':
                # Top-right corner (profile, settings, etc.)
                if center_x > img_w * 0.8 and center_y < img_h * 0.1:
                    filtered.append(det)
            elif icon_type == 'top_bar':
                # Top bar area (tabs, address bar, etc.)
                if center_y < img_h * 0.15:
                    filtered.append(det)
            elif icon_type == 'small':
                # Small icons (buttons, etc.)
                if size < 10000:  # Less than 100x100 pixels
                    filtered.append(det)
            else:
                # Return all (but still filter by size)
                filtered.append(det)
        
        if not filtered:
            # Fallback to small icons in top bar
            for det in detections:
                center_y = det['bbox']['y'] + det['bbox']['height'] / 2
                size = det['bbox']['width'] * det['bbox']['height']
                if center_y < img_h * 0.15 and size < 10000:
                    filtered.append(det)
        
        if not filtered:
            return None
        
        # Sort by size (smaller first) then confidence
        # This prefers actual buttons over large UI elements
        filtered.sort(key=lambda d: (d['bbox']['width'] * d['bbox']['height'], -d['confidence']))
        return filtered[0]
    
    def find_icon_near_text(self, image_path: str, text_location: Dict, 
                          icon_type: str = 'plus') -> Dict:
        """
        Find an icon near a text location (heuristic for new tab, close, etc.)
        
        Args:
            image_path: Path to image
            text_location: Text location with bbox (OCR format: bbox_px tuple or dict)
            icon_type: Type of icon to find ('plus', 'close', 'menu')
            
        Returns:
            Detection result or None
        """
        detections = self.detect(image_path, conf_threshold=0.2)
        
        if not detections:
            return None
        
        # Handle both OCR bbox format (tuple) and YOLO format (dict)
        if isinstance(text_location.get('bbox_px'), tuple):
            left, top, width, height = text_location['bbox_px']
            text_center = (left + width / 2, top + height / 2)
        elif isinstance(text_location.get('bbox'), dict):
            bbox = text_location['bbox']
            text_center = (
                bbox['x'] + bbox['width'] / 2,
                bbox['y'] + bbox['height'] / 2
            )
        else:
            return None
        
        # Find closest detection to the text
        best_detection = None
        min_distance = float('inf')
        
        for det in detections:
            det_center = (
                det['bbox']['x'] + det['bbox']['width'] / 2,
                det['bbox']['y'] + det['bbox']['height'] / 2
            )
            
            # Calculate distance
            distance = np.sqrt(
                (det_center[0] - text_center[0])**2 + 
                (det_center[1] - text_center[1])**2
            )
            
            # Check if icon is in expected direction
            if icon_type == 'plus':
                # Plus icon should be to the right of "New Tab" text
                if det_center[0] > text_center[0] and distance < 100:
                    if distance < min_distance:
                        min_distance = distance
                        best_detection = det
            elif icon_type == 'close':
                # Close icon should be near tab
                if distance < 50:
                    if distance < min_distance:
                        min_distance = distance
                        best_detection = det
            else:
                # Just find closest
                if distance < min_distance:
                    min_distance = distance
                    best_detection = det
        
        return best_detection


def test_detector():
    """Test the YOLO detector"""
    detector = YOLODetector()
    
    # Test on an image
    test_image = "images/1"
    if Path(test_image).exists():
        print(f"\nTesting on {test_image}...")
        detections = detector.detect(test_image)
        print(f"Found {len(detections)} detections")
        
        for i, det in enumerate(detections[:5]):  # Show first 5
            print(f"  Detection {i+1}:")
            print(f"    Confidence: {det['confidence']:.2f}")
            print(f"    Center: ({det['center_pct']['x']:.1f}%, {det['center_pct']['y']:.1f}%)")
            print(f"    BBox: {det['bbox']}")


if __name__ == "__main__":
    test_detector()
