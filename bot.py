import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pyautogui
import numpy as np
from PIL import Image
import cv2
import time
from typing import List, Tuple, Optional
import easyocr
from ultralytics import YOLO
import torch

class ScreenBot:
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the bot with OCR and YOLO models
        
        Args:
            confidence_threshold: Minimum confidence for YOLO detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Detect and configure GPU device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"   Using device: {self.device}")
        else:
            print("⚠️  No GPU detected. Using CPU (will be slower)")
            print(f"   Using device: {self.device}")
        
        # Initialize OCR reader (supports multiple languages)
        print(f"Loading OCR model on {self.device}...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        print("OCR model loaded!")
        
        # Initialize YOLO model (using YOLOv8)
        print(f"Loading YOLO model on {self.device}...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
            
            # Explicitly move YOLO model to GPU if available
            if self.use_gpu:
                self.yolo_model.to(self.device)
            
            print("YOLO model loaded!")
        except Exception as e:
            print(f"Warning: Could not load YOLO model. Object detection will be unavailable: {e}")
            self.yolo_model = None
        
        # Safety: Fail-safe pause
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True
        
        # Screen region of interest (x, y, width, height) - None means full screen
        self.screen_region = (0, 0, 1000, 1080)  # Default region to exclude right menu
        print(f"Default screen region set: x=0, y=0, width=1000, height=1080")
    
    def set_screen_region(self, x: int, y: int, width: int, height: int):
        """
        Set the screen region of interest for detection
        
        Args:
            x: Left edge of region
            y: Top edge of region
            width: Width of region
            height: Height of region
        """
        self.screen_region = (x, y, width, height)
        print(f"Screen region set: x={x}, y={y}, width={width}, height={height}")
    
    def clear_screen_region(self):
        """Clear the screen region to use full screen"""
        self.screen_region = None
        print("Screen region cleared - using full screen")
    
    def _to_screen_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert coordinates from region to screen coordinates"""
        if self.screen_region is None:
            return (x, y)
        region_x, region_y, _, _ = self.screen_region
        return (x + region_x, y + region_y)
    
    def take_screenshot(self, full_screen: bool = False) -> np.ndarray:
        """
        Take a screenshot and return as numpy array
        
        Args:
            full_screen: If True, always return full screen (ignores region). 
                        If False, crops to region if set.
        """
        screenshot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Crop to region if set and not requesting full screen
        if self.screen_region is not None and not full_screen:
            x, y, width, height = self.screen_region
            img = img[y:y+height, x:x+width]
        
        return img
    
    def find_text_ocr(self, text_to_find: str, screen_img: Optional[np.ndarray] = None, return_bbox: bool = False) -> List[Tuple[int, int]]:
        """
        Find text on screen using OCR
        
        Args:
            text_to_find: Text to search for (case-insensitive)
            screen_img: Optional pre-captured screenshot
            return_bbox: If True, returns (center_x, center_y, bbox, text, confidence) tuples
            
        Returns:
            List of (x, y) coordinates where text was found (center of text)
            OR List of (x, y, bbox, text, confidence) if return_bbox=True
        """
        if screen_img is None:
            screen_img = self.take_screenshot()
        
        # Convert to RGB for OCR
        rgb_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        results = self.ocr_reader.readtext(rgb_img)
        
        matches = []
        text_lower = text_to_find.lower()
        
        for (bbox, text, confidence) in results:
            if text_lower in text.lower():
                # Calculate center of bounding box (relative to cropped region)
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = int(sum(x_coords) / len(x_coords))
                center_y = int(sum(y_coords) / len(y_coords))
                
                # Convert to screen coordinates
                screen_x, screen_y = self._to_screen_coords(center_x, center_y)
                
                if return_bbox:
                    # Adjust bbox coordinates to screen space
                    adjusted_bbox = [(self._to_screen_coords(int(p[0]), int(p[1]))) for p in bbox]
                    matches.append((screen_x, screen_y, adjusted_bbox, text, confidence))
                else:
                    matches.append((screen_x, screen_y))
                print(f"Found '{text}' at ({screen_x}, {screen_y}) with confidence {confidence:.2f}")
        
        return matches
    
    def get_all_ocr_text(self, screen_img: Optional[np.ndarray] = None) -> List[Tuple]:
        """Get all text detected by OCR on screen with bounding boxes"""
        if screen_img is None:
            screen_img = self.take_screenshot()
        
        # Convert to RGB for OCR
        rgb_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        results = self.ocr_reader.readtext(rgb_img)
        
        all_text = []
        for (bbox, text, confidence) in results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            # Convert to screen coordinates
            screen_x, screen_y = self._to_screen_coords(center_x, center_y)
            # Adjust bbox coordinates to screen space
            adjusted_bbox = [self._to_screen_coords(int(p[0]), int(p[1])) for p in bbox]
            all_text.append((screen_x, screen_y, adjusted_bbox, text, confidence))
        
        return all_text
    
    def find_objects_yolo(self, object_class: Optional[str] = None, screen_img: Optional[np.ndarray] = None, return_bbox: bool = False) -> List[Tuple]:
        """
        Find objects on screen using YOLO
        
        Args:
            object_class: Optional class name to filter (e.g., 'button', 'person', 'mouse')
                         If None, returns all detected objects
            screen_img: Optional pre-captured screenshot
            return_bbox: If True, returns (center_x, center_y, class_name, confidence, bbox) tuples
            
        Returns:
            List of (x, y, class_name, confidence) tuples
            OR List of (x, y, class_name, confidence, (x1, y1, x2, y2)) if return_bbox=True
        """
        if self.yolo_model is None:
            print("YOLO model not available!")
            return []
        
        if screen_img is None:
            screen_img = self.take_screenshot()
        
        # Run YOLO detection with explicit device specification
        results = self.yolo_model(screen_img, conf=self.confidence_threshold, device=self.device)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Filter by class if specified
                if object_class is None or object_class.lower() in class_name.lower():
                    # Get bounding box (relative to cropped region)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Convert to screen coordinates
                    screen_x, screen_y = self._to_screen_coords(center_x, center_y)
                    
                    if return_bbox:
                        # Convert bbox to screen coordinates
                        screen_x1, screen_y1 = self._to_screen_coords(int(x1), int(y1))
                        screen_x2, screen_y2 = self._to_screen_coords(int(x2), int(y2))
                        detections.append((screen_x, screen_y, class_name, confidence, (screen_x1, screen_y1, screen_x2, screen_y2)))
                    else:
                        detections.append((screen_x, screen_y, class_name, confidence))
                    print(f"Found '{class_name}' at ({screen_x}, {screen_y}) with confidence {confidence:.2f}")
        
        return detections
    
    def click(self, x: int, y: int, button: str = 'left', clicks: int = 1):
        """Click at specified coordinates"""
        print(f"Clicking at ({x}, {y}) with {button} button, {clicks} times")
        pyautogui.click(x, y, button=button, clicks=clicks)
        time.sleep(0.2)  # Small delay after clicking
    
    def click_current(self, button: str = 'left', clicks: int = 1):
        """Click at current cursor position"""
        print(f"Clicking at current position with {button} button, {clicks} times")
        pyautogui.click(button=button, clicks=clicks)
        time.sleep(0.2)  # Small delay after clicking
    
    def move_rel(self, x_offset: int, y_offset: int):
        """Move mouse cursor relative to current position"""
        print(f"Moving cursor by ({x_offset}, {y_offset})")
        pyautogui.moveRel(x_offset, y_offset)
    
    def press_key(self, key: str, presses: int = 1):
        """Press a keyboard key"""
        print(f"Pressing '{key}' {presses} times")
        pyautogui.press(key, presses=presses)
    
    def type_text(self, text: str, interval: float = 0.05):
        """Type text"""
        print(f"Typing: {text}")
        pyautogui.write(text, interval=interval)
    
    def find_and_click_text(self, text: str, index: int = 0) -> bool:
        """
        Find text using OCR and click it
        
        Args:
            text: Text to find and click
            index: Which occurrence to click if multiple found (0 = first)
            
        Returns:
            True if found and clicked, False otherwise
        """
        print(f"Searching for text: '{text}'...")
        matches = self.find_text_ocr(text)
        
        if not matches:
            print(f"Text '{text}' not found on screen")
            return False
        
        if index >= len(matches):
            print(f"Index {index} out of range. Found {len(matches)} occurrence(s). Using index {len(matches)-1} instead.")
            index = len(matches) - 1
        
        x, y = matches[index]
        self.click(x, y)
        return True
    
    def find_and_point_text(self, text: str, index: int = 0) -> bool:
        """
        Find text using OCR and move cursor to it
        
        Args:
            text: Text to find and point to
            index: Which occurrence to point to if multiple found (0 = first)
            
        Returns:
            True if found and pointed, False otherwise
        """
        print(f"Searching for text: '{text}'...")
        matches = self.find_text_ocr(text)
        
        if not matches:
            print(f"Text '{text}' not found on screen")
            return False
        
        if index >= len(matches):
            print(f"Index {index} out of range. Found {len(matches)} occurrence(s). Using index {len(matches)-1} instead.")
            index = len(matches) - 1
        
        x, y = matches[index]
        print(f"Moving cursor to ({x}, {y})")
        pyautogui.moveTo(x, y)
        return True
    
    def find_and_click_object(self, object_class: str, index: int = 0) -> bool:
        """
        Find object using YOLO and click it
        
        Args:
            object_class: Class name to find (e.g., 'button', 'person')
            index: Which occurrence to click if multiple found
            
        Returns:
            True if found and clicked, False otherwise
        """
        print(f"Searching for object class: '{object_class}'...")
        detections = self.find_objects_yolo(object_class)
        
        if not detections:
            print(f"Object class '{object_class}' not found on screen")
            return False
        
        if index >= len(detections):
            print(f"Index {index} out of range. Found {len(detections)} occurrence(s). Using index {len(detections)-1} instead.")
            index = len(detections) - 1
        
        x, y, class_name, confidence = detections[index]
        self.click(x, y)
        return True
    
    def find_and_point_object(self, object_class: str, index: int = 0) -> bool:
        """
        Find object using YOLO and move cursor to it
        
        Args:
            object_class: Class name to find (e.g., 'button', 'person')
            index: Which occurrence to point to if multiple found
            
        Returns:
            True if found and pointed, False otherwise
        """
        print(f"Searching for object class: '{object_class}'...")
        detections = self.find_objects_yolo(object_class)
        
        if not detections:
            print(f"Object class '{object_class}' not found on screen")
            return False
        
        if index >= len(detections):
            print(f"Index {index} out of range. Found {len(detections)} occurrence(s). Using index {len(detections)-1} instead.")
            index = len(detections) - 1
        
        x, y, class_name, confidence = detections[index]
        print(f"Moving cursor to ({x}, {y})")
        pyautogui.moveTo(x, y)
        return True
    
    def list_available_objects(self, screen_img: Optional[np.ndarray] = None):
        """List all objects currently detected on screen"""
        if self.yolo_model is None:
            print("YOLO model not available!")
            return
        
        print("Scanning screen for objects...")
        detections = self.find_objects_yolo(screen_img=screen_img)
        
        if not detections:
            print("No objects detected")
            return
        
        print(f"\nFound {len(detections)} objects:")
        for i, (x, y, class_name, confidence) in enumerate(detections):
            print(f"  {i}: {class_name} at ({x}, {y}) - confidence: {confidence:.2f}")


