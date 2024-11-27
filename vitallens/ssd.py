import logging
import math
import numpy as np
from prpy.numpy.signal import interpolate_vals
from typing import Tuple
import mediapipe as mp
import cv2

INPUT_SIZE = (240, 320)
MAX_SCAN_FRAMES = 60
MAX_LIVE_FEED_FRAMES = 900  # Capture 900 frames

# Define landmark indices for regions of interest (ROIs)
_left_cheek = [117, 118, 119, 120, 100, 142, 203, 206, 205, 50, 117]
_right_cheek = [346, 347, 348, 349, 329, 371, 423, 426, 425, 280, 346]
_forehead = [109, 10, 338, 337, 336, 285, 8, 55, 107, 108, 109]

class FaceDetector:
    def __init__(
        self,
        max_faces: int,
        fs: float,
        score_threshold: float,
        iou_threshold: float):
        """Initialise the face detector."""
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_faces = max_faces
        self.fs = fs
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=score_threshold)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_faces)
   
    def __call__(
        self,
        inputs: Tuple[np.ndarray, str],
        inputs_shape: Tuple[tuple, float],
        fps: float,livefeed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on video frames, modified for live video input."""
        
        if livefeed:
            # Capture live feed from webcam
            cap = cv2.VideoCapture(0)
            frames = []
            frame_count = 0
            while frame_count < MAX_LIVE_FEED_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB as MediaPipe uses RGB images
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
            cap.release()
            inputs = np.array(frames)
        elif isinstance(inputs, str):
            # Load video frames directly using OpenCV
            cap = cv2.VideoCapture(inputs)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB as MediaPipe uses RGB images
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            inputs = np.array(frames)
            cap.release()

            # Now `inputs` should be a numpy array, and it's safe to check its shape
            #print(f"Inputs shape: {inputs.shape}")
        # Determine number of batches
        n_frames = inputs_shape[0]
        n_batches = math.ceil((n_frames / (fps / self.fs)) / MAX_SCAN_FRAMES)
        if n_batches > 1:
            logging.info("Running face detection in {} batches...".format(n_batches))
        # Determine frame offsets for batches
        offsets_lengths = [(i[0], len(i)) for i in np.array_split(np.arange(n_frames), n_batches)]
        # Process in batches
        results = [self.scan_batch(inputs=inputs, batch=i, n_batches=n_batches, start=int(s), end=int(s+l), fps=fps)
                   for i, (s, l) in enumerate(offsets_lengths)]
        boxes = np.concatenate([r[0] for r in results], axis=0)
        classes = np.concatenate([r[1] for r in results], axis=0)
        scan_idxs = np.concatenate([r[2] for r in results], axis=0)
        scan_every = int(np.max(np.diff(scan_idxs)))
        n_frames_scan = boxes.shape[0]
        
        # Add print statements to debug
        #print(f"Number of frames processed: {inputs.shape[0]}")
        #print(f"Number of face detections: {boxes.shape[0]}")
        #print(f"Shape of boxes: {boxes.shape}")
        #print(f"Shape of classes: {classes.shape}")
        #print(f"Shape of idxs: {scan_idxs.shape}")

        # Ensure the number of detections matches the number of frames
        if boxes.shape[0] != inputs.shape[0]:
            raise ValueError(f"Number of detections ({boxes.shape[0]}) does not match number of frames ({inputs.shape[0]})")
        # Check if any faces found
        if boxes.shape[1] == 0:
            logging.warning("No faces found")
            return [], []

        # Assort info: idx, scanned, scan_found_face, confidence
        idxs = np.repeat(scan_idxs[:, np.newaxis], boxes.shape[1], axis=1)[..., np.newaxis]  # Frame index and confidence score
        scanned = np.ones((n_frames_scan, boxes.shape[1], 1), dtype=np.int32)  # Scanned =1 if not scanned =0
        scan_found_face = np.where(classes[..., 1:2] < self.score_threshold,
                                   np.zeros([n_frames_scan, boxes.shape[1], 1], dtype=np.int32), scanned)  # Scan_found_face =1 if not found =0
        info = np.r_['2', idxs, scanned, scan_found_face, classes[..., 1:2]]

        return boxes, info
    def scan_batch(
        self,
        inputs: np.ndarray,
        batch: int,
        n_batches: int,
        start: int,
        end: int,
        fps: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse video and run inference for one batch."""
        
        inputs_batch = inputs[start:end]
        boxes, classes, landmarks_list = [], [], []
        
        for frame in inputs_batch:
            face_mesh_results = self.face_mesh.process(frame)
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    left_cheek = self.extract_landmarks(face_landmarks.landmark, _left_cheek)
                    right_cheek = self.extract_landmarks(face_landmarks.landmark, _right_cheek)
                    forehead = self.extract_landmarks(face_landmarks.landmark, _forehead)

                    landmarks_list.append((left_cheek, right_cheek, forehead))
                    
                    x_values = [landmark.x for landmark in face_landmarks.landmark]
                    y_values = [landmark.y for landmark in face_landmarks.landmark]
                    bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
                    boxes.append(bbox)
                    classes.append([1.0, 1.0])  # Confidence score placeholder
            else:
                boxes.append([0, 0, 0, 0])
                classes.append([0, 0])

        boxes = np.array(boxes).reshape([-1, self.max_faces, 4])
        classes = np.array(classes).reshape([-1, self.max_faces, 2])
        idxs = np.arange(start, end)

        return boxes, classes, idxs, landmarks_list

    def extract_landmarks(self, landmarks, indices):
        """Extract landmarks for the specified region indices."""
        max_index = len(landmarks) - 1
        selected_landmarks = []
        for i in indices:
            if i <= max_index:
                selected_landmarks.append((landmarks[i].x, landmarks[i].y))
            else:
                selected_landmarks.append((0, 0))
        return selected_landmarks
