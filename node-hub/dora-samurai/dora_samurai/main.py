"""SAMURAI Video Object Tracking Dora Node.

This node provides zero-shot video object tracking using the SAMURAI model,
which adapts SAM2 for tracking with motion-aware memory.
"""

import cv2
import numpy as np
import pyarrow as pa
import torch
from dora import Node
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from samurai.sam2.build_sam import build_sam2_video_predictor_hf


class SAMURAITracker:
    """SAMURAI video tracker that manages state for tracking objects through video."""
    
    def __init__(self, model_id="facebook/sam2.1-hiera-base-plus", device="auto"):
        """Initialize SAMURAI tracker.
        
        Args:
            model_id: HuggingFace model ID for SAM2 model
            device: Device to run inference on ('cuda', 'mps', 'cpu', or 'auto')
        """
        # Determine best available device if set to auto
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.predictor = build_sam2_video_predictor_hf(model_id, device=device)
        self.state = None
        self.frame_buffer = []
        self.frame_count = 0
        self.tracking_initialized = False
        self.object_ids = []
        
    def add_frame(self, frame):
        """Add a frame to the tracking buffer.
        
        Args:
            frame: RGB image as numpy array
        
        Returns:
            frame_idx: Index of the added frame
        """
        if self.state is None:
            # Convert frame list to format expected by SAMURAI
            self.frame_buffer.append(frame)
            
            if len(self.frame_buffer) == 1:  # Initialize with first frame
                with torch.inference_mode():
                    self.state = self.predictor.init_state(
                        self.frame_buffer,
                        offload_video_to_cpu=True
                    )
        else:
            # If state already initialized, we can just append to the buffer
            self.frame_buffer.append(frame)
            # Update the state to include the new frame
            self.state["images"].append(torch.from_numpy(frame).to(self.device))
            self.state["num_frames"] += 1
        
        return len(self.frame_buffer) - 1
    
    def add_object(self, frame_idx, box, obj_id=None):
        """Add a new object to track based on a bounding box.
        
        Args:
            frame_idx: Index of the frame containing the object
            box: Bounding box in xyxy format (x1, y1, x2, y2)
            obj_id: Optional object ID (integer)
            
        Returns:
            obj_id: ID of the added object
            mask: Binary mask for the object
        """
        if not self.state:
            raise RuntimeError("Tracker must be initialized with frames before adding objects")
            
        if obj_id is None:
            # Automatically assign an object ID if not provided
            obj_id = len(self.object_ids)
        
        with torch.inference_mode():
            # Add bounding box for tracking
            _, _, masks = self.predictor.add_new_points_or_box(
                self.state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box
            )
            
            # Record that we've added this object
            if obj_id not in self.object_ids:
                self.object_ids.append(obj_id)
                
        # Return the object ID and mask
        return obj_id, masks[0]
    
    def track(self, start_frame=0, max_frames=None):
        """Track objects through video sequence.
        
        Args:
            start_frame: First frame to track from
            max_frames: Maximum number of frames to track (None for all)
            
        Yields:
            frame_idx: Index of the current frame
            object_ids: List of object IDs present in the frame
            masks: Binary masks for each object
        """
        if not self.state or not self.object_ids:
            raise RuntimeError("Must add frames and objects before tracking")
        
        with torch.inference_mode():
            # Mark that tracking has started
            self.tracking_initialized = True
            
            # Track through video frames
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                self.state, 
                start_frame_idx=start_frame,
                max_frame_num_to_track=max_frames
            ):
                yield frame_idx, object_ids, masks
    
    def reset(self):
        """Reset the tracker state."""
        self.state = None
        self.frame_buffer = []
        self.frame_count = 0
        self.tracking_initialized = False
        self.object_ids = []


def main():
    """Main entry point for the SAMURAI Dora node."""
    # Initialize PyArrow array to make sure library is loaded
    pa.array([])
    
    # Initialize the Dora node
    node = Node()
    
    # Create SAMURAI tracker
    tracker = SAMURAITracker(device="auto")
    
    # Track node state
    frames = {}  # Map frame IDs to actual frames
    current_tracks = defaultdict(list)  # Map object IDs to their tracks
    all_object_ids = set()  # Keep track of all object IDs
    frame_buffer = []  # Buffer of frames for SAMURAI
    boxes_buffer = []  # Buffer of boxes waiting to be processed
    next_obj_id = 0  # Next object ID to assign
    tracking_active = False  # Whether tracking is currently active
    
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            # Process incoming video frames
            if "image" in event_id or "frame" in event_id:
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]
                frame_id = metadata.get("frame_id", event_id)

                # Process different image encodings
                if encoding in ["bgr8", "rgb8", "jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    storage_type = np.uint8
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")

                # Extract the frame data based on encoding
                if encoding == "bgr8":
                    frame = storage.to_numpy().astype(storage_type).reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                elif encoding == "rgb8":
                    frame = storage.to_numpy().astype(storage_type).reshape((height, width, channels))
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    storage = storage.to_numpy()
                    frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Store the frame
                frames[frame_id] = frame
                
                # Add frame to SAMURAI tracker
                frame_idx = tracker.add_frame(frame)
                frame_buffer.append((frame_id, frame_idx))
                
                # Process any pending bounding boxes
                pending_boxes = [b for b in boxes_buffer if b[0] == frame_id]
                for frame_id, box, box_label in pending_boxes:
                    obj_id = next_obj_id
                    next_obj_id += 1
                    
                    # Initialize tracking with this box
                    tracker.add_object(frame_idx, box, obj_id)
                    all_object_ids.add(obj_id)
                    
                    # Remove processed box from buffer
                    boxes_buffer = [b for b in boxes_buffer if b != (frame_id, box, box_label)]
                
                # If we have frames and objects, start tracking if not already active
                if not tracking_active and tracker.object_ids:
                    tracking_active = True
                    
                    # Start a separate thread for tracking
                    try:
                        # Process accumulated frames
                        for frame_idx, object_ids, masks in tracker.track():
                            if frame_idx < len(frame_buffer):
                                orig_frame_id, _ = frame_buffer[frame_idx]
                                
                                # Convert masks to the format expected by Dora
                                for i, obj_id in enumerate(object_ids):
                                    mask = masks[i].cpu().numpy() > 0.0
                                    
                                    # Store the track for this object
                                    current_tracks[obj_id].append((orig_frame_id, mask))
                                    
                                    # Calculate bounding box from mask
                                    y_indices, x_indices = np.where(mask)
                                    if len(y_indices) > 0 and len(x_indices) > 0:
                                        x1, y1 = np.min(x_indices), np.min(y_indices)
                                        x2, y2 = np.max(x_indices), np.max(y_indices)
                                        bbox = [x1, y1, x2, y2]
                                    else:
                                        bbox = [0, 0, 0, 0]
                                    
                                    # Send output mask and box for this object
                                    node.send_output(
                                        "masks",
                                        pa.array([{
                                            "mask": mask.ravel(),
                                            "object_id": obj_id
                                        }]),
                                        metadata={
                                            "frame_id": orig_frame_id,
                                            "width": width,
                                            "height": height
                                        }
                                    )
                                    
                                    node.send_output(
                                        "tracks",
                                        pa.array([{
                                            "bbox": bbox,
                                            "object_id": obj_id
                                        }]),
                                        metadata={
                                            "frame_id": orig_frame_id,
                                            "width": width,
                                            "height": height
                                        }
                                    )
                    except Exception as e:
                        node.send_error(f"Error during tracking: {str(e)}")

            # Process incoming bounding boxes for tracking initialization
            elif "boxes" in event_id or "bbox" in event_id:
                metadata = event["metadata"]
                frame_id = metadata.get("frame_id", metadata.get("image_id"))
                encoding = metadata.get("encoding", "xyxy")
                
                # Parse bounding box data based on different possible formats
                if isinstance(event["value"], pa.StructArray):
                    try:
                        box_data = event["value"][0]
                        if "bbox" in box_data.type:
                            boxes = box_data["bbox"].values.to_numpy().reshape(-1, 4)
                            labels = box_data.get("labels", pa.array([0] * len(boxes)))
                            if hasattr(labels, "to_numpy"):
                                labels = labels.to_numpy(zero_copy_only=False)
                    except (KeyError, IndexError):
                        boxes = event["value"].to_numpy().reshape(-1, 4)
                        labels = [0] * (len(boxes) // 4)
                else:
                    boxes = event["value"].to_numpy().reshape(-1, 4)
                    labels = [0] * (len(boxes) // 4)
                
                # Convert box format if needed
                if encoding != "xyxy":
                    # Convert from xywh to xyxy
                    if encoding == "xywh":
                        for i, box in enumerate(boxes):
                            x, y, w, h = box
                            boxes[i] = [x, y, x + w, y + h]
                    else:
                        raise RuntimeError(f"Unsupported box encoding: {encoding}")
                
                # If we have the corresponding frame, initialize tracking
                if frame_id in frames:
                    frame = frames[frame_id]
                    frame_idx = None
                    
                    # Find the frame index in our buffer
                    for i, (f_id, f_idx) in enumerate(frame_buffer):
                        if f_id == frame_id:
                            frame_idx = f_idx
                            break
                    
                    if frame_idx is not None:
                        # Add each box as a separate object to track
                        for i, box in enumerate(boxes):
                            obj_id = next_obj_id
                            next_obj_id += 1
                            label = labels[i] if i < len(labels) else 0
                            
                            # Initialize tracking with this box
                            tracker.add_object(frame_idx, box, obj_id)
                            all_object_ids.add(obj_id)
                    else:
                        # Store boxes to process when the frame arrives
                        for i, box in enumerate(boxes):
                            label = labels[i] if i < len(labels) else 0
                            boxes_buffer.append((frame_id, box, label))
                else:
                    # Store boxes to process when the frame arrives
                    for i, box in enumerate(boxes):
                        label = labels[i] if i < len(labels) else 0
                        boxes_buffer.append((frame_id, box, label))
            
            # Handle commands
            elif event_id == "command":
                command = event["value"].as_py()
                if command == "reset":
                    # Reset the tracker
                    tracker.reset()
                    frames = {}
                    current_tracks = defaultdict(list)
                    all_object_ids = set()
                    frame_buffer = []
                    boxes_buffer = []
                    next_obj_id = 0
                    tracking_active = False
                    
                    node.send_output("status", pa.array(["reset_complete"]))

        elif event_type == "ERROR":
            print(f"Event Error: {event['error']}")

if __name__ == "__main__":
    main()