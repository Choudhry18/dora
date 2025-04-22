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
from samurai.build_sam import build_sam2_video_predictor_hf


class SAMURAITracker:
    def __init__(self, model_name="facebook/sam2.1-hiera-base-plus", device="cuda:0"):
        """Initialize the SAMURAI tracker with the specified model."""
        self.model_name = model_name
        self.device = device
        self.predictor = None
        self.inference_state = None
        self.tracking_objects = {}  # Dictionary to track object IDs
        self.next_object_id = 0  # Counter for assigning new IDs
        self.is_initialized = False

    def initialize(self):
        """Initialize the model."""
        self.predictor = build_sam2_video_predictor_hf(self.model_name, device=self.device)
        self.is_initialized = True
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.inference_mode(), torch.autocast(self.device.split(":")[0], dtype=torch.float16):
            self.inference_state = self.predictor.init_streaming_state(dummy_image)
        self.is_initialized = True

    def process_frame(self, frame):
        """Process a frame and track objects."""
        if not self.is_initialized:
            self.initialize()
            
        pil_image = Image.fromarray(frame)
            
        with torch.inference_mode(), torch.autocast(self.device.split(":")[0], dtype=torch.float16):
            if self.inference_state is None:
                self.inference_state = self.predictor.init_streaming_state(pil_image)
                return None, None, None
            
            frame_idx, object_ids, masks = self.predictor.propagate_streaming(self.inference_state, pil_image)
            
            # Convert to format expected by downstream nodes
            results = []
            for obj_id, mask in zip(object_ids, masks):
                mask_binary = mask[0].cpu().numpy() > 0
                
                # Calculate bounding box from mask
                non_zero_indices = np.argwhere(mask_binary)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max, y_max]  # Convert to [x1, y1, x2, y2]
                
                    # Map internal object IDs to consistent external IDs
                    if obj_id not in self.tracking_objects:
                        self.tracking_objects[obj_id] = self.next_object_id
                        self.next_object_id += 1
                    
                    external_id = self.tracking_objects[obj_id]
                    
                    results.append({
                        "id": external_id,
                        "box": bbox,
                        "mask": mask_binary
                    })
            
            return frame_idx, results

    def add_object(self, frame, box):
        """Add a new object to track given a bounding box."""
        if not self.is_initialized:
            self.initialize()
            
        pil_image = Image.fromarray(frame)
            
        with torch.inference_mode(), torch.autocast(self.device.split(":")[0], dtype=torch.float16):
            if self.inference_state is None:
                self.inference_state = self.predictor.init_streaming_state(pil_image)
            
            obj_id = len(self.tracking_objects)
            self.predictor.add_new_points_or_box(self.inference_state, box=box, frame_idx=0, obj_id=obj_id)
            return obj_id


def run():
    """Run the SAMURAI tracker node."""
    node = Node()
    
    device = "cuda:0"
    model_name = "facebook/sam2.1-hiera-base-plus"
    
    # Initialize tracker
    tracker = SAMURAITracker(model_name=model_name, device=device)
    frames = {}
    
    print(f"SAMURAI Tracker initialized with model: {model_name} on device: {device}")
    
    for event in node:
        event_type = event["type"]
        
        if event_type == "INPUT":
            event_id = event["id"]
            
            if "frames" in event_id:
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]

                if (
                    encoding == "bgr8"
                    or encoding == "rgb8"
                    or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]
                ):
                    channels = 3
                    storage_type = np.uint8
                else:
                    error = f"Unsupported image encoding: {encoding}"
                    raise RuntimeError(error)

                if encoding == "bgr8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                elif encoding == "rgb8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    storage = storage.to_numpy()
                    frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")
                image = Image.fromarray(frame)
                # frames[event_id] = image
                
                # If we're in tracking mode, process the frame
                if tracker.inference_state is not None:
                    frame_idx, results = tracker.process_frame(frame)
                    
                    if results:
                        # Prepare bounding boxes and masks for output
                        bboxes = []
                        masks = []
                        for result in results:
                            bboxes.append(result["box"])
                            masks.append(result["mask"])
                        
                        # Send bounding boxes
                        node.send_output(
                            "bbox",
                            pa.array(bboxes),
                            metadata={
                                "frame_id": frame_idx,
                                "width": width,
                                "height": height
                            }
                        )

                        flattened_masks = []
                        mask_shapes = []
                        for mask in masks:
                            mask_shapes.append(mask.shape)  # Save original shape
                            flattened_masks.append(mask.ravel())  # Flatten each mask

                        # Send masks with shape metadata
                        node.send_output(
                            "masks",
                            pa.array(flattened_masks),  # Send array of flattened masks
                            metadata={
                                "frame_id": frame_idx,
                                "width": width,
                                "height": height,
                                "mask_shapes": mask_shapes  # Include shape information
                            }
                        )
                        

                # Convert back to BGR for visualization if needed
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Flatten the frame for PyArrow
                flat_frame = vis_frame.ravel()

                # Send the image
                node.send_output(
                    "image",
                    pa.array(flat_frame),
                    metadata={
                        "frame_id": frame_idx,
                        "width": width,
                        "height": height,
                        "channels": 3,
                        "encoding": "bgr8"
                    }
                )
            
            elif "boxes" in event_id:
                # Initialize tracking with bounding boxes
                if isinstance(event["value"], pa.StructArray):
                    boxes = event["value"][0].get("bbox").values.to_numpy().reshape(-1, 4)
                else:
                    boxes = event["value"].to_numpy().reshape(-1, 4)
                
                metadata = event["metadata"]
                frame_id = metadata.get("frame_id", "current")
                
                # if frame_id in frames:
                #     frame = frames[frame_id]
                #     width = frames[frame_id]["width"]
                #     height = frames[frame_id]["height"]
                    
                # Initialize tracking for each box
                for box in boxes:
                    obj_id = tracker.add_object(frame, box)
                    
                    # # Process the frame to get initial tracking results
                    # frame_idx, results = tracker.process_frame(frame_data)
                    
                    # if results:
                    #     # Prepare bounding boxes and masks for output
                    #     bboxes = []
                    #     masks = []
                    #     for result in results:
                    #         bboxes.append(result["box"])
                    #         masks.append(result["mask"])
                        
                    #     # Send bounding boxes
                    #     node.send_output(
                    #         "bbox",
                    #         pa.array(bboxes),
                    #         metadata={
                    #             "frame_id": frame_id,
                    #             "width": width, 
                    #             "height": height
                    #         }
                    #     )
                        
                    #     # Send masks
                    #     node.send_output(
                    #         "masks",
                    #         pa.array(masks),
                    #         metadata={
                    #             "frame_id": frame_id,
                    #             "width": width,
                    #             "height": height
                    #         }
                    #     )
                else:
                    print(f"Warning: Received boxes for unknown frame_id: {frame_id}")
        
        elif event_type == "ERROR":
            print(f"Error received: {event['error']}")


def main():
    """Entry point for the module."""
    run()


if __name__ == "__main__":
    main()