"""Dora node for video object segmentation using SAMURAI."""

import cv2
import numpy as np
import os
import pyarrow as pa
import sys
import torch
from dora import Node
from PIL import Image

# Add SAMURAI paths
current_dir = os.path.dirname(os.path.abspath(__file__))
samurai_path = os.path.join(current_dir, "..", "external", "samurai")
sam2_path = os.path.join(samurai_path, "sam2")

# Add to Python path
if samurai_path not in sys.path:
    sys.path.insert(0, samurai_path)
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

# Import SAMURAI components
from sam2.build_sam import build_sam2_video_predictor


def main():
    """Run a Dora node that processes video frames and bounding box inputs using SAMURAI."""
    pa.array([])  # initialize pyarrow array
    node = Node()
    
    # State tracking
    frames = {}
    frame_sequence = []
    tracked_objects = {}
    prediction_state = None
    predictor = None
    return_type = pa.Array
    
    # Initialize SAMURAI predictor
    model_name = "base_plus"  # Options: base_plus, t, s, l
    
    config_path = os.path.join(samurai_path, f"configs/samurai/sam2.1_hiera_b+.yaml")
    checkpoint_path = os.path.join(current_dir, "..", "checkpoints", f"sam2.1_hiera_base_plus.pt")
    
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if "image" in event_id:
                # Process incoming image
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]

                # Decode the image
                if encoding == "bgr8":
                    frame = storage.to_numpy().astype(np.uint8).reshape((height, width, 3))
                    frame = frame[:, :, ::-1]  # BGR to RGB
                elif encoding == "rgb8":
                    frame = storage.to_numpy().astype(np.uint8).reshape((height, width, 3))
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    storage = storage.to_numpy()
                    frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                    frame = frame[:, :, ::-1]  # BGR to RGB
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")
                
                # Store the frame
                frames[event_id] = frame
                frame_sequence.append(event_id)
                
                # Initialize SAMURAI if not already done
                if predictor is None:
                    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device="cuda:0")
                
                # If we have tracked objects and at least 2 frames, continue tracking
                if tracked_objects and len(frame_sequence) > 1:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        if prediction_state is None:
                            # Need to initialize state with frames
                            import tempfile
                            temp_dir = tempfile.mkdtemp()
                            try:
                                # Save frames to temp directory
                                for idx, frame_id in enumerate(frame_sequence):
                                    frame_path = os.path.join(temp_dir, f"{idx:08d}.jpg")
                                    Image.fromarray(frames[frame_id]).save(frame_path)
                                
                                # Initialize state
                                prediction_state = predictor.init_state(
                                    temp_dir, 
                                    offload_video_to_cpu=True, 
                                    offload_state_to_cpu=True,
                                    async_loading_frames=True
                                )
                                
                                # Add tracked objects
                                for obj_id, box in tracked_objects.items():
                                    _, _, _ = predictor.add_new_points_or_box(
                                        prediction_state, box=box, frame_idx=0, obj_id=obj_id
                                    )
                            finally:
                                import shutil
                                shutil.rmtree(temp_dir)
                        
                        # Process current frame
                        current_frame_idx = len(frame_sequence) - 1
                        frame_idx, object_ids, masks = predictor.get_current_predictions(
                            prediction_state, current_frame_idx
                        )
                        
                        # Convert masks to binary
                        binary_masks = []
                        for mask in masks:
                            mask = mask[0].cpu().numpy()
                            binary_mask = mask > 0.0
                            binary_masks.append(binary_mask)
                        
                        # Create combined mask if we have multiple objects
                        if binary_masks:
                            combined_mask = np.stack(binary_masks, axis=0)
                            
                            # Send output
                            if return_type == pa.Array:
                                node.send_output(
                                    "masks",
                                    pa.array(combined_mask.ravel()),
                                    metadata={
                                        "image_id": event_id,
                                        "width": width,
                                        "height": height,
                                    },
                                )
                            else:  # StructArray
                                node.send_output(
                                    "masks",
                                    pa.array(
                                        [
                                            {
                                                "masks": combined_mask.ravel(),
                                                "labels": [obj_id for obj_id in object_ids],
                                            },
                                        ],
                                    ),
                                    metadata={
                                        "image_id": event_id,
                                        "width": width,
                                        "height": height,
                                    },
                                )

            if "boxes2d" in event_id:
                # Process bounding box input
                if isinstance(event["value"], pa.StructArray):
                    boxes2d = event["value"][0].get("bbox").values.to_numpy()
                    labels = event["value"][0].get("labels").values.to_numpy(zero_copy_only=False)
                    return_type = pa.StructArray
                else:
                    boxes2d = event["value"].to_numpy()
                    labels = None
                    return_type = pa.Array

                metadata = event["metadata"]
                encoding = metadata["encoding"]
                if encoding != "xyxy":
                    raise RuntimeError(f"Unsupported boxes2d encoding: {encoding}")
                
                boxes2d = boxes2d.reshape(-1, 4)
                image_id = metadata["image_id"]
                
                # Store boxes for tracking
                for i, box in enumerate(boxes2d):
                    tracked_objects[i] = tuple(box)
                
                # Initialize predictor if not done yet
                if predictor is None:
                    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device="cuda:0")
                
                if image_id in frames:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        # Reset state for new objects
                        prediction_state = None
                        
                        # We need a temporary directory with the current frame
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        try:
                            # Save current frame
                            frame_path = os.path.join(temp_dir, "00000000.jpg")
                            Image.fromarray(frames[image_id]).save(frame_path)
                            
                            # Initialize state
                            prediction_state = predictor.init_state(
                                temp_dir, 
                                offload_video_to_cpu=True,
                                offload_state_to_cpu=True
                            )
                            
                            # Process all bounding boxes
                            binary_masks = []
                            for obj_id, box in tracked_objects.items():
                                _, _, masks = predictor.add_new_points_or_box(
                                    prediction_state, box=box, frame_idx=0, obj_id=obj_id
                                )
                                
                                # Convert mask to binary
                                mask = masks[0][0].cpu().numpy()
                                binary_mask = mask > 0.0
                                binary_masks.append(binary_mask)
                            
                            # Create combined mask
                            if binary_masks:
                                combined_mask = np.stack(binary_masks, axis=0)
                                
                                # Send output
                                if return_type == pa.Array:
                                    node.send_output(
                                        "masks",
                                        pa.array(combined_mask.ravel()),
                                        metadata={
                                            "image_id": image_id,
                                            "width": frames[image_id].shape[1],
                                            "height": frames[image_id].shape[0],
                                        },
                                    )
                                else:  # StructArray
                                    node.send_output(
                                        "masks",
                                        pa.array(
                                            [
                                                {
                                                    "masks": combined_mask.ravel(),
                                                    "labels": list(tracked_objects.keys()) if labels is None else labels,
                                                },
                                            ],
                                        ),
                                        metadata={
                                            "image_id": image_id,
                                            "width": frames[image_id].shape[1],
                                            "height": frames[image_id].shape[0],
                                        },
                                    )
                        finally:
                            import shutil
                            shutil.rmtree(temp_dir)

        elif event_type == "ERROR":
            print("Event Error:" + event["error"])


if __name__ == "__main__":
    main()