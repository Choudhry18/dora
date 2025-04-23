"""Generate fixed bounding boxes for testing SAMURAI."""

import numpy as np
import pyarrow as pa
from dora import Node

def main():
    """Run a Dora node that generates fixed bounding boxes."""
    pa.array([])  # initialize pyarrow array
    node = Node("fixed_boxes")
    
    # Get config
    config = node.config if hasattr(node, 'config') else {}
    boxes = config.get('boxes', [[300, 100, 50, 200]])
    boxes_array = np.array(boxes, dtype=np.float32)
    
    # Flag to track if we've sent the box already
    box_sent = False
    
    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            
            if "image" in event_id and not box_sent:
                # Send fixed boxes with reference to the image (only once)
                node.send_output(
                    "boxes2d",
                    pa.array(boxes_array.ravel()),
                    metadata={
                        "image_id": event_id,
                        "encoding": "xyxy",
                    },
                )
                # Mark that we've sent the box
                box_sent = True

if __name__ == "__main__":
    main()