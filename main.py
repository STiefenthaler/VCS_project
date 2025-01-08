from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# Load the YOLO models
Fine_tuned_model = YOLO('best.pt')
yolo8 = YOLO('yolov8m.pt')
yolo11 = YOLO('yolo11l.pt')


# Open the video files
heavy_fog_vid = cv2.VideoCapture('heavy_foggy_road.mp4')
medium_fog_vid = cv2.VideoCapture('foggy_road.mp4')
sunny_vid = cv2.VideoCapture('sunny_road.mp4')

# Assign directions to lanes
lane_directions = {
    "foggy_lane_1": "incoming",
    "foggy_lane_2": "incoming",
    "foggy_lane_3": "outgoing",
    "foggy_lane_4": "outgoing",
    "sunny_heavy_lane_1": "incoming",
    "sunny_heavy_lane_2": "incoming",
    "sunny_heavy_lane_3": "outgoing",
    "sunny_heavy_lane_4": "outgoing",
}

# Map video names to lane prefixes
video_to_lane_map = {
    "foggy_road.mp4": "foggy",
    "heavy_foggy_road.mp4": "sunny_heavy",
    "sunny_road.mp4": "sunny_heavy",
}

# Lane polygons for considering only vehicles within the lanes
lane_data = {
    "foggy_lane_1": [[124, 286], [91, 286], [56, 545], [180, 549]],
    "foggy_lane_2": [[127, 291], [155, 290], [303, 579], [191, 577]],
    "foggy_lane_3": [[202, 283], [349, 402], [353, 353], [233, 285]],
    "foggy_lane_4": [[237, 285], [258, 282], [357, 329], [350, 354]],
    "sunny_heavy_lane_1": [[234, 153], [6, 249], [140, 322], [280, 163]],
    "sunny_heavy_lane_2": [[295, 153], [158, 329], [253, 335], [314, 155]],
    "sunny_heavy_lane_3": [[337, 155], [369, 329], [464, 339], [359, 160]],
    "sunny_heavy_lane_4": [[362, 164], [476, 337], [637, 243], [438, 160]],
}

def get_relevant_lanes(video_name):
    prefix = video_to_lane_map.get(video_name, "")
    return {name: coords for name, coords in lane_data.items() if name.startswith(prefix)}


def detect_vehicles (model, cap, video_name): 
    class_list = model.names 
    # Open the video file

    line_y_red = 225  # Red line position

    # Dictionary to store object counts by class
    class_counts = defaultdict(int)

    # Dictionary to keep track of object IDs that have crossed the line
    crossed_ids = set()

    relevant_lanes = get_relevant_lanes(video_name)
    lane_polygons = [np.array(vertices, dtype=np.int32) for vertices in relevant_lanes.values()]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])  # Maintain only transportation-relevant classes

        # Ensure results are not empty
        if results and results[0].boxes.data is not None:
            # Safely access detected boxes, their class indices, and track IDs
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id
            class_indices = results[0].boxes.cls
            confidences = results[0].boxes.conf

            # Convert to usable formats if not None
            track_ids = track_ids.int().cpu().tolist() if track_ids is not None else []
            class_indices = class_indices.int().cpu().tolist() if class_indices is not None else []
            confidences = confidences.cpu().tolist() if confidences is not None else []

            # Draw the counting line
            cv2.line(frame, (0, line_y_red), (frame.shape[1], line_y_red), (0, 0, 255), 2)
            cv2.putText(frame, 'Counting line', (0, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            #darw the lane polygons
            for i, polygon in enumerate(lane_polygons):
                cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.putText(frame, f"Lane {i + 1}", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


            # Loop through each detected object
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2  # Calculate the center point
                cy = (y1 + y2) // 2

                class_name = class_list[class_idx]

                # Check if the object's center point is inside any lane polygon
                inside_lane = any(cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0 for polygon in lane_polygons)
                if inside_lane:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Check if the object has crossed the red line
                if cy > line_y_red and track_id not in crossed_ids:
                    # Mark the object as crossed
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

            # Display the counts on the frame
            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30

        else:
            print("No detections in this frame or no track IDs available.")

        # Show the frame
        cv2.imshow("YOLO Object Tracking & Counting", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

#final detection step
detect_vehicles(Fine_tuned_model, heavy_fog_vid, 'heavy_foggy_road.mp4')