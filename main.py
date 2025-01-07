from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load the YOLO models
Fine_tuned_model = YOLO('best.pt')
yolo8 = YOLO('yolov8m.pt')
yolo11 = YOLO('yolo11l.pt')



# Open the video files
heavy_fog_vid = cv2.VideoCapture('heavy_foggy_road.mp4')
medium_fog_vid = cv2.VideoCapture('foggy_road.mp4')
no_fog_vid = cv2.VideoCapture('sunny_road.mp4')

def detect_vehicles (model, cap): 
    class_list = model.names 
    # Open the video file
    cap = cv2.VideoCapture('heavy_foggy_road.mp4')

    line_y_red = 200  # Red line position

    # Dictionary to store object counts by class
    class_counts = defaultdict(int)

    # Dictionary to keep track of object IDs that have crossed the line
    crossed_ids = set()

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

            # Loop through each detected object
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2  # Calculate the center point
                cy = (y1 + y2) // 2

                class_name = class_list[class_idx]

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
detect_vehicles(yolo11, heavy_fog_vid)