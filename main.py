from ultralytics import YOLO
import argparse
import cv2
from collections import defaultdict
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# Assign directions to lanes
lane_directions = {
    "foggy_lane_1": "incoming",
    "foggy_lane_2": "incoming",
    "foggy_lane_3": "outgoing",
    "foggy_lane_4": "outgoing",
    "rainy_lane_1": "incoming",
    "rainy_lane_2": "incoming",
    "rainy_lane_3": "outgoing",
    "rainy_lane_4": "outgoing",
    "sunny_heavy_lane_1": "incoming",
    "sunny_heavy_lane_2": "incoming",
    "sunny_heavy_lane_3": "outgoing",
    "sunny_heavy_lane_4": "outgoing",
}

# Map video names to lane prefixes
video_to_lane_map = {
    "foggy_road.mp4": "foggy",
    "rainy_road.mp4": "rainy",
    "heavy_foggy_road.mp4": "sunny_heavy",
    "sunny_road.mp4": "sunny_heavy",
}

# Lane polygons for considering only vehicles within the lanes
lane_data = {
    "foggy_lane_1": [[124, 286], [91, 286], [56, 578], [180, 578]],
    "foggy_lane_2": [[127, 286], [155, 286], [303, 578], [191, 578]],
    "foggy_lane_3": [[202, 283], [349, 402], [353, 353], [233, 285]],
    "foggy_lane_4": [[237, 285], [258, 282], [357, 329], [350, 354]],
    "rainy_lane_1": [[37, 460],  [670, 261], [724, 276], [89, 509]],
    "rainy_lane_2": [[88, 509],  [763, 266], [810, 268], [124, 542]],
    "rainy_lane_3": [[384, 531], [881, 273], [929, 274], [529, 545]],
    "rainy_lane_4": [[543, 545], [930, 268], [992, 273], [700, 555]],
    "sunny_heavy_lane_1": [[234, 153], [6, 249], [140, 322], [280, 163]],
    "sunny_heavy_lane_2": [[290, 153], [140, 329], [253, 335], [314, 155]], 
    "sunny_heavy_lane_3": [[337, 155], [369, 329], [464, 339], [359, 160]],
    "sunny_heavy_lane_4": [[362, 164], [476, 337], [637, 243], [438, 160]],
}

def get_relevant_lanes(video_name):
    prefix = video_to_lane_map.get(video_name, "")
    return {name: coords for name, coords in lane_data.items() if name.startswith(prefix)}


# Define line positions for each video
video_line_positions = {
    "foggy_road.mp4": {"incoming_line_y": 470, "outgoing_line_y": 295},
    "heavy_foggy_road.mp4": {"incoming_line_y": 280, "outgoing_line_y": 190},
    "sunny_road.mp4": {"incoming_line_y": 280, "outgoing_line_y": 190},
    "rainy_road.mp4": {"incoming_line_y": 450, "outgoing_line_y": 320},
}

def get_line_positions(video_name):
    return video_line_positions.get(video_name, {"incoming_line_y": 300, "outgoing_line_y": 200})

def detect_vehicles(model, cap, video_name, tracker, model_name, tracker_name):
    """
    Processes the video using the specified YOLO model and applies vehicle detection.
    Args:
        model: The YOLO model used for detection.
        video_cap: Video capture object.
        video_name: Name of the video to use.
    """
    print(f"Processing video '{video_name}' with tracker '{tracker}'...")
    class_list = model.names
    paused = False

    # Get fps and frame_counter to find timestamps when model detects an object
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0

    # Get the line positions for the current video
    line_positions = get_line_positions(video_name)
    incoming_line_y = line_positions["incoming_line_y"]
    outgoing_line_y = line_positions["outgoing_line_y"]

    # Dictionaries to store counts by class and direction
    class_counts = {"incoming": defaultdict(int), "outgoing": defaultdict(int)}

    # Set to keep track of object IDs that have crossed the lines
    crossed_ids = {"incoming": set(), "outgoing": set()}

    relevant_lanes = get_relevant_lanes(video_name)
    lane_polygons = {name: np.array(coords, dtype=np.int32) for name, coords in relevant_lanes.items()}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the timestamp using the frame count and fps
        frame_counter += 1
        print("frame: " + str(frame_counter))
        current_time_sec = frame_counter / fps
        current_minute = int(current_time_sec // 60)
        current_second = int(current_time_sec % 60)
        current_millisecond = int((current_time_sec % 1) * 1000)
        detected_elements = {}

         # Display the time on the frame
        time_text = f"Time: {current_minute:02}:{current_second:02}.{current_millisecond:03}"
        print(time_text)
        formatted_time = f"{current_minute:02}:{current_second:02}.{current_millisecond:03}"

        
        print(tracker)

        if tracker == "botsort":
            results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7], tracker="botsort.yaml")
        elif tracker == "bytetrack":
            results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7], tracker="bytetrack.yaml")
        else:
            results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])


        if results and results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id
            class_indices = results[0].boxes.cls
            confidences = results[0].boxes.conf

            track_ids = track_ids.int().cpu().tolist() if track_ids is not None else []
            class_indices = class_indices.int().cpu().tolist() if class_indices is not None else []
            confidences = confidences.cpu().tolist() if confidences is not None else []

            # Draw the incoming and outgoing lines
            cv2.line(frame, (0, incoming_line_y), (frame.shape[1], incoming_line_y), (0, 255, 0), 2)
            cv2.putText(frame, 'Incoming line', (0, incoming_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.line(frame, (0, outgoing_line_y), (frame.shape[1], outgoing_line_y), (255, 0, 0), 2)
            cv2.putText(frame, 'Outgoing line', (0, outgoing_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw the lane polygons
            for lane_name, polygon in lane_polygons.items():
                cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=1)
                #cv2.putText(frame, lane_name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Loop through each detected object
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2  # Calculate the center point
                cy = (y1 + y2) // 2

                class_name = class_list[class_idx]

                # Check if the object's center point is inside any lane polygon
                for lane_name, polygon in lane_polygons.items():
                    if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                        direction = lane_directions[lane_name]  # Determine the direction of the lane

                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Check if the object has crossed the appropriate line
                        if direction == "incoming" and cy > incoming_line_y and track_id not in crossed_ids["incoming"]:
                            crossed_ids["incoming"].add(track_id)
                            class_counts["incoming"][class_name] += 1
                        elif direction == "outgoing" and cy < outgoing_line_y and track_id not in crossed_ids["outgoing"]:
                            crossed_ids["outgoing"].add(track_id)
                            class_counts["outgoing"][class_name] += 1
                        
                        detected_elements[formatted_time] = dict(class_counts)


            # Display the counts on the frame
            y_offset = 30
            for direction, counts in class_counts.items():
                cv2.putText(frame, f"{direction.capitalize()}:", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y_offset += 30
                for class_name, count in counts.items():
                    cv2.putText(frame, f"{class_name}: {count}", (70, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30

        else:
            print("No detections in this frame or no track IDs available.")

        # Show the frame
        output.write(frame)
        cv2.imshow("YOLO Object Tracking & Counting", frame)

        folder_name = os.path.splitext(video_name)[0]
        os.makedirs(folder_name, exist_ok=True)

        # Construct the output file path
        output_file = os.path.join(
            folder_name,
            f"{os.path.splitext(model_name)[0]}_{os.path.splitext(video_name)[0]}_{os.path.splitext(tracker_name)[0]}_detected_elements.txt"
        )

        
        # write detected elements file
        with open(output_file, "a") as file:
            for key, value in detected_elements.items():
                timestamp = list(detected_elements.keys())[0]
                formatted_output = "'" + str(timestamp) +"' : {'incoming': " +str(dict(detected_elements[timestamp]['incoming'])) + ", 'outgoing': " + str(dict(detected_elements[timestamp]['outgoing'])) + "},"
                file.write(f"{formatted_output}\n")


        # Exit video if 'q' key is pressed
        # Pause video if 'p' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('p'):  
            paused = not paused
            while paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):  # Riprendi
                    paused = False
                elif key == ord('q'):  # Esci
                    paused = False
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    # Release resources
    cap.release()
    output.release()  
    cv2.destroyAllWindows()

def main():
    """
    Main method to run the detect_vehicles function with arguments specified via CLI.
    --model to specify the YOLO model to use.
    --video to specify the video file to process.
    --tracker to optionally specify the tracking method.
    """
    # Define the CLI arguments
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a video.")
    parser.add_argument(
        "--model", 
        required=True, 
        choices=['yolov8_finetuned.pt', 'yolov8m.pt', 'yolo11l.pt'],
        help="Path to the YOLO model to use for detection."
    )
    parser.add_argument(
        "--video", 
        required=True, 
        choices=['heavy_foggy_road.mp4', 'foggy_road.mp4', 'sunny_road.mp4', 'rainy_road.mp4'],
        help="Video file to process."
    )

    parser.add_argument(
        "--tracker", 
        default="yolo", 
        choices=["yolo", "bytetrack", "botsort"],
        help="Specify the tracking method to use ('yolo', 'botsort', or 'bytetrack'). Defaults to 'yolo'."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the specified YOLO model & tracker
    model = YOLO(args.model)
    tracker=args.tracker

    # Open the video file
    video_cap = cv2.VideoCapture(args.video)
    if not video_cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {args.video}")

    # Call detect_vehicles
    detect_vehicles(model, video_cap, args.video, tracker, args.model, args.tracker)

if __name__ == "__main__":
    main()
