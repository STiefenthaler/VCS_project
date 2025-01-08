import cv2
import json
import numpy as np

# Load lane coordinates
lane_data = {
    "foggy_lane_1": [[124, 286], [91, 286], [56, 545], [180, 549]],
    "foggy_lane_2": [[127, 291], [155, 290], [303, 579], [191, 577]],
    "foggy_lane_3": [[202, 283], [349, 402], [353, 353], [233, 285]],
    "foggy_lane_4": [[237, 285], [258, 282], [357, 329], [350, 354]],
    "sunny_heavy_lane_1": [[234, 153], [6, 249], [140, 322], [280, 163]],
    "sunny_heavy_lane_2": [[295, 153],[158, 329],[253, 335],[314, 155]],
    "sunny_heavy_lane_3": [[337, 155],[369, 329],[464, 339],[359, 160]],
    "sunny_heavy_lane_4": [[362, 164], [476, 337], [637, 243], [438, 160]],
}

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

def visualize_lanes(video_path, lane_prefix):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Get the appropriate lanes
    relevant_lanes = {key: lane_data[key] for key in lane_data if key.startswith(lane_prefix)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw each lane polygon
        for lane_name, vertices in relevant_lanes.items():
            points = [tuple(coord) for coord in vertices]
            direction = lane_directions.get(lane_name, "unknown")

            # Draw the polygon
            cv2.polylines(frame, [cv2.convexHull(np.array(points))], isClosed=True, color=(0, 255, 0), thickness=2)

            # Label the lane with its direction
            center_x = sum(x for x, y in points) // len(points)
            center_y = sum(y for x, y in points) // len(points)
            cv2.putText(frame, f"{lane_name} ({direction})", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Lanes Visualization", frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


