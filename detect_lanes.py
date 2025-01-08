import cv2
import json

# List to store the clicked points
points = []

# Mouse callback function to capture clicked points
def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed_point = points.pop()
            print(f"Point removed: {removed_point}")

def capture_polygons(video_path, output_json):
    global points
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read video.")
        return

    # Create a window and set mouse callback
    cv2.namedWindow("Define Polygons")
    cv2.setMouseCallback("Define Polygons", draw_polygon)

    print("Left-click to add points, right-click to remove the last point. Press 'n' for the next lane.")
    print("Press 'q' to finish and save polygons.")

    polygons = {}
    lane_id = 1

    while True:
        # Show the frame with the drawn points
        temp_frame = frame.copy()
        for i in range(len(points)):
            cv2.circle(temp_frame, points[i], 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(temp_frame, points[i - 1], points[i], (255, 0, 0), 2)
        if len(points) > 1:
            cv2.line(temp_frame, points[-1], points[0], (255, 0, 0), 2)

        cv2.imshow("Define Polygons", temp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Press 'n' to save the current polygon and start a new one
            if len(points) > 2:  # Ensure it's a valid polygon
                polygons[f"lane_{lane_id}"] = points[:]
                print(f"Polygon for lane {lane_id} saved: {points}")
                points = []
                lane_id += 1
            else:
                print("A valid polygon needs at least 3 points.")
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the polygons to a JSON file
    with open(output_json, 'w') as f:
        json.dump(polygons, f, indent=4)
    print(f"Polygons saved to {output_json}")

# Usage
video_path = "sunny_road.mp4"  # Path to your video
output_json = "lane_polygons.json"  # File to save the polygons
capture_polygons(video_path, output_json)