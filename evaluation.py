import ast
import argparse
import os

def process_files(video_name):

    video_name = str(video_name).split(".")[0]

    # Define the video folder path
    current_path = os.getcwd()
    folder_path = f"{current_path}/{video_name}/"


    # Read ground truth file
    with open(f"{current_path}/ground_truth/{video_name}_ground_truth.txt", 'r') as file:
        truth = ast.literal_eval(file.read()) #transform file content into dictionary

    # Read all files in video folder
    # Combine folder path and pattern to get matching files
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


    # Iterate through the files and read their content
    for file_path in all_files:
        content = ""
        with open(folder_path + file_path, "r") as file:
            print(f"Reading {file_path}...")
            content = file.read()
            content = content.strip(",") #remove last comma
            content = "{" + content + "}"

            detected = ast.literal_eval(content) #transform file content into dictionary
        
        #Find how many elements are detected at each time stamp
        detected_differences = {}
        truth_differences = {}

        previous_data = None
        for timestamp, current_data in detected.items():
            if previous_data is None:
                diff = {
                    'incoming': {key: current_data['incoming'].get(key, 0) for key in current_data['incoming']},
                    'outgoing': {key: current_data['outgoing'].get(key, 0) for key in current_data['outgoing']}
                }

            else:
                diff = {
                    'incoming': {key: current_data['incoming'].get(key, 0) - previous_data['incoming'].get(key, 0)
                                for key in set(current_data['incoming']) | set(previous_data['incoming'])},
                    'outgoing': {key: current_data['outgoing'].get(key, 0) - previous_data['outgoing'].get(key, 0)
                                for key in set(current_data['outgoing']) | set(previous_data['outgoing'])}
                }
            detected_differences[timestamp] = diff
            previous_data = current_data

        previous_data = None
        for timestamp, current_data in truth.items():
            if previous_data is None:
                diff = {
                    'incoming': {key: current_data['incoming'].get(key, 0) for key in current_data['incoming']},
                    'outgoing': {key: current_data['outgoing'].get(key, 0) for key in current_data['outgoing']}
                }
            else:
                diff = {
                    'incoming': {key: current_data['incoming'].get(key, 0) - previous_data['incoming'].get(key, 0)
                                for key in set(current_data['incoming']) | set(previous_data['incoming'])},
                    'outgoing': {key: current_data['outgoing'].get(key, 0) - previous_data['outgoing'].get(key, 0)
                                for key in set(current_data['outgoing']) | set(previous_data['outgoing'])}
                }
            truth_differences[timestamp] = diff
            previous_data = current_data


        # Find total number of cars found by model and in ground truth to check if True Positive, False Positive and False negative are correct.
        total_truth = truth[list(truth.keys())[-1]]
        for key in total_truth.keys():
            print(key)
            for subkey in total_truth[key]:
                print(subkey)

                # Compute TP, FP, FN
                TP, FP, FN = 0, 0, 0
        
                for timestamp in sorted(set(detected_differences) | set(truth_differences)):
                    detected_diff = detected_differences.get(timestamp, {key:{subkey:0}})
                    truth_diff = truth_differences.get(timestamp, {key:{subkey:0}})

                    if subkey in detected_diff[key] or subkey in truth_diff[key]:
                        detected_count = detected_diff[key].get(subkey, 0)
                        truth_count = truth_diff[key].get(subkey, 0)

                        TP += min(detected_count, truth_count)
                        FP += max(0, (detected_count - truth_count))
                        FN += max(0, (truth_count - detected_count))

                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                try:
                    total_truth_count = truth[list(truth.keys())[-1]][key][subkey]
                except KeyError:
                    total_truth_count = 0
                try:
                    total_detected_count = detected[list(detected.keys())[-1]][key][subkey]
                except KeyError:
                    total_detected_count = 0

                if TP + FN == total_truth_count and TP + FP == total_detected_count:
                    with open(current_path + "/evaluation/" + str(file_path).replace("detected_elements","evaluation"), 'a') as file:

                        file.write(f"Total truth for {key} {subkey}: {total_truth_count}\n")
                        file.write(f"Total detected for {key} {subkey}: {total_detected_count}\n")
                        file.write(f"True Positives (TP) for {key} {subkey}: {TP}\n")
                        file.write(f"False Positives (FP) for {key} {subkey}: {FP}\n")
                        file.write(f"False Negatives (FN) for {key} {subkey}: {FN}\n")
                        file.write(f"Precision for {key} {subkey}: {precision:.2f}\n")
                        file.write(f"Recall for {key} {subkey}: {recall:.2f}\n")
                        file.write(f"F1-Score for {key} {subkey}: {f1_score:.2f}\n\n")

                        print("TP, FP OR FN CORRECTLY COMPUTED\n")
                        print(f"Total truth for {key} {subkey}: {total_truth_count}")
                        print(f"Total detected for {key} {subkey}: {total_detected_count}")
                        print(f"True Positives (TP) for {key} {subkey}: {TP}")
                        print(f"False Positives (FP) for {key} {subkey}: {FP}")
                        print(f"False Negatives (FN) for {key} {subkey}: {FN}")
                        print(f"Precision for {key} {subkey}: {precision:.2f}")
                        print(f"Recall for {key} {subkey}: {recall:.2f}")
                        print(f"F1-Score for {key} {subkey}: {f1_score:.2f}")

                else:
                    with open(current_path + "/evaluation/" + str(file_path).replace("detected_elements","evaluation_ERROR"), 'a') as file:

                        file.write(f"Total truth for {key} {subkey}: {total_truth_count}\n")
                        file.write(f"Total detected for {key} {subkey}: {total_detected_count}\n")
                        file.write(f"True Positives (TP) for {key} {subkey}: {TP}\n")
                        file.write(f"False Positives (FP) for {key} {subkey}: {FP}\n")
                        file.write(f"False Negatives (FN) for {key} {subkey}: {FN}\n\n")

                        print("ERROR WHEN COMPUTING TP, FP OR FN\n")
                        print(f"Total truth for {key} {subkey}: {total_truth_count}")
                        print(f"Total detected for {key} {subkey}: {total_detected_count}")
                        print(f"True Positives (TP) for {key} {subkey}: {TP}")
                        print(f"False Positives (FP) for {key} {subkey}: {FP}")
                        print(f"False Negatives (FN) for {key} {subkey}: {FN}")


def main():
    """
    Main method to evaluate model with argument specified via CLI.
    --video to specify the video file to process.
    """
    # Define the CLI arguments
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a video.")
    parser.add_argument(
        "--video", 
        required=True, 
        choices=['heavy_foggy_road.mp4', 'foggy_road.mp4', 'sunny_road.mp4', 'rainy_road.mp4'],
        help="Video file to process."
    )


    # Parse the arguments
    args = parser.parse_args()

    # Call detect_vehicles
    process_files(args.video)

if __name__ == "__main__":
    main()
                
       