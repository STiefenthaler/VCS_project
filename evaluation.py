import ast
import argparse
import os

def process_files(video_name):
    video_name = str(video_name).split(".")[0]
    current_path = os.getcwd()
    folder_path = f"{current_path}/{video_name}/"

    with open(f"{current_path}/ground_truth/{video_name}_ground_truth.txt", 'r') as file:
        truth = ast.literal_eval(file.read())

    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file_path in all_files:
        with open(folder_path + file_path, "r") as file:
            print(f"Reading {file_path}...")
            content = file.read().strip(",")
            content = "{" + content + "}"
            detected = ast.literal_eval(content)

        detected_differences = {}
        truth_differences = {}

        previous_data = None
        for timestamp, current_data in detected.items():
            diff = {key: {subkey: current_data[key].get(subkey, 0) - (previous_data[key].get(subkey, 0) if previous_data else 0)
                          for subkey in set(current_data[key]) | (set(previous_data[key]) if previous_data else set())}
                    for key in ['incoming', 'outgoing']}
            detected_differences[timestamp] = diff
            previous_data = current_data

        previous_data = None
        for timestamp, current_data in truth.items():
            diff = {key: {subkey: current_data[key].get(subkey, 0) - (previous_data[key].get(subkey, 0) if previous_data else 0)
                          for subkey in set(current_data[key]) | (set(previous_data[key]) if previous_data else set())}
                    for key in ['incoming', 'outgoing']}
            truth_differences[timestamp] = diff
            previous_data = current_data

        aggregated_truth_differences = {}
        aggregated_detected_differences = {}
                
        for timestamp, values in truth_differences.items():
            # Somma tutti i valori in "incoming" e "outgoing" per il timestamp corrente
            incoming_sum = sum(values["incoming"].values())
            outgoing_sum = sum(values["outgoing"].values())
            
            # Aggiungi i risultati aggregati al dizionario
            aggregated_truth_differences[timestamp] = {
                "incoming": incoming_sum,
                "outgoing": outgoing_sum
            }
        
        for timestamp, values in detected_differences.items():
            # Somma tutti i valori in "incoming" e "outgoing" per il timestamp corrente
            incoming_sum = sum(values["incoming"].values())
            outgoing_sum = sum(values["outgoing"].values())
            
            # Aggiungi i risultati aggregati al dizionario
            aggregated_detected_differences[timestamp] = {
                "incoming": incoming_sum,
                "outgoing": outgoing_sum
            }
 
        TP, FP, FN = 0, 0, 0
        # Calcolo delle metriche TP, FP, FN
        for timestamp in sorted(set(aggregated_detected_differences) | set(aggregated_truth_differences)):
            detected_diff = aggregated_detected_differences.get(timestamp, {'incoming': 0, 'outgoing': 0})
            truth_diff = aggregated_truth_differences.get(timestamp, {'incoming': 0, 'outgoing': 0})
            
            for key in ['incoming', 'outgoing']:
                detected_count = detected_diff.get(key, 0)
                truth_count = truth_diff.get(key, 0)

                # Calculate TP, FP, FN
                TP += min(detected_count, truth_count)
                FP += max(0, detected_count - truth_count)
                FN += max(0, truth_count - detected_count)

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Scrittura dei risultati nel file di valutazione
        with open(current_path + "/evaluation/" + str(file_path).replace("detected_elements", "evaluation"), 'a') as file:
            file.write(f"Total True Positives (TP): {TP}\n")
            file.write(f"Total False Positives (FP): {FP}\n")
            file.write(f"Total False Negatives (FN): {FN}\n")
            file.write(f"Aggregated Precision: {precision:.2f}\n")
            file.write(f"Aggregated Recall: {recall:.2f}\n")
            file.write(f"Aggregated F1-Score: {f1_score:.2f}\n\n")

        # Visualizzazione dei risultati
        print("Aggregated Evaluation:")
        print(f"Total True Positives (TP): {TP}")
        print(f"Total False Positives (FP): {FP}")
        print(f"Total False Negatives (FN): {FN}")
        print(f"Aggregated Precision: {precision:.2f}")
        print(f"Aggregated Recall: {recall:.2f}")
        print(f"Aggregated F1-Score: {f1_score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a video.")
    parser.add_argument(
        "--video", 
        required=True, 
        choices=['heavy_foggy_road.mp4', 'foggy_road.mp4', 'sunny_road.mp4', 'rainy_road.mp4'],
        help="Video file to process."
    )
    args = parser.parse_args()
    process_files(args.video)

if __name__ == "__main__":
    main()
