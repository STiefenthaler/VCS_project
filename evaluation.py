from collections import defaultdict
import ast

def process_files(detected_file, ground_truth_file, output_prefix, chosen_key):
    # Read model results file
    with open(detected_file, 'r') as file:
        detected = ast.literal_eval(file.read())
    
    # Read ground truth file
    with open(ground_truth_file, 'r') as file:
        truth = ast.literal_eval(file.read())

    # Ensure that if a key in truth does not exist in detected for a specific timestamp, it is treated as 0 in detected
    for timestamp in detected:
        if timestamp in truth:
            for key in truth[timestamp]:
                if key not in detected[timestamp]:
                    detected[timestamp][key] = 0

    # Find how many elements are detected at each time stamp
    model_differences = {}
    truth_differences = {}

    previous_data = None
    for timestamp, current_data in detected.items():
        if previous_data is None:
            diff = {key: current_data.get(key, 0) for key in set(current_data)}
        else:
            diff = {key: current_data.get(key, 0) - previous_data.get(key, 0)
                    for key in set(current_data) | set(previous_data)}
        model_differences[timestamp] = diff
        previous_data = current_data


    previous_data = None
    for timestamp, current_data in truth.items():
        if previous_data is None:
            diff = {key: current_data.get(key, 0) for key in set(current_data)}
        else:
            diff = {key: current_data.get(key, 0) - previous_data.get(key, 0)
                    for key in set(current_data) | set(previous_data)}
        truth_differences[timestamp] = diff
        previous_data = current_data
    
    # Save differences in file, side by side
    with open(f'{output_prefix}_differences.txt', 'w') as file:
        for timestamp in sorted(set(model_differences) | set(truth_differences)):
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Detected Difference: {model_differences.get(timestamp, {})}\n")
            file.write(f"Truth Difference: {truth_differences.get(timestamp, {})}\n\n")
    
    # Find total number of cars found by model and in ground truth to check if True Positive, False Positive and False negative are correct.
    total_truth = sum(truth[list(truth.keys())[-1]].values())
    total_detected = sum(detected[list(detected.keys())[-1]].values())

    with open(f'{output_prefix}_totals.txt', 'w') as file:
        file.write(f"Totale in ground truth (ultimo record): {total_truth}\n")
        file.write(f"Totale in detected (ultimo record): {total_detected}\n")

    # Compute TP, FP, FN
    TP, FP, FN = 0, 0, 0
    total_truth_cars = truth[list(truth.keys())[-1]][chosen_key]
    total_detected_cars = detected[list(detected.keys())[-1]][chosen_key]
    
    for timestamp in sorted(set(model_differences) | set(truth_differences)):
        detected_diff = model_differences.get(timestamp, {})
        truth_diff = truth_differences.get(timestamp, {})

        ##################
         # Calcoliamo la somma totale delle categorie
        total_detected = sum(detected_diff.values())
        total_truth = sum(truth_diff.values())
        #############

        if chosen_key in detected_diff or chosen_key in truth_diff:
            detected_count = detected_diff.get(chosen_key, 0)
            truth_count = truth_diff.get(chosen_key, 0)

            TP += min(detected_count, truth_count)
            FP += max(0, detected_count - truth_count)
            FN += max(0, truth_count - detected_count)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if TP + FN == total_truth_cars and TP + FP == total_detected_cars:
        with open(f'{output_prefix}_{key}_evaluation.txt', 'w') as file:

            file.write(f"Total truth cars: {total_truth_cars}\n")
            file.write(f"Total detected cars: {total_detected_cars}\n")
            file.write(f"True Positives (TP) for {chosen_key}: {TP}\n")
            file.write(f"False Positives (FP) for {chosen_key}: {FP}\n")
            file.write(f"False Negatives (FN) for {chosen_key}: {FN}\n")
            file.write(f"Precision for {chosen_key}: {precision:.2f}\n")
            file.write(f"Recall for {chosen_key}: {recall:.2f}\n")
            file.write(f"F1-Score for {chosen_key}: {f1_score:.2f}\n")

            print("TP, FP OR FN CORRECTLY COMPUTED\n")
            print(f"{output_prefix} {chosen_key}".upper()+"\n")
            print(f"Total truth cars: {total_truth_cars}\n")
            print(f"Total detected cars: {total_detected_cars}\n")
            print(f"True Positives (TP) for {chosen_key}: {TP}\n")
            print(f"False Positives (FP) for {chosen_key}: {FP}\n")
            print(f"False Negatives (FN) for {chosen_key}: {FN}\n")
            print(f"Precision for {chosen_key}: {precision:.2f}\n")
            print(f"Recall for {chosen_key}: {recall:.2f}\n")
            print(f"F1-Score for {chosen_key}: {f1_score:.2f}\n")

    else:
        with open(f'{output_prefix}_evaluation_ERROR.txt', 'w') as file:

            file.write(f"Total truth cars: {total_truth_cars}\n")
            file.write(f"Total detected cars: {total_detected_cars}\n")
            file.write(f"True Positives (TP) for {chosen_key}: {TP}\n")
            file.write(f"False Positives (FP) for {chosen_key}: {FP}\n")
            file.write(f"False Negatives (FN) for {chosen_key}: {FN}\n")

            print("ERROR WHEN COMPUTING TP, FP OR FN\n")
            print(f"{output_prefix} {chosen_key}".upper() +"\n")
            print(f"Total truth cars: {total_truth_cars}\n")
            print(f"Total detected cars: {total_detected_cars}\n")
            print(f"True Positives (TP) for {chosen_key}: {TP}\n")
            print(f"False Positives (FP) for {chosen_key}: {FP}\n")
            print(f"False Negatives (FN) for {chosen_key}: {FN}\n")


# Process both files
process_files('sunny_road_detected_elements.txt', 'sunny_road_ground_truth.txt', 'sunny', 'car')
process_files('foggy_road_detected_elements.txt', 'foggy_road_ground_truth.txt', 'foggy', 'incoming')
process_files('foggy_road_detected_elements.txt', 'foggy_road_ground_truth.txt', 'foggy', 'outgoing')
                
       