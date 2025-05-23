import json
import re
import argparse
from sklearn.metrics import precision_score, recall_score

if __name__ == "__main__":
# Load JSON data
    parser = argparse.ArgumentParser(description="A simple Python script with arguments.")
    parser.add_argument("-r", "--result", type=str, help="result path", required=True)
    args = parser.parse_args()

    file_path = args.result
    print(file_path)
    # file_path = "/mnt/afs/machengqian/mllm/qwen2vl/results/lfw_qwen2_5vl_7b_grpo_0311_low_kl_cot_v1_t_0.json"
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize counters
    mapping = {"Yes": 1, "No": 0}
    correct_predictions = 0
    total_predictions = len(data)
    format_predicitions = 0
    ground_truths = []
    predictions = []
    unformat_count = 0
    # Calculate accuracy
    for key, values in data.items():
        try:
            value = values[0]
            # if len(value[value.rfind("{") : value.rfind("}") + 1]) < 2:
            #     # print(key)
            #     continue
            # prediction = json.loads(value[value.rfind("{") : value.rfind("}") + 1])["Answer"]  # Extract ground truth
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", value, re.DOTALL)

            if match:
                answer = match.group(1).strip()  # Extract and remove extra spaces/newlines
                # print("Extracted Answer:", answer)
                if "Yes" in answer:
                    prediction = "Yes"
                else:
                    prediction = "No"
                format_predicitions += 1
            else:
                # print("No <answer> tag found.")
                unformat_count += 1
                continue

            ground_truth = values[1]
            ground_truths.append(mapping[ground_truth])
            predictions.append(mapping[prediction])                         # Extract prediction
            if ground_truth == prediction:                 # Compare values
                correct_predictions += 1
        except Exception as e:
            print(key)


    accuracy = (correct_predictions / total_predictions) * 100
    precision = precision_score(ground_truths, predictions) * 100
    recall = recall_score(ground_truths, predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"Unformatted: {unformat_count}")

    format_accuracy = (correct_predictions / format_predicitions) * 100
    print(f"Accuracy: {format_accuracy:.2f}%")