import json
import argparse
from crime_extraction import get_crime
from judge_extraction import calc_time_sum, calc_amt_sum
from law_extraction import get_penalcode_index_from_text

class MetricsCalculator:
    def __init__(self, gen_file, exp_file):
        self.gen_file = gen_file
        self.exp_file = exp_file
        self.gen_data = self.load_data(gen_file)
        self.exp_data = self.load_data(exp_file)
        
        # Initialize counters for metrics
        self.total_crime_rec = self.total_crime_prec = 0
        self.total_time_score = self.total_amount_score = 0
        self.total_penalcode_index_rec = self.total_penalcode_index_prec = 0
        self.time_num = self.amount_num = 0

        assert self.gen_data.keys() == self.exp_data.keys(), "Mismatch between gen_data and exp_data keys"
        self.n = len(self.exp_data)  # Total number of items in data

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            return {item['id']: item['document'] for item in (json.loads(line) for line in file)}

    def get_all_from_text(self, text):
        return get_crime(text), calc_time_sum(text), calc_amt_sum(text), get_penalcode_index_from_text(text)

    def calculate_recall_and_precision(self, expected, actual):
        expected_set = set(expected)
        actual_set = set(actual)
        true_positive = len(expected_set & actual_set)

        recall = true_positive / len(expected_set) if len(expected_set) > 0 else 0
        precision = true_positive / len(actual_set) if len(actual_set) > 0 else 0

        return recall, precision

    def calculate_percent_for_judge(self, exp_val, act_val):
        if exp_val == act_val == 0:
            return 1.0
        if (exp_val >= 0 and act_val) < 0 or (exp_val < 0 and act_val >= 0):  # Different signs
            return 0.0
        if (exp_val - 10000) * (act_val - 10000) < 0:  # Both must either have or lack the death penalty
            return 0.0
        x = abs(exp_val - act_val) / max(exp_val, act_val)
        y = 1 - x
        return y

    def calc_metrics(self):
        for exp_id, exp_ans in self.exp_data.items():
            gen_ans = self.gen_data[exp_id]

            exp_crime, exp_time, exp_amount, exp_penalcode_index = self.get_all_from_text(exp_ans)
            gen_crime, gen_time, gen_amount, gen_penalcode_index = self.get_all_from_text(gen_ans)

            crime_rec, crime_prec = self.calculate_recall_and_precision(exp_crime, gen_crime)
            penalcode_index_rec, penalcode_index_prec = self.calculate_recall_and_precision(exp_penalcode_index, gen_penalcode_index)

            # Accumulate the results
            self.total_crime_rec += crime_rec
            self.total_crime_prec += crime_prec
            self.total_penalcode_index_rec += penalcode_index_rec
            self.total_penalcode_index_prec += penalcode_index_prec

            if exp_time >= 0 or gen_time >= 0:
                time_score = self.calculate_percent_for_judge(exp_time, gen_time)
                self.total_time_score += time_score
                self.time_num += 1

            if exp_amount >= 0 or gen_amount >= 0:
                amount_score = self.calculate_percent_for_judge(exp_amount, gen_amount)
                self.total_amount_score += amount_score
                self.amount_num += 1

    def print_results(self):
        avg_crime_rec = self.total_crime_rec / self.n
        avg_crime_prec = self.total_crime_prec / self.n
        avg_penalcode_index_rec = self.total_penalcode_index_rec / self.n
        avg_penalcode_index_prec = self.total_penalcode_index_prec / self.n

        # Calculate F1 scores
        f1_crime = 2 * (avg_crime_prec * avg_crime_rec) / (avg_crime_prec + avg_crime_rec) if (avg_crime_prec + avg_crime_rec) != 0 else 0
        f1_penalcode_index = 2 * (avg_penalcode_index_prec * avg_penalcode_index_rec) / (avg_penalcode_index_prec + avg_penalcode_index_rec) if (avg_penalcode_index_prec + avg_penalcode_index_rec) != 0 else 0

        # Calculate average judge time score and average amount score
        avg_time_score = self.total_time_score / self.time_num if self.time_num > 0 else 0
        avg_amount_score = self.total_amount_score / self.amount_num if self.amount_num > 0 else 0

        # Print the results
        print(f"Average Judge Time Score: {avg_time_score:.4f}, Average Amount Score: {avg_amount_score:.4f}")
        print(f"Average Crime Recall: {avg_crime_rec:.4f}, Average Crime Precision: {avg_crime_prec:.4f}, F1 Score: {f1_crime:.4f}")
        print(f"Average Penalcode Index Recall: {avg_penalcode_index_rec:.4f}, Average Penalcode Index Precision: {avg_penalcode_index_prec:.4f}, F1 Score: {f1_penalcode_index:.4f}")
        print(self.time_num, self.amount_num)


def main():
    parser = argparse.ArgumentParser(description="Process a JSON file to calculate metrics.")
    parser.add_argument('--gen_file', type=str, required=True, help='Path to the input generated JSON file')
    parser.add_argument('--exp_file', type=str, required=True, help='Path to the expected JSON file')
    args = parser.parse_args()

    # Create an instance of MetricsCalculator
    calculator = MetricsCalculator(args.gen_file, args.exp_file)
    
    # Calculate the metrics
    calculator.calc_metrics()
    
    # Print the results
    calculator.print_results()
    print(f"This is the metrics from file {args.gen_file}!")


if __name__ == "__main__":
    main()
