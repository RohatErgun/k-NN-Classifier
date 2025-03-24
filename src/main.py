import math
import sys
from collections import Counter
import os


def knn_classifier(k, train_set, test_set):
    count_prediction = 0

    for test_point, test_label in test_set:
        dist_list = []

        for train_point, train_label in train_set:
            dis = calculate_distance(test_point, train_point)
            dist_list.append((dis, train_label))

        dist_list.sort()
        k_nearest_labels = [label for _, label in dist_list[:k]]

        prediction_label = Counter(k_nearest_labels).most_common(1)[0][0]

        if prediction_label == test_label:
            count_prediction += 1

    return count_prediction / len(test_set)


def classify_single_vector(k, train_set, test_vector):
    dist_list = []

    for train_point, train_label in train_set:
        dis = calculate_distance(test_vector, train_point)
        dist_list.append((dis, train_label))

    dist_list.sort()
    k_nearest_labels = [label for _, label in dist_list[:k]]

    return Counter(k_nearest_labels).most_common(1)[0][0]


def calculate_distance(point1, point2):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(point1, point2)))


def load_files(train_file, test_file):
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("File not found")
        return None, None

    train_set = []
    test_set = []

    with open(train_file, 'r') as f:
        for i in f:
            data = i.strip().split(',')

            train_set.append((list(map(float, data[:-1])), data[-1]))

    with open(test_file, 'r') as f:
        for i in f:
            data = i.strip().split(',')
            test_set.append((list(map(float, data[:-1])), data[-1]))

    return train_set, test_set


def load_train_file(train_file):
    if not os.path.exists(train_file):
        print("File not found")
        return None

    train_set = []
    with open(train_file, 'r') as f:
        for i in f:
            data = i.strip().split(',')
            train_set.append((list(map(float, data[:-1])), data[-1]))

    return train_set


def user_input(num):
    print("Enter Vector separated with commas")
    while True:
        try:
            user_in = input(">>  ").strip()
            input_vector = list(map(float, user_in.split(',')))

            if len(input_vector) != num:
                print(f"expected {num}")
                continue

            return input_vector
        except ValueError:
            print("Values must be numbers!")


if __name__ == "__main__":

    print("Enter Train_file name: ")
    train_file = input(": ")

    print("Enter Test_file name or enter 'u' for user input:")
    test_file_or_userInput = input(": ")

    print("Enter K value: ")
    k = int(input(": "))

    if test_file_or_userInput == "userInput":
        train_set = load_train_file(train_file)
        if not train_set:
            print("Training file is empty.")
            sys.exit(1)
        nums = len(train_set[0][0])
        vec = user_input(nums)
        prediction = classify_single_vector(k, train_set, vec)

        print(f"\nPredicted class: {prediction}")

    else:
        train_set, test_set = load_files(train_file,
                                         test_file_or_userInput if test_file_or_userInput != "userInput" else None)
        if not train_set or not test_set:
            print("Error: Could not load data.")
            sys.exit(1)

        accuracy = knn_classifier(k, train_set, test_set)
        print(f"Accuracy: {accuracy * 100:.2f}%")
