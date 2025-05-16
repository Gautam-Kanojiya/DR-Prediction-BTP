## Diabetic Retinotherapy Classifier - Backend Testing
import pandas as pd
import numpy as np
from utils import classifier

if __name__ == "__main__":

    # Global Variables
    training_data_path = "src/dataset/training_data/images/"
    training_data_labels_path = "src/dataset/training_data/training_labels.csv"
    testing_data_path = "src/dataset/testing_data/images/"
    testing_data_labels_path = "src/dataset/testing_data/testing_labels.csv"
    THRESHOLD = 0.5
    create_model = False
    validate = False
    try:
        model_grade_0 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 0)
        model_grade_1 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 1)
        model_grade_2 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 2)
        model_grade_3 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 3)
        model_grade_4 = classifier.compile_qnn_model(training_data_labels_path, testing_data_labels_path, 0.1, validate, 4)
    except Exception as err:
        print(f"ERROR : {err}")

    y_test = []
    testing_data_labels = pd.read_csv(testing_data_labels_path)
    for i in range(len(testing_data_labels)):
        grade = testing_data_labels['retinopathy_grade'][i]
        val = grade
        y_test.append(val)

    y_pred = []
    n = 103
    for i in range(n):
        print(f"Image-{i}", end = " ")
        response = classifier.classify(0.5, i, model_grade_0, model_grade_1, model_grade_2, model_grade_3, model_grade_4)
        print(f"Actual Value: {y_test[i]}")
        predicted_grade = max(response, key=response.get)
        predicted_index = int(predicted_grade.split("_")[1])
        y_pred.append(predicted_index)
    print(y_pred)
    print(np.array(y_test)-np.array(y_pred))
