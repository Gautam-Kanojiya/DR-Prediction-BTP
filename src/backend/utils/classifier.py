# Updated classifier.py to include macular edema prediction + DR prediction using edema info

import importlib, pkg_resources
import cirq
import sympy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
importlib.reload(pkg_resources)

def load_dataset(training_data_labels_path, testing_data_labels_path, index):
    """This function loads the dataset into the program"""
    print("TASK : Loading the Training Dataset")
    x_train = []
    training_data = pd.read_csv("src/dataset/training_data/training_data.csv")
    for i in range(413):
        data = np.array(training_data[i:i+1])
        data_resize = data.reshape(4,4)
        x_train.append(data_resize)

    x_train = np.array(x_train)

    y_train = []
    training_dataset_labels = pd.read_csv(training_data_labels_path)
    for i in range(len(training_dataset_labels)):
        grade = training_dataset_labels['retinopathy_grade'][i]
        val = 1 if(grade == index) else 0
        y_train.append(val)

    print("TASK : Loading the Test Dataset")
    x_test = []
    testing_data = pd.read_csv("src/dataset/testing_data/testing_data.csv")
    for i in range(103):
        data = np.array(testing_data[i:i+1])
        data_resize = data.reshape(4,4)
        x_test.append(data_resize)
    x_test = np.array(x_test)

    y_test = []
    testing_data_labels = pd.read_csv(testing_data_labels_path)
    for i in range(len(testing_data_labels)):
        grade = testing_data_labels['retinopathy_grade'][i]
        val = 1 if(grade == index) else 0
        y_test.append(val)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# ----------------------------- Dataset Loading ----------------------------- #
def load_edema_dataset(training_labels_path, testing_labels_path, severity_index):
    x_train = []
    training_data = pd.read_csv("src/dataset/training_data/training_data.csv")
    for i in range(413):
        data = np.array(training_data[i:i+1]).reshape(4,4)
        x_train.append(data)

    y_train = []
    labels = pd.read_csv(training_labels_path)
    for grade in labels['risk_of_macular_edema']:
        y_train.append(1 if grade == severity_index else 0)

    x_test = []
    test_data = pd.read_csv("src/dataset/testing_data/testing_data.csv")
    for i in range(103):
        data = np.array(test_data[i:i+1]).reshape(4,4)
        x_test.append(data)

    y_test = []
    labels = pd.read_csv(testing_labels_path)
    for grade in labels['risk_of_macular_edema']:
        y_test.append(1 if grade == severity_index else 0)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# ----------------------------- Quantum Circuit Utilities ----------------------------- #
def convert_to_circuit(image):
    values = image.flatten()
    size = image.shape
    qubits = cirq.GridQubit.rect(*size)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def quantum_circuit(THRESHOLD, data):
    data = np.array(data)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    binary = np.array(data > THRESHOLD, dtype=np.float32)
    circuits = [convert_to_circuit(img) for img in binary]
    return tfq.convert_to_tensor(circuits)

# ----------------------------- Model Builders ----------------------------- #
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_macular_quantum_model(index):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit([cirq.X(readout), cirq.H(readout)])
    builder = CircuitLayerBuilder(data_qubits, readout)
    if index == 0:
        builder.add_layer(circuit, cirq.XX, "xx1")
        builder.add_layer(circuit, cirq.ZZ, "zz1")
    elif index == 1:
        builder.add_layer(circuit, cirq.XX, "xx2")
    elif index == 2:
        builder.add_layer(circuit, cirq.XX, "xx3")
        builder.add_layer(circuit, cirq.YY, "yy3")
    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)

def create_quantum_model(index):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit([cirq.X(readout), cirq.H(readout)])
    builder = CircuitLayerBuilder(data_qubits, readout)
    if index == 0:
        builder.add_layer(circuit, cirq.XX, "xx1")
    elif index == 1:
        builder.add_layer(circuit, cirq.XX, "xx2")
    elif index == 2:
        builder.add_layer(circuit, cirq.XX, "xx3")
        builder.add_layer(circuit, cirq.ZZ, "zz3")
    elif index == 3:
        builder.add_layer(circuit, cirq.XX, "xx4")
    elif index == 4:
        builder.add_layer(circuit, cirq.XX, "xx5")
    circuit.append(cirq.H(readout))
    return circuit, cirq.Z(readout)

def load_qnn_model(index):
    model_circuit, model_readout = create_quantum_model(index)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])

def load_macular_qnn_model(index):
    model_circuit, model_readout = create_macular_quantum_model(index)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])

# ----------------------------- Compilation and Training ----------------------------- #
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))

def compile_qnn_model(training_labels_path, testing_labels_path, THRESHOLD, validate, grade, edema_models):
    x_train, y_train, x_test, y_test = load_dataset(training_labels_path, testing_labels_path, grade)
    model = load_qnn_model(grade)
    model.compile(loss=tf.keras.losses.Hinge(), optimizer=tf.keras.optimizers.Adam(), metrics=[hinge_accuracy])

    def append_edema_predictions(images):
        new_images = []
        for img in images:
            preds = [model.predict(quantum_circuit(THRESHOLD, [img]))[0, 0] for model in edema_models]
            padded_preds = np.array([*preds, 0.0])
            new_row = padded_preds.reshape(1, 4)
            # Make it a 5x4 image by appending a new row
            new_img = np.vstack((img, new_row))
            new_images.append(new_img)
        return np.array(new_images)

    # Append edema predictions to training and test images
    x_train = append_edema_predictions(x_train)
    x_test = append_edema_predictions(x_test)

    x_train_q = quantum_circuit(THRESHOLD, x_train)
    x_test_q = quantum_circuit(THRESHOLD, x_test)
    y_train = 2.0 * y_train - 1.0
    y_test = 2.0 * y_test - 1.0

    model.fit(x_train_q, y_train, batch_size=32, epochs=15, validation_data=(x_test_q, y_test))
    return model

def compile_edema_qnn_model(training_labels_path, testing_labels_path, THRESHOLD, validate, severity):
    x_train, y_train, x_test, y_test = load_edema_dataset(training_labels_path, testing_labels_path, severity)
    model = load_macular_qnn_model(severity)
    model.compile(loss=tf.keras.losses.Hinge(), optimizer=tf.keras.optimizers.Adam(), metrics=[hinge_accuracy])

    x_train_q = quantum_circuit(THRESHOLD, x_train)
    x_test_q = quantum_circuit(THRESHOLD, x_test)
    y_train = 2.0 * y_train - 1.0
    y_test = 2.0 * y_test - 1.0

    model.fit(x_train_q, y_train, batch_size=32, epochs=15, validation_data=(x_test_q, y_test))
    return model

# ----------------------------- Classification ----------------------------- #
def classify_with_edema(THRESHOLD, image, edema_models, dr_models):
    preds = [model.predict(quantum_circuit(THRESHOLD, [image]))[0, 0] for model in edema_models]
    padded_preds = np.array([*preds, 0.0])
    new_row = padded_preds.reshape(1, 4)
    combined = np.vstack((image, new_row))
    tf_circuit = quantum_circuit(THRESHOLD, [combined])

    dr_scores = [model.predict(tf_circuit)[0,0] for model in dr_models]
    predicted_grade = int(np.argmax(dr_scores))
    return predicted_grade