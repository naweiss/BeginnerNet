from math_functions import sigmoid, dsigmoid
from math import exp
import random

def activation_function(x, derivative=False):
    if derivative:
        return dsigmoid(x)
    else:
        return sigmoid(x)


class Matrix:
    def __init__(self, data):
        # init vals
        self.rows = 0
        self.cols = 0
        self.data = []

        # assign vals
        self.data = data
        self.rows = len(data)
        general_row = data[0]
        if type(general_row) is not list:
            self.data = [self.data]
            self.rows = 1
            self.cols = len(self.data[0])
        else:
            self.cols = len(general_row)

    def transpose(self):
        new_data = []
        for i in range(self.cols):
            new_row = []
            for row in self.data:
                new_row.append(row[i])
            new_data.append(new_row)
        return Matrix(new_data)

    def sigmoid(self, derivative=False):
        new_mat = []
        if derivative:
            for i in range(self.rows):
                new_row = []
                for j in range(self.cols):
                    x = self.data[i][j]
                    new_row.append(activation_function(x, derivative=True))
                new_mat.append(new_row)
        else:
            for i in range(self.rows):
                new_row = []
                for j in range(self.cols):
                    x = self.data[i][j]
                    new_row.append(activation_function(x))
                new_mat.append(new_row)
        return Matrix(new_mat)

    def scalar_mul(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise Exception(
                "both matrices are not the same size [{},{}] * [{},{}]".format(self.rows, self.cols, other.rows,
                                                                               other.cols))
        new_mat = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                new_row.append(self.data[i][j] * other.data[i][j])
            new_mat.append(new_row)
        return Matrix(new_mat)

    def __add__(self, other):
        if type(other) is Matrix:
            if self.rows != other.rows or self.cols != other.cols:
                raise Exception(
                    "both matrices are not the same size [{},{}] + [{},{}]".format(self.rows, self.cols, other.rows,
                                                                                   other.cols))
            new_mat = []
            for i in range(self.rows):
                new_row = []
                for j in range(self.cols):
                    new_row.append(self.data[i][j] + other.data[i][j])
                new_mat.append(new_row)
            return Matrix(new_mat)
        else:
            new_mat = self.data
            for i in range(self.rows):
                for j in range(self.cols):
                    new_mat[i][j] += other
            return Matrix(new_mat)

    def __sub__(self, other):
        if type(other) is Matrix:
            if self.rows != other.rows or self.cols != other.cols:
                raise Exception(
                    "both matrices are not the same size [{},{}] - [{},{}]".format(self.rows, self.cols, other.rows,
                                                                                   other.cols))
            new_mat = []
            for i in range(self.rows):
                new_row = []
                for j in range(self.cols):
                    new_row.append(self.data[i][j] - other.data[i][j])
                new_mat.append(new_row)
            return Matrix(new_mat)
        else:
            new_mat = self.data
            for i in range(self.rows):
                for j in range(self.cols):
                    new_mat[i][j] -= other
            return Matrix(new_mat)

    def __mul__(self, other):
        if type(other) is Matrix:
            if self.cols != other.rows:
                raise Exception(
                    "both matrices are not the same size [{},{}] * [{},{}]".format(self.rows, self.cols, other.rows,
                                                                                   other.cols))
            other_trans = other.transpose()
            new_mat = []
            for i in range(self.rows):
                new_row = []
                for j in range(other.cols):
                    summer = sum([self.data[i][k] * other_trans.data[j][k] for k in range(other_trans.cols)])
                    new_row.append(summer)
                new_mat.append(new_row)
            return Matrix(new_mat)
        else:
            new_mat = self.data
            for i in range(self.rows):
                for j in range(self.cols):
                    new_mat[i][j] *= other
            return Matrix(new_mat)

    def __str__(self):
        output = "--- [{},{}] MATRIX ---\n".format(self.rows, self.cols)
        for row in self.data:
            for col in row:
                output += str(col) + "\t"
            output += '\n'
        return output


def softmax(output):
    outputs = output.data[0]
    summing = 0
    for i in range(len(outputs)):
        summing += exp(outputs[i])
    new_outputs = []
    for i in range(len(outputs)):
        item = exp(outputs[i]) / summing
        new_outputs.append(item)
    return new_outputs


class NeuralNetworkMatrix:
    def __init__(self, layers, learning_rate=0.03):
        self.layers = layers
        self.Z = [0] * len(layers)
        self.A = [0] * len(layers)
        self.deltas = [0] * (len(self.layers) - 1)
        self.learning_rate = learning_rate
        random.seed(1)
        self.weights = []
        for i in range(len(layers) - 1):
            w = [[random.uniform(-1, 1) for _ in range(layers[i + 1])] for _ in
                 range(layers[i])]
            self.weights.append(Matrix(w))

    def forward_propagation(self, x):
        X = Matrix(x)
        i = 1
        self.Z[0] = X
        self.A[0] = X.sigmoid()
        for weight in self.weights:
            # print("weights ->", weight)
            current_z = X * weight
            self.Z[i] = current_z
            X = current_z.sigmoid()
            self.A[i] = X
            i += 1
        return X

    def predict(self, x):
        return self.forward_propagation(x)

    def test(self, data):
        print("\t\tStarted Testing")
        data = self.shuffle_data(data)
        test_inputs = []
        test_inputs[:] = data["inputs"]
        outputs = [data["outputs"][i].index(max(data["outputs"][i])) for i in range(len(data["outputs"]))]
        avg = 0.0
        size = len(test_inputs)
        for i in range(len(test_inputs)):
            if i % 100 == 0:
                print("\t\ttesting item {}/{}".format(i + 1, size))
            current_input = test_inputs[i]
            expected_output = outputs[i]
            output = self.predict(current_input)
            # output = softmax(output)
            if expected_output == self.normal(output):
                avg += 1
        return avg * 100 / len(test_inputs)

    def shuffle_data(self, data):
        inputs = data["inputs"]
        outputs = data["outputs"]
        combined = list(zip(inputs, outputs))
        random.shuffle(combined)
        inputs[:], outputs[:] = zip(*combined)
        return {"inputs": inputs, "outputs": outputs}

    def new_train(self, data, test_data, batch_size=100, epochs=500):
        """
        for each batch:
            accumulate the sum_error.
            calculate each item delta.
        :param epochs:
        :param data:
        :param batch_size:
        :return:
        """
        print("Started training...")
        data = self.shuffle_data(data)
        inputs, outputs = data.values()
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        batches_outputs = [outputs[i:i + batch_size] for i in range(0, len(outputs), batch_size)]
        size = len(batches)
        for epoch in range(epochs):
            sum_error = 0
            for i in range(len(batches)):
                if i % 10 == 0:
                    print("training example {}/{}".format(i + 1, size))
                out = self.forward_propagation(batches[i])
                self.update_weights(batches_outputs[i], out)
                # sub_mat = out - Matrix(batches_outputs[i])
                # sum_error += sum([0.5 * ((sub_mat.data[0][j]) ** 2) for j in range(len(sub_mat.data[0]))])
                # calculate deltas for each weight layer
                # remove deltas*learning_rate from weights.
            acc = self.test(test_data)
            print("\tepoch {}/{} ,acc= {}%".format(epoch + 1, epochs, round(acc, 2)))

    def calc_deltas(self, batches_outputs, output):
        # calc last delta
        batches_outputs = Matrix(batches_outputs)
        d = (batches_outputs - output).scalar_mul(self.Z[-1].sigmoid(derivative=True))
        last_delta = self.Z[-2].sigmoid().transpose() * d
        deltas = [d]
        changes = [last_delta]
        for i in reversed(range(len(self.weights) - 1)):
            delta = (deltas[-1] * self.weights[i + 1].transpose()).scalar_mul(self.Z[i + 1].sigmoid(derivative=True))
            deltas.append(delta)
            change = self.Z[i].sigmoid().transpose() * delta
            changes.append(change)
        changes = list(reversed(changes))

        # update weights
        for i in range(len(changes)):
            self.weights[i] += changes[i] * self.learning_rate

    def update_weights(self, batches_outputs, output):
        # calc last delta
        batches_outputs = Matrix(batches_outputs)
        d_last = (batches_outputs - output).scalar_mul(self.Z[-1].sigmoid(derivative=True))
        change_last = self.A[-2].transpose() * d_last
        num_layers = len(self.layers)
        deltas = [None] * num_layers
        changes = [None] * (num_layers - 1)
        deltas[num_layers - 1] = d_last
        changes[num_layers - 2] = change_last
        list_to_run = list(reversed(range(num_layers - 1)))[:-1]
        for i in list_to_run:
            d_next = deltas[i + 1]
            W = self.weights[i].transpose()
            S_Z = self.Z[i].sigmoid(derivative=True)
            d = (d_next * W).scalar_mul(S_Z)
            deltas[i] = d

            A = self.A[i - 1].transpose()
            D = deltas[i + 1]
            W = self.weights[i].transpose()
            S_Z = self.Z[i].sigmoid(derivative=True)
            change = A * (D * W).scalar_mul(S_Z)
            changes[i-1] = change
        # update weights
        for i in range(len(changes)):
            self.weights[i] -= changes[i] * self.learning_rate

    @staticmethod
    def calc_sum_error(expected_output, outputs):
        return (1 / len(expected_output)) * sum(
            [abs(expected_output[i] - outputs[i]) for i in range(len(expected_output))])

    @staticmethod
    def normalize_output(output):
        new_output = []
        for item in output:
            if item >= 0.5:
                new_output.append(1.0)
            else:
                new_output.append(0)
        return new_output

    def calc_cost(self, output, expected_output):
        """
        Z is the sum of inputs * weights
        A is the sigmoid of Z
        :param output:
        :param expected_output:
        :param Z:
        :return:
        """
        expected_output = Matrix(expected_output)
        for i in range(len(self.weights)):
            delta = (output - expected_output) * self.Z[i + 1].sigmoid(derivative=True)
            self.deltas[i] = delta * self.Z[i + 1].sigmoid().transpose()

    def normal(self, output):
        return output.data[0].index(max(output.data[0]))