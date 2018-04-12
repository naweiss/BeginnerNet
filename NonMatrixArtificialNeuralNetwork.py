from math_functions import sigmoid, dsigmoid
import random

def printing(array):
    n, m = len(array), len(array[0])
    data_str = [[str(cell) for cell in row] for row in array]
    lens = [max(map(len, col)) for col in zip(*data_str)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in data_str]
    sizes = '[' + str(n) + ' x ' + str(m) + ']'
    print('\n'.join([sizes] + table))

class NonMatrixArtificialNeuralNetwork:
    def __init__(self, layers, lr=0.7):
        self.layers = layers
        self.lr = lr
        self.W = []
        self.Z = []
        self.A = []
        for i in range(1, len(layers)):
            w = [[random.uniform(-1, 1) for _ in range(layers[i])] for _ in
                 range(layers[i - 1])]
            self.W.append(w)

    def forward_prop(self, x):
        output = x
        self.A = [x]
        self.Z = [x]
        for k in range(len(self.W)):
            new_output = []
            current_z = []
            current_a = []
            for j in range(len(self.W[k][0])):
                summer = 0
                for i in range(len(output)):
                    summer += output[i] * self.W[k][i][j]
                current_z.append(summer)
                summer = sigmoid(summer)
                current_a.append(summer)
                new_output.append(summer)
            self.Z.append(current_z)
            self.A.append(current_a)
            output = new_output
        return output

    def back_prop(self, x, y):
        y_hat = self.forward_prop(x)
        cost_derivative = self.dcost(y, y_hat)
        print("cost:", sum([0.5 * (y[i] - y_hat[i]) ** 2 for i in range(len(y))]))
        deltas = [None] * len(self.layers)
        deltas[-1] = [cost_derivative[i] * dsigmoid(self.Z[-1][i]) for i in range(len(cost_derivative))]
        changes = [None] * len(self.W)
        changes[-1] = self.calc_changes_for_weights(deltas, len(self.layers) - 2)
        for k in reversed(range(len(self.layers) - 1)):
            deltas[k] = self.calc_deltas_for_current_layer(k, deltas)
            changes[k] = self.calc_changes_for_weights(deltas, k)
        return changes

    def calc_deltas_for_current_layer(self, k, deltas):
        deltas_output = []
        for j in range(self.layers[k]):
            holder = []
            for i in range(len(deltas[k + 1])):
                d = deltas[k + 1][i]
                sig = dsigmoid(self.Z[k + 1][i])
                w = self.W[k][j][i]
                mul = d * sig * w
                holder.append(mul)
            deltas_output.append(sum(holder))
        return deltas_output

    def calc_changes_for_weights(self, deltas, k):
        changes = []
        for i in range(len(self.W[k])):
            new_row = []
            for j in range(len(self.W[k][i])):
                d = deltas[k + 1][j]
                sig = dsigmoid(self.Z[k + 1][j])
                a = self.A[k][j]
                result = d * a
                new_row.append(result)
            changes.append(new_row)
        return changes

    def train(self, data, epochs=500):
        i = 0
        for epoch in range(epochs):
            inputs, outputs = self.shuffle_data(data)
            for i in range(len(inputs)):
                x = inputs[i]
                y = outputs[i]
                self.update_weights(self.back_prop(x, y))

    def update_weights(self, changes):
        for k in range(len(self.W)):
            for i in range(len(self.W[k])):
                for j in range(len(self.W[k][i])):
                    self.W[k][i][j] -= self.lr * changes[k][i][j]

    def shuffle_data(self, data):
        random.shuffle(data)
        inputs, outputs = zip(*data)
        return (inputs, outputs)

    def dcost(self, y, y_hat):
        assert len(y) == len(y_hat)
        return [y_hat[i] - y[i] for i in range(len(y))]

    def predict(self, x):
        return self.forward_prop(x)
