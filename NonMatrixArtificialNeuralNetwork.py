from math_functions import sigmoid, dsigmoid
import random

def printing(array):
    """
    Pretty print matrix
    """
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
            # print("weight:")
            # printing(self.W[k])
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

    def train(self, data, epochs=500):
        i = 0
        for epoch in range(epochs):
            data = self.shuffle_data(data)
            inputs, outputs = data.values()
            changes = []
            for i in range(len(inputs)):
                x = inputs[i]
                y = outputs[i]
                changes.append(self.back_prop(x, y))
            self.update_weights(changes)

    def update_weights(self, all_changes):
        # for k in range(len(self.W)):
        #     for i in range(len(self.W[k])):
        #         for j in range(len(self.W[k][i])):
        #             summer = 0
        #             for l in range(len(all_changes)):
        #                 summer += all_changes[l][k][i][j]
        #             self.W[k][i][j] -= self.lr * (summer / len(all_changes))
        change = []
        for i in range(len(self.W)):
            change_row = []
            for j in range(len(self.W[i])):
                change_col = []
                for k in range(len(self.W[i][j])):
                    winter = 0
                    for l in range(len(all_changes)):
                        winter += (all_changes[l][i][j][k] / len(all_changes))
                    change_col.append(winter)
                change_row.append(change_col)
            change.append(change_row)

        for k in range(len(self.W)):
            current_w = self.W[k]
            for i in range(len(current_w)):
                for j in range(len(current_w[i])):
                    current_w[i][j] -= self.lr * change[k][i][j]
            self.W[k] = current_w

    def shuffle_data(self, data):
        inputs = data["inputs"]
        outputs = data["outputs"]
        combined = list(zip(inputs, outputs))
        random.shuffle(combined)
        inputs[:], outputs[:] = zip(*combined)
        return {"inputs": inputs, "outputs": outputs}

    def dcost(self, y, y_hat):
        assert len(y) == len(y_hat)
        return [y_hat[i] - y[i] for i in range(len(y))]

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

    def predict(self, x):
        return self.forward_prop(x)
