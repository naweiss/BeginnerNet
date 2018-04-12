from NonMatrixArtificialNeuralNetwork import NonMatrixArtificialNeuralNetwork

if __name__ == '__main__':
    # xor network
    nn = NonMatrixArtificialNeuralNetwork([2, 2, 1], lr=0.75)
    data = {
        "inputs":
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ],
        "outputs":
            [
                [0],
                [1],
                [1],
                [0]
            ]
    }
    inputs = list(zip(data["inputs"], data["outputs"]))
    nn.train(inputs, epochs=1000)

    for x, y in inputs:
        y_hat = nn.predict(x)
        print("===\nexpected= {}, we got= {}".format(y, y_hat))
