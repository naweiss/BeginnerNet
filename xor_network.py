from NonMatrixArtificialNeuralNetwork import NonMatrixArtificialNeuralNetwork

if __name__ == '__main__':
    # xor network
    nn = NonMatrixArtificialNeuralNetwork([2, 2, 3, 1], lr=0.75)
    data = {
        "inputs":
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]
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

    nn.train(data, epochs=1000)

    for inp, out in inputs:
        x = inp
        y_hat = nn.predict(x)
        y = out
        print("===\nexpected= {}, we got= {}".format(y, y_hat))
