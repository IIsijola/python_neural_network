def adjustWeights(self):
    # 	delta weights = learning_rate- * Error * (activation function derivative- * output) * cost derivative-

    for i, layer in enumerate(reversed(self.NeuronList)):
        errorMatrix = deepcopy(layer.errorMatrix)
        outputMatrix = deepcopy(layer.outputMatrix)

        print "Error matrix of {} layer".format(layer.name)
        print errorMatrix

        print layer.weightMatrix

        print "Output matrix of {} layer".format(layer.name)
        print outputMatrix

        layer.errorMatrix.transpose()

        # Calculate gradients
        outputMatrix.map(self.sigmoid_prime)

        print "Output matrix of {} layer multiplied by sigmoid prime".format(layer.name)
        print outputMatrix

        print
        print
        print

        outputMatrix_ = outputMatrix.multiply(layer.errorMatrix)
        outputMatrix_.scalarMultiplication(self.LearningRate)

        # Calculate change in weights
        outputMatrix.transpose()
        delta = outputMatrix.multiply(outputMatrix_)
        delta.scalarMultiplication(-1)

        layer.modifyWeights(delta)

    # print outputMatrix