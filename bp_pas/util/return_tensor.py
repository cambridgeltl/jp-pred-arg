class ReturnTensor:
    """
    A ReturnTensor wrapper, has an output tensor
    """

    def __init__(self, output, weights=[]):
        self.output = output
        self.weights = weights

class TrainableReturnTensor:
    """
    A TrainableReturnTensor wrapper, replaces weight tensor list with a loss
    """

    def __init__(self, output, loss):
        self.output = output
        self.loss = loss
