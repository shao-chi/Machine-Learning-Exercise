import numpy as np 


class Layer:
    def __init__(self):
        self.weight = None
        self.bias = None
        self.learning_rate = None

    def forward(self, inputs):
        return np.matmul(inputs, self.weight) + self.bias

    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def softmax(self, inputs):
        inputs_exp = np.exp(inputs - np.max(inputs))

        return inputs_exp / inputs_exp.sum()

    def softmax_backward(self, outputs, loss_grad):
        softmax_grad = outputs * (1 - outputs)
        
        return loss_grad * softmax_grad

    def ReLU_backward(self, outputs, loss_grad):
        relu_grad = outputs > 0
        return loss_grad * relu_grad

    def backward(self, outputs, loss_grad):
        grad_outputs = np.dot(loss_grad, np.transpose(self.weight))

        weight_grad = np.transpose(np.dot(np.transpose(loss_grad), outputs))
        bias_grad = np.sum(loss_grad, axis=0)

        self.weight = self.weight - self.learning_rate * weight_grad
        self.bias = self.bias - self.learning_rate * bias_grad

        return grad_outputs


class ZeroInit_Layer(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.001):
        self.weight = np.zeros((input_units, output_units))
        self.bias = np.zeros(output_units)
        self.learning_rate = learning_rate

class RandomInit_Layer(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.001):
        self.weight = np.random.randn(input_units, output_units) * 0.01
        self.bias = np.zeros(output_units)
        self.learning_rate = learning_rate


def CrossEntropy(outputs, ans):
    out_for_ans = outputs[np.arange(len(outputs)), ans]
    return -out_for_ans + np.log(np.sum(np.exp(outputs), axis=-1))


def gradient_CrossEntropy(outputs, ans):
    softmax = np.exp(outputs) / np.exp(outputs).sum(axis=-1, keepdims=True)

    return (softmax - ans) / outputs.shape[0]

def error_rate(outputs, ans):
    error = 0
    for i in range(len(outputs)):
        if np.argmax(outputs[i]) != np.argmax(ans[i]):
            error += 1

    return error / len(outputs)