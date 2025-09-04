import numpy as np

class Dense:
    def __init__(self, in_size, out_size, activation="relu"):
        self.W = np.random.randn(in_size, out_size) * 0.1
        self.b = np.zeros((1, out_size))
        self.activation = activation

    def _act(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "softmax":
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        return x

    def _act_deriv(self, x):
        if self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return np.ones_like(x)

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self._act(self.z)
        return self.a

    def backward(self, grad, lr):
        dz = grad * self._act_deriv(self.z)
        dW = self.input.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        grad_input = dz @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db
        return grad_input


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, grad, lr):
        for l in reversed(self.layers):
            grad = l.backward(grad, lr)

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        for epoch in range(epochs):
            out = self.forward(X)

            loss = -np.mean(np.sum(y * np.log(out + 1e-9), axis=1))

            grad = (out - y) / X.shape[0]
            self.backward(grad, lr)

            if verbose and (epoch+1) % (epochs//10) == 0:
                print(f"Эпоха {epoch+1}/{epochs}, loss={loss:.4f}")

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)
