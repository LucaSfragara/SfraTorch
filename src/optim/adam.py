import numpy as np


class Adam:
    """
    Adam optimizer implementation.
    Adam (Adaptive Moment Estimation) combines ideas from RMSprop and momentum.
    It computes adaptive learning rates for each parameter using estimates of
    first and second moments of the gradients.
    """

    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Initialize Adam optimizer.

        Args:
            model: The model with layers containing parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term for numerical stability (default: 1e-8)
        """
        # Initialize the model layers.
        self.l = model.layers
        self.L = len(model.layers)

        # Assign hyperparameters.
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        # Initialize time step counter.
        self.t = 0

        # Initialize first moment estimates (mean) for weights and biases to zeros.
        self.m_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.m_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

        # Initialize second moment estimates (uncentered variance) for weights and biases to zeros.
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        """
        Perform a single optimization step.
        Updates parameters using Adam update rule.
        """
        # Increment time step.
        self.t += 1

        # Update weights and biases for each layer.
        for i in range(self.L):
            # Update biased first moment estimate for weights.
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * self.l[i].dLdW

            # Update biased first moment estimate for biases.
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * self.l[i].dLdb

            # Update biased second raw moment estimate for weights.
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (self.l[i].dLdW ** 2)

            # Update biased second raw moment estimate for biases.
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.l[i].dLdb ** 2)

            # Compute bias-corrected first moment estimate for weights.
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected first moment estimate for biases.
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate for weights.
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)

            # Compute bias-corrected second raw moment estimate for biases.
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update weights using Adam update rule.
            self.l[i].W = self.l[i].W - self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)

            # Update biases using Adam update rule.
            self.l[i].b = self.l[i].b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class AdamW:
    """
    AdamW optimizer implementation.
    AdamW is a variant of Adam that decouples weight decay from the gradient-based update.
    This often leads to better generalization performance.
    """

    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer.

        Args:
            model: The model with layers containing parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
        """
        # Initialize the model layers.
        self.l = model.layers
        self.L = len(model.layers)

        # Assign hyperparameters.
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize time step counter.
        self.t = 0

        # Initialize first moment estimates (mean) for weights and biases to zeros.
        self.m_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.m_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

        # Initialize second moment estimates (uncentered variance) for weights and biases to zeros.
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        """
        Perform a single optimization step.
        Updates parameters using AdamW update rule with decoupled weight decay.
        """
        # Increment time step.
        self.t += 1

        # Update weights and biases for each layer.
        for i in range(self.L):
            # Update biased first moment estimate for weights.
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * self.l[i].dLdW

            # Update biased first moment estimate for biases.
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * self.l[i].dLdb

            # Update biased second raw moment estimate for weights.
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (self.l[i].dLdW ** 2)

            # Update biased second raw moment estimate for biases.
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.l[i].dLdb ** 2)

            # Compute bias-corrected first moment estimate for weights.
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected first moment estimate for biases.
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate for weights.
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)

            # Compute bias-corrected second raw moment estimate for biases.
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update weights using AdamW update rule (decoupled weight decay).
            # First apply the Adam update, then apply weight decay separately.
            self.l[i].W = self.l[i].W - self.lr * (m_W_hat / (np.sqrt(v_W_hat) + self.eps) + self.weight_decay * self.l[i].W)

            # Update biases using Adam update rule (typically no weight decay on biases).
            self.l[i].b = self.l[i].b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
