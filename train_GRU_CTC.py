import numpy as np
from models.GRU import GRU
from CTC.CTC import CTCLoss
from src.nn.linear import Linear  

class GRU_CTC:
    def __init__(self, input_size, hidden_size, num_layers, num_symbols):
        """
        Args:
            input_size: Dimensionality of input features.
            hidden_size: Hidden size of the GRU.
            num_layers: Number of GRU layers.
            num_symbols: Number of output symbols (including blank, assumed index 0).
        """
        self.gru = GRU(input_size, hidden_size, num_layers)
        self.output_size = hidden_size
        self.linear = Linear(self.output_size, num_symbols)
    
    def forward(self, x):
        """
        Forward pass:
            x: shape (seq_len, batch_size, input_size)
        Returns:
            probs: shape (seq_len, batch_size, num_symbols)
        """
        self.gru_out, self.h_n = self.gru.forward(x)
        seq_len, batch_size, _ = self.gru_out.shape

        x_flat = self.gru_out.reshape(seq_len * batch_size, -1)
        self.logits_flat = self.linear.forward(x_flat)
        self.logits = self.logits_flat.reshape(seq_len, batch_size, -1)

        max_logits = np.max(self.logits, axis=2, keepdims=True)
        exp_logits = np.exp(self.logits - max_logits)
        self.probs = exp_logits / (np.sum(exp_logits, axis=2, keepdims=True) + 1e-8)
        return self.probs
    
    
    def backward(self, dY):
        """
        Backward pass:
            dY: gradient from CTCLoss, shape (seq_len, batch_size, num_symbols)
        Returns:
            d_x: gradient with respect to the network input,
                 shape (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = dY.shape
        dY_flat = dY.reshape(seq_len * batch_size, -1)

        d_gru_out_flat = self.linear.backward(dY_flat)
        d_gru_out = d_gru_out_flat.reshape(seq_len, batch_size, -1)

        d_x = self.gru.backward(d_gru_out)
        return d_x

def train():
    seq_len     = 15
    batch_size  = 3
    input_size  = 8
    hidden_size = 16
    num_layers  = 2
    num_symbols = 5
    num_epochs  = 10
    learning_rate = 0.001

    model = GRU_CTC(input_size, hidden_size, num_layers, num_symbols)
    ctc_loss_fn = CTCLoss(BLANK=0)

    x = np.random.randn(seq_len, batch_size, input_size)

    target_sequences = []
    target_lengths = []
    for b in range(batch_size):
        t_len = np.random.randint(1, seq_len // 2 + 1)
        target_lengths.append(t_len)
        target_seq = np.random.randint(1, num_symbols, size=(t_len,))
        target_sequences.append(target_seq)
    max_target_len = max(target_lengths)
    targets_padded = np.zeros((batch_size, max_target_len), dtype=int)
    for i, seq in enumerate(target_sequences):
        targets_padded[i, :len(seq)] = seq

    input_lengths = np.full((batch_size,), seq_len, dtype=int)
    target_lengths = np.array(target_lengths, dtype=int)

    for epoch in range(num_epochs):
        probs = model.forward(x)

        loss = ctc_loss_fn.forward(probs, targets_padded, input_lengths, target_lengths)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

        dY = ctc_loss_fn.backward()
        probs_grad = model.backward(dY)

        model.linear.W -= learning_rate * model.linear.dLdW
        model.linear.b -=  learning_rate * model.linear.dLdb

        for cell in model.gru.layers:
            cell.Wrx -= learning_rate * cell.dWrx
            cell.Wzx -= learning_rate * cell.dWzx
            cell.Wnx -= learning_rate * cell.dWnx
            cell.Wrh -= learning_rate * cell.dWrh
            cell.Wzh -= learning_rate * cell.dWzh
            cell.Wnh -= learning_rate * cell.dWnh
            cell.brx -= learning_rate * cell.dbrx
            cell.bzx -= learning_rate * cell.dbzx
            cell.bnx -= learning_rate * cell.dbnx
            cell.brh -= learning_rate * cell.dbrh
            cell.bzh -= learning_rate * cell.dbzh
            cell.bnh -= learning_rate * cell.dbnh

if __name__ == '__main__':
    train()