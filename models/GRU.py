import numpy as np
from src.gru_cell import GRUCell



class GRU:
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = []

        for i in range(num_layers):
            input_shape = input_size if i == 0 else hidden_size
            gru_cell = GRUCell(input_shape, hidden_size)
            self.layers.append(gru_cell)

        self.caches = [[] for _ in range(num_layers)]

    def forward(self, input_seq, h0=None):
        """
        Args:
            input_seq: np.array of shape (seq_len, batch_size, input_size)
            h0: (optional) np.array of shape (num_layers, batch_size, hidden_size);
                if None, zeros are used.
        Returns:
            out: np.array of shape (seq_len, batch_size, hidden_size)
            h_n: final hidden states, shape (num_layers, batch_size, hidden_size)
        """
        seq_len, batch_size, _ = input_seq.shape

        if h0 is None:
            h0 = np.zeros((self.num_layers, batch_size, self.hidden_size))

        assert (h0 is not None), "h0 needs to be zeros "

        out = input_seq

        self.hiddens = []

        for l_i, cell in enumerate(self.layers):
            layer_cache = []
            layer_output = []
            h0_i = l_i
            h_prev = h0[h0_i]

            for t in range(seq_len):
                x_t = out[t]
                h_new = np.zeros((batch_size, self.hidden_size))
                cache_t = []

                for b in range(batch_size):
                    prev_h = h_prev[b].copy()
                    h_new[b] = cell.forward(x_t[b], prev_h)

                    cache_t.append({
                        'x':            x_t[b].copy(),
                        'hidden_prev':  prev_h,
                        'r':            cell.r.copy(),
                        'z':            cell.z.copy(),
                        'n':            cell.n.copy(),
                    })

                layer_cache.append(cache_t)
                layer_output.append(h_new)
                h_prev =  h_new.copy()

            self.caches[l_i] = layer_cache
            self.hiddens.append(h_new)

            out = np.stack(layer_output, axis=0)

        h_n = np.stack(self.hiddens, axis=0)
        return out, h_n

    def backward(self, d_out):
        """
        Backward pass through the GRU.
        Args:
            d_out: gradient with respect to the GRU output,
                   shape (seq_len, batch_size, hidden_size)
        Returns:
            d_input: gradient with respect to the GRU input,
                     shape (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = d_out.shape
        d_layer = d_out

        for l_i in reversed(range(self.num_layers)):
            cell = self.layers[l_i]
            layer_cache = self.caches[l_i]

            d_h_next = np.zeros((batch_size, self.hidden_size))
            d_input_layer = np.zeros((seq_len, batch_size, cell.Wrx.shape[1]))

            for t in reversed(range(seq_len)):
                d_h = d_layer[t] + d_h_next

                input_dim = self.input_size if l_i == 0 else self.hidden_size
                d_x_t = np.zeros((batch_size, input_dim))
                d_h_prev = np.zeros((batch_size, self.hidden_size))

                for b in range(batch_size):
                    cache = layer_cache[t][b]
                    cell.x, cell.hidden, cell.r, cell.z, cell.n = (
                        cache['x'],
                        cache['hidden_prev'],
                        cache['r'],
                        cache['z'],
                        cache['n']
                    )

                    dx, dh_prev_t = cell.backward(d_h[b])
                    d_x_t[b] = dx
                    d_h_prev[b] = dh_prev_t

                d_input_layer[t] = d_x_t
                d_h_next = d_h_prev
            d_layer = d_input_layer

        return d_layer