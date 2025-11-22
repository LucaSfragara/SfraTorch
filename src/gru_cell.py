import numpy as np
from src.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        self.z = self.z_act.forward(x@ self.Wzx.T + self.bzx + self.hidden @ self.Wzh.T + self.bzh)
        self.r = self.r_act.forward(x@ self.Wrx.T + self.brx + self.hidden @ self.Wrh.T + self.brh)

        self.n = self.h_act.forward(x@self.Wnx.T + self.bnx + self.r * (h_prev_t@self.Wnh.T+self.bnh))

        h_t = (1-self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        delta = delta.reshape(-1, 1)
        x = self.x.reshape(-1, 1)
        h_prev = self.hidden.reshape(-1, 1)

        z = self.z.reshape(-1, 1)
        r = self.r.reshape(-1, 1)
        n = self.n.reshape(-1, 1)

        dn = delta * (1 - z)
        dz = delta * (h_prev - n)
        dh_prev_t = delta * z
        dAn = dn * (1 - n**2)

        Wnh_h_prev = self.Wnh @ h_prev
        Wnh_h_prev_plus_bnh = Wnh_h_prev + self.bnh.reshape(-1,1)
        dr_raw = dAn * Wnh_h_prev_plus_bnh
        dr = dr_raw * r * (1 - r)

        dz = dz * z * (1 - z)

        dh_prev_from_n = self.Wnh.T @ (dAn * r)
        dh_prev_from_r = self.Wrh.T @ dr
        dh_prev_from_z = self.Wzh.T @ dz
        dh_prev_t = dh_prev_t + dh_prev_from_n + dh_prev_from_r + dh_prev_from_z

        dx_n = self.Wnx.T @ dAn
        dx_r = self.Wrx.T @ dr
        dx_z = self.Wzx.T @ dz
        dx = dx_n + dx_r + dx_z

        self.dWnx += dAn @ x.T
        self.dbnx += dAn.flatten()
        self.dWnh += (dAn * r) @ h_prev.T
        self.dbnh += (dAn * r).flatten()

        self.dWrx += dr @ x.T
        self.dbrx += dr.flatten()
        self.dWrh += dr @ h_prev.T
        self.dbrh += dr.flatten()

        self.dWzx += dz @ x.T
        self.dbzx += dz.flatten()
        self.dWzh += dz @ h_prev.T
        self.dbzh += dz.flatten()

        dx = dx.squeeze(-1)
        dh_prev_t = dh_prev_t.squeeze(-1)

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t

