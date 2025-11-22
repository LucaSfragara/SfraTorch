import numpy as np
import numpy as np

class CTC(object):

    def __init__(self, BLANK=0):

        self.BLANK = BLANK

    def extend_target_with_blank(self, target):


        extended_symbols = [self.BLANK]

        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        skip_connect = np.zeros(N, dtype=int)


        # Implement skip connections
        for s in range(2, N):

            if extended_symbols[s] != self.BLANK and extended_symbols[s] != extended_symbols[s-2]:
                skip_connect[s] = 1


        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))


        return extended_symbols, skip_connect



    def get_forward_probs(self, logits, extended_symbols, skip_connect):

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        alpha[0, 0] = logits[0, extended_symbols[0]]
        if S > 1:
            alpha[0, 1] = logits[0, extended_symbols[1]]

        for t in range(1, T):
            for s in range(S):
                alpha[t, s] = alpha[t - 1, s]  # Same symbol
                if s > 0:
                    alpha[t, s] += alpha[t - 1, s - 1]  # Transition from previous symbol
                if s > 1 and skip_connect[s]:
                    alpha[t, s] += alpha[t - 1, s - 2]  # Skip connection
                alpha[t, s] *= logits[t, extended_symbols[s]]

        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros((T, S))

        # Initialize at last time step
        beta[T-1, S-1] = 1.0
        if S > 1:
            beta[T-1, S-2] = 1.0

        # Backward recursion
        for t in range(T-2, -1, -1):
            for s in range(S):
                beta[t, s] = beta[t+1, s] * logits[t+1, extended_symbols[s]]
                if s < S - 1:
                    beta[t, s] += beta[t+1, s+1] * logits[t+1, extended_symbols[s+1]]
                if s < S - 2 and skip_connect[s+2]:
                    beta[t, s] += beta[t+1, s+2] * logits[t+1, extended_symbols[s+2]]
        return beta

    def get_posterior_probs(self, alpha, beta):


        T, S = alpha.shape
        gamma = np.zeros_like(alpha)

        for t in range(T):
            sumgamma = np.sum(alpha[t, :] * beta[t, :])
            if sumgamma > 0:
                gamma[t, :] = (alpha[t, :] * beta[t, :]) / sumgamma
            else:
                gamma[t, :] = np.zeros(S) # Handle cases where sumgamma is zero

        return gamma

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
        
		"""
		# -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
		# <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        
        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            target_seq = target[batch_itr, :target_lengths[batch_itr]]
            logit_seq = logits[:input_lengths[batch_itr], batch_itr, :]

            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            alpha = self.ctc.get_forward_probs(logit_seq, extended_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logit_seq, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            log_probs = np.log(logit_seq)
            total_loss[batch_itr] = -np.sum(gamma * log_probs[:, extended_symbols])

        return np.mean(total_loss)
		

    def backward(self):
    
        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            target_seq = self.target[batch_itr, :self.target_lengths[batch_itr]]
            logit_seq = self.logits[:self.input_lengths[batch_itr], batch_itr, :]

            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            alpha = self.ctc.get_forward_probs(logit_seq, extended_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logit_seq, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for t in range(self.input_lengths[batch_itr]):
                for s, symbol in enumerate(extended_symbols):
                    dY[t, batch_itr, symbol] -= gamma[t, s] / logit_seq[t, symbol]

        return dY


