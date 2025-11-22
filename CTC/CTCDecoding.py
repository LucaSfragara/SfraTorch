import numpy as np
from typing import List, Dict
from copy import deepcopy

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        decoded_path = []
        blank = 0
        path_prob = 1
        seq_length = y_probs.shape[1]
        batch_idx = 0
        for t in range(seq_length):
            symbol_idx = np.argmax(y_probs[:, t, batch_idx])
            path_prob *= y_probs[symbol_idx, t, batch_idx]
            if symbol_idx != blank:
                if not decoded_path or symbol_idx != decoded_path[-1]:
                    decoded_path.append(symbol_idx)
        result = ''.join([self.symbol_set[idx-1] for idx in decoded_path])
        return result, path_prob



class BeamSearchDecoder:
    def __init__(self, symbol_set, beam_width):
        self.alphabet = symbol_set
        self.width = beam_width

    def decode(self, y_probs):
        self.y_probs = y_probs
        self.blank_list = ['']
        self.blank_score = {'': y_probs[0, 0, 0]}
        self.symbol_list = [ch for ch in self.alphabet]
        self.symbol_score = {}
        
        for idx, ch in enumerate(self.alphabet):
            self.symbol_score[ch] = y_probs[idx+1, 0, 0]
            
        for t in range(0, y_probs.shape[1]-1):
            t+=1
            # Prune current beams based on a cutoff score
            pruned_blank = []
            pruned_blank_score = {}
            pruned_symbol = []
            pruned_symbol_score = {}
            score_candidates = list(self.blank_score.values()) + list(self.symbol_score.values())
            score_candidates.sort()
            
            score_candidates_len = -self.width
            
            cutoff = score_candidates[-1] if len(score_candidates) < self.width else score_candidates[score_candidates_len]
            for seq in self.blank_list:
                if self.blank_score[seq] and self.blank_score[seq] >= cutoff:
                    pruned_blank_score[seq] = self.blank_score[seq]
                    pruned_blank.append(seq)
                else:
                    pass
            for seq in self.symbol_list:
                if self.symbol_score[seq] and self.symbol_score[seq] >= cutoff:
                    pruned_symbol_score[seq] = self.symbol_score[seq]
                    pruned_symbol.append(seq)
                else:
                    pass
                    
            self.blank_list = pruned_blank
            self.blank_score = pruned_blank_score
            self.symbol_list = pruned_symbol
            self.symbol_score = pruned_symbol_score
            
            # Extend beams with new symbols and blanks
            new_symbol_list, new_symbol_score = self.extend_with_symbol(t)
            new_blank_list, new_blank_score = self.extend_with_blank(t)
            
            self.blank_list = new_blank_list
            self.symbol_list = new_symbol_list
            self.blank_score = new_blank_score
            self.symbol_score = new_symbol_score
            
        merged_list = deepcopy(self.blank_list)
        merged_score = deepcopy(self.blank_score)
        
        for seq in self.symbol_list:
            if seq in merged_list:
                merged_score[seq] += self.symbol_score[seq]
            else:
                merged_list.append(seq)
                merged_score[seq] = self.symbol_score[seq]
        best_seq = merged_list[0]
        best_val = merged_score[best_seq]
        
        for seq in merged_score:
            if merged_score[seq] > best_val:
                best_seq = seq
                best_val = merged_score[seq]
        return best_seq, merged_score
        
    
    def extend_with_symbol(self, t):
        
        new_sym_list = []
        new_sym_score = {}
        
        for seq in self.blank_list:
            idx = 0
            for ch in self.alphabet:
                candidate = seq + ch
                
                new_sym_score[candidate] = self.blank_score[seq] * self.y_probs[idx + 1, t, 0]
                new_sym_list.append(candidate)
                idx += 1
                
        for seq in self.symbol_list:
            idx = 0
            for ch in self.alphabet:
                candidate = seq if seq and ch == seq[-1] else seq + ch
                if candidate not in new_sym_score:
                    new_sym_score[candidate] = self.symbol_score[seq] * self.y_probs[idx + 1, t, 0]
                    new_sym_list.append(candidate)
                else:
                   new_sym_score[candidate] += self.symbol_score[seq] * self.y_probs[idx + 1, t, 0]
                idx += 1
        return new_sym_list, new_sym_score

    def extend_with_blank(self, t):
        new_blank_list = []
        new_blank_score = {}
        for seq in self.blank_list:
            new_blank_list.append(seq)
            new_blank_score[seq] = self.blank_score[seq] * self.y_probs[0, t, 0]
        for seq in self.symbol_list:
            if seq not in new_blank_list:
                new_blank_score[seq] = self.symbol_score[seq] * self.y_probs[0, t, 0]
                new_blank_list.append(seq)
                
            else:
                new_blank_score[seq] += self.symbol_score[seq] * self.y_probs[0, t, 0]
                
        return new_blank_list, new_blank_score


