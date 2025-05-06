import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)


        _, seq_len, batch_size = y_probs.shape
        self.symbol_set = ["-"] + self.symbol_set
        for bi in range(batch_size):
            path = " "
            path_prob = 1
            for si in range(seq_len):
                i_max_prob = np.argmax(y_probs[:, si, bi])
                path_prob *= y_probs[i_max_prob, si, bi]
                if path[-1] != self.symbol_set[i_max_prob]:
                    path += self.symbol_set[i_max_prob]
            path = path.replace('-', '')
            decoded_path = path[1:]

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        
        self.symbol_set = ["-"] + self.symbol_set
        
        best_paths = dict()
        temp_best_paths = dict()
        best_paths["-"] = 1
        
        for t in range(T):
            cur_sym_probs = y_probs[:, t]
            temp_best_paths = dict()
            for path, score in best_paths.items():
                for sym, sym_prob in enumerate(cur_sym_probs):
                    if path[-1] == "-":
                        new_path = path[:-1] + self.symbol_set[sym]
                    elif path[-1] != self.symbol_set[sym] and not (t==T-1 and self.symbol_set[sym] == "-"):
                        new_path = path + self.symbol_set[sym]
                    else: 
                        new_path = path
                    if new_path in temp_best_paths:
                        temp_best_paths[new_path] += sym_prob * score
                    else:
                        temp_best_paths[new_path] = sym_prob * score
        
            #Limit to beam width:
            # print(temp_best_paths, self.beam_width)
            if len(temp_best_paths) >= self.beam_width:
                best_paths = dict(sorted(temp_best_paths.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])
                
        bestPath = max(best_paths, key=best_paths.get)
        FinalPathScore = dict()
        for path, score in temp_best_paths.items():
            if path[-1] == "-":
                FinalPathScore[path[:-1]] = score
            else:
                FinalPathScore[path] = score
        
        return bestPath, FinalPathScore
