import numpy as np

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    def forward(self, x_seq):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        
        for x_t in x_seq:
            x_t = x_t.reshape(-1,1)
            concat = np.vstack((h_prev, x_t))
            
            f = self.sigmoid(self.Wf @ concat + self.bf)
            i = self.sigmoid(self.Wi @ concat + self.bi)
            o = self.sigmoid(self.Wo @ concat + self.bo)
            g = self.tanh(self.Wc @ concat + self.bc)
            
            c_next = f * c_prev + i * g
            h_next = o * np.tanh(c_next)
            
            h_prev, c_prev = h_next, c_next
        
        y_hat = self.Wy @ h_next + self.by
        return y_hat.flatten()

def forecast_multistep(model, seed_seq, steps):
    seq = seed_seq.copy()
    preds = []
    for _ in range(steps):
        pred = model.forward(seq)[0]
        preds.append(pred)
        seq = np.vstack([seq[1:], [[pred]]])
    return np.array(preds).reshape(-1,1)
