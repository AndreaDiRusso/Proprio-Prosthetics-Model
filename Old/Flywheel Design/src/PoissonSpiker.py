from neuron import h

class PoissonSpiker(object):
    """NetStim Object with variable lambda"""
    def __init__(self):
        self.soma = h.Section(name='soma', cell=self)
        self.syn_ = h.Exp2Syn(self.soma(0.5))
        self.stim = h.NetStim()
        self.stim.start = 0

        self.netcon = h.NetCon(self.stim, self.syn_)

        self.stim.interval = 10
        self.stim.number = 1e9
        self.stim.noise = 1

    def update_lambda(self, value):
        self.stim.interval = value