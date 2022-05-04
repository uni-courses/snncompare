class LIF_neuron:
    """Creates a Leaky-Integrate-and-Fire neuron specification. Leaky-
    Integrate-and-Fire neural process with activation input and spike
    output ports a_in and s_out.

    Realizes the following abstract behavior:
    u[t] = u[t-1] * (1-du) + a_in
    v[t] = v[t-1] * (1-dv) + u[t] + bias
    s_out = v[t] > vth
    v[t] = v[t] - s_out*vth
    """

    def __init__(self, bias, du, dv, vth):
        self.bias = bias  # Amount of voltage added every timestep.
        self.du = du  # Change in current over time.
        self.dv = dv  # Change in voltage over time.
        self.vth = vth  # Threshold Voltage of the neuron.
