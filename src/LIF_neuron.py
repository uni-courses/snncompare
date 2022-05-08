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

    def __init__(self, bias: int, du: int, dv: int, vth: int):
        self.bias = bias  # Amount of voltage added every timestep.
        self.du = du  # Change in current over time.
        self.dv = dv  # Change in voltage over time.
        self.vth = vth  # Threshold Voltage of the neuron.

        # Initialise default values.
        self.v_reset = 0
        self.v = 0
        self.u = 0
        self.s_out = 1
        self.spikes = False
        self.a_in = 0
        self.a_in_next = 0

    def simulate_neuron_one_timestep(self, a_in: int) -> bool:
        """

        :param a_in: int:

        """
        self.set_compute_u(a_in)
        self.set_compute_v(a_in)
        return self.spikes

    # TODO: make this function only accessible to object itself.
    def set_compute_u(self, a_in):
        """

        :param a_in:

        """
        self.u = self.u * (1 - self.du) + a_in

    # TODO: make this function only accessible to object itself.
    def set_compute_v(self, current_u) -> None:
        """

        :param current_u:

        """
        new_voltage = self.v * (1 - self.dv) + self.bias
        if new_voltage > self.vth:
            self.spikes = True

            # TODO: Mention discrepancy between lava-nc.org documentation
            # and github.com/lava-nc LIF object.
            # Documentation says voltage gets set to:
            # self.v = self.v - self.s_out * self.vth
            # Github object/code says voltage is reset to 0.

            # Reset voltage based on output voltage.
            self.v = 0
        else:
            self.spikes = False
            self.v = new_voltage
