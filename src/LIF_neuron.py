# -*- coding: utf-8 -*-
"""File represents LIF neuron object."""


class LIF_neuron:
    """Creates a Leaky-Integrate-and-Fire neuron specification. Leaky-
    Integrate-and-Fire neural process with activation input and spike output
    ports a_in and s_out.

    Realizes the following abstract behavior:
    u[t] = u[t-1] * (1-du) + a_in
    v[t] = v[t-1] * (1-dv) + u[t] + bias
    s_out = v[t] > vth
    v[t] = v[t] - s_out*vth
    """

    # pylint: disable=too-many-instance-attributes
    # Eleven is considered is reasonable in this case.
    def __init__(
        self, name: int, bias: float, du: float, dv: float, vth: float
    ) -> None:
        # pylint: disable=R0913
        self.bias = Bias(bias)  # Amount of voltage added every timestep.
        self.du = Du(du)  # Change in current over time.
        self.dv = Dv(dv)  # Change in voltage over time.
        self.name = name  # Set the identifier of the neuron.
        self.vth = Vth(vth)  # Threshold Voltage of the neuron.

        # Initialise default values.
        self.v_reset: float = 0.0
        self.v: V = V(0)
        self.u: U = U(0)
        self.s_out = 1
        self.spikes = False
        self.a_in: float = 0.0
        self.a_in_next: float = 0.0

    def simulate_neuron_one_timestep(self, a_in: int) -> bool:
        """Computes what the new current u and new voltage v will be based on
        the default neuron properties, du,dv, bias, previous current u,
        previous voltage v, and the incoming input signal/value of a_in. Based
        on this new voltage it computes whether the neuron spikes or not. Then
        returns the boolean signal indicating whether it will spike (True) or
        not (False).

        :param a_in: int: the input current into this neuron.
        :param a_in: int:
        """
        self.set_compute_u(a_in)
        self.set_compute_v()
        return self.spikes

    # TODO: make this function only accessible to object itself.
    def set_compute_u(self, a_in):
        """Computes the new current u based on the previous current u, du, and
        the incoming input signal/value of a_in. After computation overwrites
        the previous value of the u with the new value for u.

        :param a_in: int: the input current into this neuron.
        """
        self.u = U(self.u.get() * (1 - self.du.get()) + a_in)

    # TODO: make this function only accessible to object itself.
    def set_compute_v(self) -> None:
        """Computes the new voltage v based on the previous current v, the new
        current u, the bias and the dv.

        Then overwarites the
        """
        new_voltage = self.v.get() * (1 - self.dv.get()) + self.bias.get()
        print(f"self.name={self.name}")
        print(f"new_voltage={new_voltage}")
        if new_voltage > self.vth.get():
            self.spikes = True

            # TODO: Mention discrepancy between lava-nc.org documentation
            # and github.com/lava-nc LIF object.
            # Documentation says voltage gets set to:
            # self.v = self.v - self.s_out * self.vth
            # Github object/code says voltage is reset to 0.

            # Reset voltage based on output voltage.
            self.v = V(0)
        else:
            self.spikes = False
            self.v = V(new_voltage)


class Bias:
    """Creates a bias object that contains a float, and a get() function."""

    # pylint: disable=R0903

    def __init__(self, bias: float) -> None:
        self.bias = bias

    def get(self) -> float:
        """Returns the bias value as a float."""
        return self.bias


class Du:
    """Creates a du object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, du: float) -> None:
        self.du = du

    def get(self) -> float:
        """Returns the du value as a float."""
        return self.du


class Dv:
    """Creates a dv object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, dv: float) -> None:
        self.dv = dv

    def get(self) -> float:
        """Returns the dv value as a float."""
        return self.dv


class U:
    """Creates a u object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, u: float) -> None:
        self.u = u

    def get(self) -> float:
        """Returns the u (current) value as a float."""
        return self.u


class V:
    """Creates a v object that contains a float, and a get() fvnction."""

    # pylint: disable=R0903
    def __init__(self, v: float) -> None:
        self.v = v

    def get(self) -> float:
        """Returns the v (voltage) value as a float."""
        return self.v


class Vth:
    """Creates a vth object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, vth: float) -> None:
        self.vth = vth

    def get(self) -> float:
        """Returns the vth (threshold voltage) value as a float."""
        return self.vth


def print_neuron_properties(neurons, spikes=None, ids=None):
    """

    :param neurons:
    :param spikes:  (Default value = None)
    :param ids:  (Default value = None)

    """
    spacing = 4
    if ids is not None:
        for x in ids:
            print(f"{str(x) : <{spacing+5}}", end=" ")

    if spikes is not None:
        for x in spikes:
            print("")
            print(f"spk={x : <{spacing+1}}", end=" ")
    for x in neurons:
        print("")
        print(f"u={str(x.u.get()) : <{spacing+3}}", end=" ")
        print("")
        print(f"du={str(x.du.get()) : <{spacing+2}}", end=" ")
        print("")
        print(f"v={str(x.v.get()) : <{spacing+3}}", end=" ")
        print("")
        print(f"dv={str(x.dv.get()) : <{spacing+2}}", end=" ")
        print("")
        print(f"bias={str(x.bias.get()) : <{spacing}}", end=" ")
        print("")
        print(f"vth={str(x.vth.get()) : <{spacing+1}}", end=" ")
        print("\n")
