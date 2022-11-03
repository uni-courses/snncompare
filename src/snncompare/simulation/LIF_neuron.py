"""File represents LIF neuron object."""

from typing import Any, List

import networkx as nx
import numpy as np
from lava.proc.lif.process import LIF


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
        self.v: V = V(0.0)
        self.u: U = U(0.0)
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
        :param a_in: int:
        """
        self.set_compute_u(a_in)
        self.set_compute_v()  # Also sets self.spikes
        return self.spikes

    # TODO: make this function only accessible to object itself.
    def set_compute_u(self, a_in: int) -> None:
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
        new_voltage = (
            self.v.get() * (1 - self.dv.get()) + self.u.get() + self.bias.get()
        )
        if new_voltage > self.vth.get():
            self.spikes = True

            # TODO: Mention discrepancy between lava-nc.org documentation
            # and github.com/lava-nc LIF object.
            # Documentation says voltage gets set to:
            # self.v = self.v - self.s_out * self.vth
            # Github object/code says voltage is reset to 0.

            # Reset voltage based on output voltage.
            self.v = V(0.0)
        else:
            self.spikes = False
            self.v = V(new_voltage)


class Bias:
    """Creates a bias object that contains a float, and a get() function."""

    # pylint: disable=R0903

    def __init__(self, bias: float) -> None:
        if isinstance(bias, float):
            self.bias = bias
        else:
            raise Exception(
                "Error, bias type is not float, instead, it is:"
                + f"{type(bias)}"
            )

    def get(self) -> float:
        """Returns the bias value as a float."""
        return self.bias


class Du:
    """Creates a du object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, du: float) -> None:
        if isinstance(du, float):
            self.du = du
        else:
            raise Exception(
                "Error, du type is not float, instead, it is:" + f"{type(du)}"
            )

    def get(self) -> float:
        """Returns the du value as a float."""
        return self.du


class Dv:
    """Creates a dv object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, dv: float) -> None:
        if isinstance(dv, float):
            self.dv = dv
        else:
            raise Exception(
                "Error, dv type is not float, instead, it is:" + f"{type(dv)}"
            )

    def get(self) -> float:
        """Returns the dv value as a float."""
        return self.dv


class U:
    """Creates a u object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, u: float) -> None:
        if isinstance(u, float):
            self.u = u
        else:
            raise Exception(
                "Error, u type is not float, instead, it is:" + f"{type(u)}"
            )

    def get(self) -> float:
        """Returns the u (current) value as a float."""
        return self.u


class V:
    """Creates a v object that contains a float, and a get() fvnction."""

    # pylint: disable=R0903
    def __init__(self, v: float) -> None:
        if isinstance(v, float):
            self.v = v
        else:
            raise Exception(
                "Error, v type is not float, instead, it is:" + f"{type(v)}"
            )

    def get(self) -> float:
        """Returns the v (voltage) value as a float."""
        return self.v


class Vth:
    """Creates a vth object that contains a float, and a get() function."""

    # pylint: disable=R0903
    def __init__(self, vth: float) -> None:
        if isinstance(vth, float):
            self.vth = vth
        else:
            raise Exception(
                "Error, vth type is not float, instead, it is:"
                + f"{type(vth)}"
            )

    def get(self) -> float:
        """Returns the vth (threshold voltage) value as a float."""
        return self.vth


# pylint: disable=R0912
def print_neuron_properties(
    neurons: List[Any],
    static: bool,
    ids: List[Any] = None,
    spikes: List[Any] = None,
) -> None:
    """Prints the neuron properties in human readable format.

    :param neurons:
    :param static:
    :param ids:  (Default value = None)
    :param spikes:  (Default value = None)
    """
    spacing = 5
    if ids is not None:
        for x in ids:
            print(f"{str(x) : <{spacing+5}}", end=" ")

    if spikes is not None:
        print("")
        for x in spikes:
            print(f"spk={x : <{spacing+1}}", end=" ")
    if static:
        print("")
        for x in neurons:
            print(f"du={str(round(x.du.get(),2)) : <{spacing+2}}", end=" ")
        print("")
        for x in neurons:
            print(f"dv={str(round(x.dv.get(),2)) : <{spacing+2}}", end=" ")
        print("")
        for x in neurons:
            if isinstance(x, LIF_neuron):
                print(
                    f"bias={str(round(x.bias.get(),2)) : <{spacing}}", end=" "
                )
            elif isinstance(x, LIF):
                print(
                    f"bias={str(round(x.bias_mant.get(),2)) : <{spacing}}",
                    end=" ",
                )
            else:
                print(type(x))
                raise Exception("Unsupported neuron type.")
        print("")
        for x in neurons:
            print(f"vth={str(round(x.vth.get(),2)) : <{spacing+1}}", end=" ")
        print("\n")
    else:
        print("")
        for x in neurons:
            print(
                f"u={str(round_if_array(x.u.get())) : <{spacing+3}}", end=" "
            )
        print("")
        for x in neurons:
            print(
                f"v={str(round_if_array(x.v.get())) : <{spacing+3}}", end=" "
            )
    print("")


def round_if_array(value: Any) -> float:
    """Rounds an incoming value up to 2 decimals and unpacks array if lif
    neuron property is returns as array.

    :param value:
    """
    if isinstance(value, np.ndarray):
        return round(value[0], 2)
    return round(value, 2)


def print_neuron_properties_per_graph(
    G: nx.DiGraph, static: Any, t: int
) -> None:
    """Prints bias,du,dv,vth of neuron.

    Supports both lava and networkx neurons.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param static:
    """
    lava_neurons = []
    nx_neurons = []
    for node in G.nodes:
        # TODO: also convert lava LIF function of t
        lava_neurons.append(G.nodes[node]["lava_LIF"])
        nx_neurons.append(G.nodes[node]["nx_LIF"][t])

    print("Lava neuron values:")
    print_neuron_properties(
        lava_neurons, static=static, ids=G.nodes, spikes=None
    )
    print("Networkx neuron values:")
    print_neuron_properties(
        nx_neurons, static=static, ids=G.nodes, spikes=None
    )
