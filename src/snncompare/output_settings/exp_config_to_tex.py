"""Outputs data on the experiment config to files importable by latex."""
from typing import List

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config


@typechecked
def get_min_max_synaptic_excitation_amplitude_lines(
    *, exp_config: Exp_config
) -> List[str]:
    """Returns 2 lines with the minimum and maximum synaptic weight increase
    per simulation, that is found in the used exp_config."""
    lines: List[str] = []
    if "change_syn" in exp_config.radiations.__dict__.keys():
        if exp_config.radiations.excitatory == [
            True
        ] and exp_config.radiations.inhibitory == [False]:
            lines.append(
                "min_synaptic_amplitude = "
                + f"{min(exp_config.radiations.amplitude)}"
            )
            lines.append(
                "max_synaptic_amplitude = "
                + f"{max(exp_config.radiations.amplitude)}"
            )
    else:
        raise ValueError(
            "Error, expected the synaptic excitation to be in the exp_config."
        )
    return lines


@typechecked
def get_min_max_synaptic_excitation_lines(
    *, exp_config: Exp_config
) -> List[str]:
    """Returns 2 lines with the minimum and maximum probability of a synaptic
    weight increase occurring per synapse per timestep, that is found in the
    used exp_config.

    TODO: probability per timestep or per synapse.
    """
    lines: List[str] = []
    if "change_syn" in exp_config.radiations.__dict__.keys():
        if exp_config.radiations.excitatory == [
            True
        ] and exp_config.radiations.inhibitory == [False]:
            lines.append(
                "min_synaptic_see_probability_procent = "
                + f"{min(exp_config.radiations.probability_per_t)}"
            )
            lines.append(
                "max_synaptic_see_probability_procent = "
                + f"{max(exp_config.radiations.probability_per_t)}"
            )
    else:
        raise ValueError(
            "Error, expected the synaptic excitation to be in the exp_config."
        )
    return lines


# Chapter: Abstract, minimum probability of neuron death occurrence, per
# neuron, per timestep.
# Chapter: Abstract, maximum probability of neuron death occurrence, per
# neuron, per timestep.
@typechecked
def get_min_max_neuron_death_probability_procent(
    *, exp_config: Exp_config
) -> List[str]:
    """Returns 2 lines with the minimum and maximum probability of a neuron
    death occurring per simulation, that is found in the used exp_config.

    TODO: verify procent is procent.
    """
    lines: List[str] = []
    if "neuron_death" in exp_config.radiations.__dict__.keys():
        if exp_config.radiations.excitatory == [
            False
        ] and exp_config.radiations.inhibitory == [True]:
            lines.append(
                "min_neuron_death_probability_procent = "
                + f"{min(exp_config.radiations.probability_per_t)}"
            )
            lines.append(
                "min_neuron_death_probability_procent = "
                + f"{max(exp_config.radiations.probability_per_t)}"
            )
    else:
        raise ValueError(
            "Error, expected the neuron death to be in the exp_config."
        )
    return lines


@typechecked
def get_neuron_death_probabilities_procent(
    *, exp_config: Exp_config
) -> List[str]:
    """Returns 1 latex line with the latex code for the neuron death
    probabilities."""
    lines: List[str] = []
    if "neuron_death" in exp_config.radiations.__dict__.keys():
        if exp_config.radiations.excitatory == [
            False
        ] and exp_config.radiations.inhibitory == [True]:
            lines.append(f"${exp_config.radiations.probability_per_t}$")
    else:
        raise ValueError(
            "Error, expected the neuron death to be in the exp_config."
        )
    return lines


@typechecked
def get_nr_of_synaptic_weight_increases(
    *, exp_config: Exp_config
) -> List[str]:
    """Returns 2 lines with the minimum and maximum probability of a synaptic
    weight increase occurring per synapse per timestep, that is found in the
    used exp_config.

    TODO: probability per timestep or per synapse.
    """
    lines: List[str] = []
    if "change_syn" in exp_config.radiations.__dict__.keys():
        if exp_config.radiations.excitatory == [
            True
        ] and exp_config.radiations.inhibitory == [False]:
            lines.append(
                "nr_of_synaptic_weight_increases = "
                + f"{exp_config.radiations.nr_of_synaptic_weight_increases}"
            )

            # TODO: verify
            weight_increases: List[int] = list(
                range(
                    0,
                    len(exp_config.radiations.nr_of_synaptic_weight_increases)
                    + 1,
                )
            )

            lines.append(
                "indices_nr_of_synaptic_weight_increases = "
                + f"{weight_increases}"
            )

            min_weight_increase: float = min(
                exp_config.radiations.nr_of_synaptic_weight_increases
            )
            lines.append(
                "min_nr_of_synaptic_weight_increases = "
                + f"{min_weight_increase}"
            )

            max_weight_increase: float = max(
                exp_config.radiations.nr_of_synaptic_weight_increases
            )
            lines.append(
                "max_nr_of_synaptic_weight_increases = "
                + f"{max_weight_increase}"
            )
    else:
        raise ValueError(
            "Error, expected the synaptic excitation to be in the exp_config."
        )
    return lines


# Chapter: Methodology, number of synaptic excitation events per simulation.
# Chapter: Methodology, the average synaptic weight increase in percentages,
# per simulation.

# Chapter: Methodology, maximum neuron death probability per simulation in this
#  experiment config.

# Chapter: Methodology, all neuron death probabilities per simulation.

# Chapter: Methodology: used m_values in experiment.
# Chapter: Methodology: list of used pseudo random seeds in experiment.

# Chapter: Methodology: Table with Experiment run config settings.

# Chapter: Methodology: The number of scores per adaptation type, per
# radiation setting.

# TODO: Chapter: Results: The number of boolean pass/fails per adapted snn per
#  dot in the boxplot.
