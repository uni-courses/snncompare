"""Creates plots with the p-values."""
from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from simplt.export_plot import create_target_dir_if_not_exists
from simplt.line_plot.line_plot import plot_multiple_lines
from typeguard import typechecked

from snncompare.exp_config import Exp_config


# pylint: disable=R0914
@typechecked
def create_annova_plot(
    *,
    create_p_values: bool,
    exp_config: Exp_config,
    lines: Dict[float, Dict[str, Dict[float, float]]],
    title: str,
) -> None:
    """Creates plot for annova."""
    multiple_y_series: List[List[float]] = []
    lineLabels: List[str] = []
    default_p_value: float = 0.05
    output_dir: str = "latex/Images/p_values"
    create_target_dir_if_not_exists(some_path=output_dir)

    for rad_probability in list(lines.keys()):
        for adaptation_mechanism, xy_values in lines[rad_probability].items():
            multiple_y_series.append(list(xy_values.values()))
            lineLabels.append(
                f"rad:{rad_probability*100} [%], {adaptation_mechanism}"
            )  # add a label for each dataseries
            single_x_series = list(xy_values.keys())

    if create_p_values:
        filename: str = f"{exp_config.unique_id}_p_vals"
        y_axis_label: str = "Probability [-]"
        multiple_y_series.append(
            [default_p_value] * len(multiple_y_series[-1])
        )
        lineLabels.append(
            "Significance Threshold"
        )  # add a label for each dataseries

    else:
        filename = f"{exp_config.unique_id}_f_vals"
        y_axis_label = "Effect size [-]"
    some_list = np.array(multiple_y_series, dtype=float)
    pprint("some_list")
    pprint(some_list)

    plot_multiple_lines(
        extensions=[".png"],
        filename=filename,
        label=lineLabels,
        legendPosition=0,
        output_dir=output_dir,
        x=single_x_series,
        x_axis_label="redundancy [Backup Neurons]",
        y_axis_label=y_axis_label,
        y_series=some_list,
        title=title,
        x_ticks=single_x_series,
    )


# pylint: disable=R0914
@typechecked
def create_stat_sign_plot(
    *,
    exp_config: Exp_config,
    y_series: Dict[str, List[float]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """Creates the P-values for a logistic regression with various degrees of
    redundancy.."""
    output_dir: str = "latex/Images/p_values"
    create_target_dir_if_not_exists(some_path=output_dir)

    adaptation_scores: Dict[
        str, Dict[int, List[float]]
    ] = get_adapatation_data(
        exp_config=exp_config,
        y_series=y_series,
    )

    (
        adap_coefficients,
        adap_p_values,
    ) = annova_compute_p_values_per_adaptation_type(
        adaptation_scores=adaptation_scores
    )
    return adap_coefficients, adap_p_values


# pylint: disable=R0914
@typechecked
def output_p_val_plot(
    *,
    adap_p_values: Dict[str, Dict[int, float]],
    adap_coefficients: Dict[str, Dict[int, float]],
    output_dir: str,
    create_p_values: bool,
    default_p_value: float,
    img_index: int,
    lineLabels: List[str],
    multiple_y_series: List[List[float]],
    title: str,
    unique_id_exp: str,
) -> None:
    """Computes the p-values per adaptation type."""
    if create_p_values:
        filename: str = f"{unique_id_exp}_p_val_{img_index}"
        y_axis_label: str = "Probability [-]"
        for adaptation_name in list(adap_p_values.keys()):
            multiple_y_series.append(
                list(adap_p_values[adaptation_name].values())
            )
            lineLabels.append(
                f"p_val:{adaptation_name}"
            )  # add a label for each dataseries
            single_x_series = list(adap_p_values[adaptation_name].keys())
        multiple_y_series.append(
            [default_p_value] * len(multiple_y_series[-1])
        )
        lineLabels.append(
            "Significance Threshold"
        )  # add a label for each dataseries

    else:
        filename = f"{unique_id_exp}_coeff_{img_index}"
        y_axis_label = "Effect size [-]"
        for adaptation_name in list(adap_coefficients.keys()):
            multiple_y_series.append(
                list(adap_coefficients[adaptation_name].values())
            )
            lineLabels.append(
                f"coef:{adaptation_name}"
            )  # add a label for each dataseries
            single_x_series = list(adap_coefficients[adaptation_name].keys())

    some_list = np.array(multiple_y_series, dtype=float)

    plot_multiple_lines(
        extensions=[".png"],
        filename=filename,
        label=lineLabels,
        legendPosition=0,
        output_dir=output_dir,
        x=single_x_series,
        x_axis_label="redundancy [Backup Neurons]",
        y_axis_label=y_axis_label,
        y_series=some_list,
        title=title,
        x_ticks=single_x_series,
    )


@typechecked
def compute_p_values_per_adaptation_type(
    *,
    adaptation_scores: Dict[str, Dict[int, List[float]]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """Computes the p-values per adaptation type."""
    adap_coefficients: Dict[str, Dict[int, float]] = {}
    adap_p_values: Dict[str, Dict[int, float]] = {}
    for (
        adaptation_type,
        combined_redundancy_scores,
    ) in adaptation_scores.items():
        adap_coefficients[adaptation_type] = {}
        adap_p_values[adaptation_type] = {}

        for redundancy_scores in build_consecutive_dicts(
            input_dict=combined_redundancy_scores
        ):
            # Combine the scores and create the target variable
            combined_scores = np.concatenate(list(redundancy_scores.values()))

            fertilizer_amounts: List[int] = []
            for redundancy, score_per_redundancy in redundancy_scores.items():
                fertilizer_amounts += [redundancy] * len(score_per_redundancy)
            # Create a pandas DataFrame
            data = pd.DataFrame(
                {"Scores": combined_scores, "Fertilizer": fertilizer_amounts}
            )

            # Add a constant column for the intercept
            data = sm.add_constant(data)

            # Perform logistic regression
            logit = sm.Logit(data["Scores"], data[["const", "Fertilizer"]])
            result = logit.fit()

            # TODO: get results from other than protected class (_results).
            # pylint:disable=W0212
            coefficient: float = result._results.params[1]
            result.summary()
            p_value: float = result._results._cache["pvalues"][1]

            adap_p_values[adaptation_type][
                max(redundancy_scores.keys())
            ] = p_value
            adap_coefficients[adaptation_type][
                max(redundancy_scores.keys())
            ] = coefficient

    return adap_coefficients, adap_p_values


@typechecked
def build_consecutive_dicts(
    *, input_dict: Dict[int, List[float]]
) -> List[Dict[int, List[float]]]:
    """Build a list of consecutive dictionaries by progressively adding keys
    from the input dictionary.

    Args:
        input_dict (Dict): The input dictionary.

    Returns:
        List[Dict]: A list of consecutive dictionaries, where each dictionary
         contains a subset of keys from the input dictionary.
    """
    result_list: List[Dict[int, List[float]]] = []
    keys = list(input_dict.keys())
    keys.sort()

    # Always include at least the first dict keys.
    for i in range(2, len(keys) + 1):
        partial_dict = {key: input_dict[key] for key in keys[:i]}
        result_list.append(partial_dict)

    return result_list


@typechecked
def get_adapatation_data(
    *, exp_config: Exp_config, y_series: Dict[str, List[float]]
) -> Dict[str, Dict[int, List[float]]]:
    """Gets the adaptation scores for the experiment configuration."""
    adaptation_types: Dict[str, List[int]] = get_sorted_adaptation_types(
        exp_config=exp_config
    )

    adaptation_scores: Dict[
        str, Dict[int, List[float]]
    ] = get_adaptation_scores(
        adaptation_types=adaptation_types,
        y_series=y_series,
    )
    return adaptation_scores


@typechecked
def get_adaptation_scores(
    *, adaptation_types: Dict[str, List[int]], y_series: Dict[str, List[float]]
) -> Dict[str, Dict[int, List[float]]]:
    """Returns dict with the boolean adaptation scores per adaptation type per
    redundancy."""
    adaptation_scores: Dict[str, Dict[int, List[float]]] = {}
    for adaptation_type, redundancies in adaptation_types.items():
        adaptation_scores[adaptation_type] = {}
        for redundancy in redundancies:
            adaptation_name: str = f"{adaptation_type}_{redundancy}"
            scores_with_adaptation: List[float] = get_snn_performance(
                adaptation_name=adaptation_name, y_series=y_series
            )
            adaptation_scores[adaptation_type][
                redundancy
            ] = scores_with_adaptation
        score_without_adaptation: List[float] = get_snn_performance(
            adaptation_name="rad_snn_algo_graph", y_series=y_series
        )
        adaptation_scores[adaptation_type][0] = score_without_adaptation
    return adaptation_scores


def get_snn_performance(
    *, adaptation_name: str, y_series: Dict[str, List[float]]
) -> List[float]:
    """Returns the y-values for a given adaptation type."""
    for col_name, y_vals in y_series.items():
        if col_name == adaptation_name:
            return y_vals
    raise ValueError("Error, expected to find adaptation name.")


@typechecked
def get_sorted_adaptation_types(
    *, exp_config: Exp_config
) -> Dict[str, List[int]]:
    """Returns the different adaptation types, sorted on redundancy level,
    small to large."""
    adaptation_types: Dict[str, List[int]] = {}

    # Group adaptations by type
    for run_config_adaptation in exp_config.adaptations:
        adaptation_type = run_config_adaptation.adaptation_type
        if adaptation_type not in adaptation_types:
            adaptation_types[adaptation_type] = []
        adaptation_types[adaptation_type].append(
            run_config_adaptation.redundancy
        )

    # Sort adaptations within each type by redundancy size
    for adaptation_type, redundancies in adaptation_types.items():
        # Sort the list from small to large.
        redundancies.sort()
    return adaptation_types


@typechecked
def annova_compute_p_values_per_adaptation_type(
    *,
    adaptation_scores: Dict[str, Dict[int, List[float]]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """Computes the p-values per adaptation type."""
    adap_coefficients: Dict[str, Dict[int, float]] = {}
    adap_p_values: Dict[str, Dict[int, float]] = {}
    for (
        adaptation_type,
        combined_redundancy_scores,
    ) in adaptation_scores.items():
        adap_coefficients[adaptation_type] = {}
        adap_p_values[adaptation_type] = {}

        for redundancy_scores in build_consecutive_dicts(
            input_dict=combined_redundancy_scores
        ):
            f_statistic, p_value = stats.f_oneway(
                *list(redundancy_scores.values())
            )

            adap_p_values[adaptation_type][
                max(redundancy_scores.keys())
            ] = p_value
            adap_coefficients[adaptation_type][
                max(redundancy_scores.keys())
            ] = f_statistic

    return adap_coefficients, adap_p_values
