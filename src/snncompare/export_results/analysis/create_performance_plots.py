"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""


# Take in exp_config or run_configs
# If exp_config, get run_configs
import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_plots.plot_graphs import export_plot
from snncompare.export_results.analysis.annova_p_plot import (
    create_annova_plot,
    create_stat_sign_plot,
)
from snncompare.export_results.analysis.helper import (
    delete_non_radiation_data,
    get_boxplot_title,
)
from snncompare.export_results.analysis.load_plot_data_from_file import (
    load_boxplot_data,
)
from snncompare.run_config.Run_config import Run_config


# pylint: disable=R0912
# pylint: disable=R0914
@typechecked
def create_performance_plots(
    *,
    completed_run_configs: List[Run_config],
    exp_config: Exp_config,
) -> None:
    """Ensures all performance boxplots are created.

    Loops through the wanted run configs, then gets the radiations that are
    tested.
    Then gets the redundancy levels per radiation that are tested.
    Per radiation, creates a separate plot. Per plot it:
      - creates 4 columns:
        - no adaptation, no radiation
        - adaptation, no radiation
        - no adaptation, radiation
        - adaptation, radiation
    Per column, it contains a list of seeds, each seed has 2 counts.
        - the nr of correct results.
        - the nr of incorrect results.
    the boxplot then turns this data into a boxplot.

    So to get this data,
    - per run config
      -loops through the seeds,
        - per radiation level,
          - per column
            Get the result of a run config and store it in the boxplot data.
    """
    robustness_plot_data: Dict[
        float, Dict[str, List[float]]
    ] = load_boxplot_data(
        completed_run_configs=completed_run_configs, exp_config=exp_config
    )

    p_lines: Dict[float, Dict[str, Dict[float, float]]] = {}
    f_lines: Dict[float, Dict[str, Dict[float, float]]] = {}
    titles: List[str] = []
    for i, rad_probability in enumerate(sorted(robustness_plot_data.keys())):
        title: str = get_boxplot_title(img_index=i, exp_config=exp_config)
        titles.append(title)
        adap_coefficients, adap_p_values = create_stat_sign_plot(
            exp_config=exp_config,
            y_series=robustness_plot_data[rad_probability],
        )
        p_lines[rad_probability] = adap_p_values
        f_lines[rad_probability] = adap_coefficients

    # TODO: get title that does not depend on img id.
    create_annova_plot(
        create_p_values=True,
        exp_config=exp_config,
        data=p_lines,
        titles=titles,
    )
    create_annova_plot(
        create_p_values=False,
        exp_config=exp_config,
        data=f_lines,
        titles=titles,
    )

    # Create boxplot
    for i, rad_probability in enumerate(sorted(robustness_plot_data.keys())):
        # Do not plot the unradiated data in boxplot, as that is trivial 100%.
        adaptation_only_data: Dict[str, List[float]] = copy.deepcopy(
            robustness_plot_data[rad_probability]
        )
        delete_non_radiation_data(
            y_series=adaptation_only_data,
        )
        create_dotted_boxplot(
            y_series=adaptation_only_data,
            filename=f"{exp_config.unique_id}_{i}",
            title=f"Simulated radiation robustness of {titles[i]}",
        )


def create_dotted_boxplot(
    filename: str, y_series: Dict[str, List[float]], title: str
) -> None:
    """Creates a dotted boxplot."""

    # Create a pandas dataframe that stores the y-values to show the measured
    # scores per dataframe.
    dataseries = []
    for col_name, y_vals in y_series.items():
        dataseries.append(
            pd.DataFrame(
                {"": np.repeat(col_name, len(y_vals)), "value": y_vals}
            )
        )
    # Merge the columns into a single figure.
    df = pd.concat(dataseries)
    # boxplot
    sns.boxplot(x="", y="value", data=df)
    # add stripplot
    sns.stripplot(
        x="", y="value", data=df, color="orange", jitter=0.2, size=2.5
    )

    # add title
    plt.title(title, loc="left")

    # Rotate the x-axis labels.
    # ha stands for horizontal alignment, the top right of the x-axis label
    # is positioned below the respective x-tick.
    plt.xticks(rotation=45, ha="right")

    # plt.xlabel(x_axis_label)
    plt.ylabel("Score [-]")

    # Ensure the bottom x-tick labels are within the image.
    plt.tight_layout()

    # Fix scale of boxplot.
    plt.ylim(0, 1.2)

    # Export plot to file.
    export_plot(
        some_plt=plt,
        # some_plt=ax1,
        # filename=f"latex/Images/{filename}",
        filename=filename,
        extensions=["png"],
    )

    # Clear figure data.
    plt.clf()
    plt.close()

    # show the graph
    # plt.show()
