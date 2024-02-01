
import datetime
import math
from typing import Optional
from IPython.display import HTML
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

def select(
        data: xarray.Dataset,
        variable: str,
        level: Optional[int] = None,
        max_steps: Optional[int] = None
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
        data: xarray.Dataset,
        center: Optional[float] = None,
        robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

def plot_data(
        data: dict[str, xarray.Dataset],
        fig_title: str,
        plot_size: float = 5,
        robust: bool = False,
        cols: int = 4
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                 plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)

    def update(frame):
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

    ani = animation.FuncAnimation(
        fig=figure, func=update, frames=max_steps, interval=250)
    plt.close(figure.number)
    return HTML(ani.to_jshtml())


def plot_predictions(predictions, eval_targets):
    # @title Choose predictions to plot

    plot_pred_variable = widgets.Dropdown(
        options=predictions.data_vars.keys(),
        value="2m_temperature",
        description="Variable")
    plot_pred_level = widgets.Dropdown(
        options=predictions.coords["level"].values,
        value=500,
        description="Level")
    plot_pred_robust = widgets.Checkbox(value=True, description="Robust")
    plot_pred_max_steps = widgets.IntSlider(
        min=1,
        max=predictions.dims["time"],
        value=predictions.dims["time"],
        description="Max steps")

    widgets.VBox([
        plot_pred_variable,
        plot_pred_level,
        plot_pred_robust,
        plot_pred_max_steps,
        widgets.Label(value="Run the next cell to plot the predictions. Rerunning this cell clears your selection.")
    ])

    # @title Plot predictions

    plot_size = 5
    plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)

    data = {
        "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps),
                         robust=plot_pred_robust.value),
        "Predictions": scale(select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps),
                             robust=plot_pred_robust.value),
        "Diff": scale((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps) -
                       select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                      robust=plot_pred_robust.value, center=0),
    }
    fig_title = plot_pred_variable.value
    if "level" in predictions[plot_pred_variable.value].coords:
        fig_title += f" at {plot_pred_level.value} hPa"

    plot_data(data, fig_title, plot_size, plot_pred_robust.value)
