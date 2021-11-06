from EMBEDR.human_round import human_round
import EMBEDR.plotting_utility as putl
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

def sweep_lineplot(hyperparam_array,
                   values,
                   log_hp=True,
                   log_values=True,
                   fig=None,
                   fig_size=(12, 5),
                   axis=None,
                   show_border=True,
                   threshold=-3,
                   threshold_color='lightgrey',
                   threshold_width=3,
                   line_color='k',
                   line_width=None,
                   line_alpha=None,
                   plot_median=True,
                   median_kws=None,
                   plot_percentiles=[90],
                   perc_kws=None,
                   xticks=None,
                   xticklabels=None,
                   xlabel=r"$k_{\mathrm{Eff}}$",
                   xlabel_size=16,
                   xlim=None,
                   yticks=None,
                   yticklabels=None,
                   ylabel=r"EMBEDR $p$-Value",
                   ylabel_size=16,
                   ylim=None,
                   title=r"EMBEDR Sweep: per-sample $p$-Value",
                   title_size=16,
                   title_pad=None):
    
    if isinstance(values, dict):
        values = np.asarray([values[key] for key in hyperparam_array])
    n_hp, n_samples = values.shape

    if xticks is None:
        xticks = hyperparam_array

    if yticks is None:
        yticks = -1 * np.array([0, 2, 3, 4, 5])

    if log_hp:
        hyperparam_array = np.log10(hyperparam_array)

        xticks = np.log10(xticks)
        if xticklabels is None:
            xticklabels = [int(human_round(10**xt)) for xt in xticks]

    if log_values:
        values = np.log10(values)

        if yticklabels is None:
            yticklabels = [f"{10.**yt:.1g}" for yt in yticks]

    if fig is None:
        fig = plt.figure(figsize=fig_size)

    if axis is None:
        axis = fig.add_subplot(111)
        spine_alpha = 1 if show_border else 0
        axis = putl.make_border_axes(axis, spine_alpha=spine_alpha)

    if line_width is None:
        line_width = 0.2 + 10 / n_samples
    if line_alpha is None:
        line_alpha = 0.2 + 10 / n_samples

    _ = axis.plot(hyperparam_array, values, color=line_color,
                  lw=line_width, alpha=line_alpha)

    ylim = axis.get_ylim()

    _ = axis.axhline(threshold, lw=threshold_width,
                     color=threshold_color, zorder=-10)

    axis.set_ylim(ylim)

    if plot_median:
        med_val = np.median(values, axis=1)
        if median_kws is None:
            median_kws = {'marker': 's',
                          'markersize': 6,
                          'markeredgecolor': 'w',
                          'color': line_color,
                          'lw': 1}
        _ = axis.plot(hyperparam_array, med_val, **median_kws)

    if plot_percentiles:
        perc_val = np.percentile(values, plot_percentiles, axis=1).T.squeeze()

        if perc_kws is None:
            perc_kws = {'color': line_color, 'lw': 2}
        _ = axis.plot(hyperparam_array, perc_val, **perc_kws)

    axis.set_xticks(xticks)
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    else:
        axis.set_xticklabels(xticklabels)

    axis.set_yticks(yticks)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)
    else:
        axis.set_yticklabels(yticklabels)

    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=xlabel_size)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=ylabel_size)

    if title is not None:
        axis.set_title(title, fontsize=title_size, pad=title_pad)

    return axis


def sweep_lineplot_byCat(hyperparam_array,
                         values,
                         metadata,
                         label,
                         label_cmap='husl',
                         labels_2_show='all',
                         n_cols=3,
                         n_rows=None,
                         fig=None,
                         fig_size=(4, 2),
                         axes_sharey=True,
                         show_border=True,
                         xticks=None,
                         xticklabels=None,
                         xlabel=r"$k_{\mathrm{Eff}}$",
                         xlabel_size=16,
                         xlim=None,
                         yticks=None,
                         yticklabels=None,
                         ylabel=r"EMBEDR $p$-Value",
                         ylabel_size=16,
                         ylim=None,
                         verbose=False,
                         **kwargs):

    if isinstance(values, dict):
        values = np.asarray([values[key] for key in hyperparam_array])
    n_hp, n_samples = values.shape

    [labels,
     label_counts,
     long_labels,
     lab_2_idx_map,
     label_cmap] = putl.process_categorical_label(metadata, label,
                                                  cmap=label_cmap)

    n_labels = len(label_counts)

    if labels_2_show == 'all':
        labels_2_show = label_counts.index.values

    if n_rows is None:
        n_rows = int(np.ceil(n_labels / n_cols))

    if fig is None:
        fig_size = (n_cols * fig_size[0], n_rows * fig_size[1])
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4, n_rows * 2),
                                 sharey=axes_sharey)

    for rNo, axes_row in enumerate(axes):
        for cNo, axis in enumerate(axes_row):
            spine_alpha=None
            if rNo * n_cols + cNo >= n_labels:
                spine_alpha = 1 if show_border else 0
            axes[rNo, cNo] = putl.make_border_axes(axis,
                                                   spine_alpha=spine_alpha)

    for lNo, label in enumerate(label_counts.index):
        rowNo = int(lNo / n_cols)
        colNo = lNo % n_cols
        axis = axes[rowNo, colNo]
        
        good_idx = labels == label
        n_labs = sum(good_idx)
        if verbose:
            print(f"There are {n_labs} samples with label = {label}")
        
        title = long_labels[lNo].title()
        if "Multipotent" in title:
            title = " ".join(title.split(" Multipotent "))

        axis = sweep_lineplot(hyperparam_array,
                              values[:, good_idx],
                              fig=fig,
                              axis=axis,
                              line_color=label_cmap[lab_2_idx_map[label]],
                              xlabel_size=12,
                              ylabel_size=12,
                              title=title,
                              title_size=10,
                              title_pad=-8)

    fig.tight_layout()

    return axes
    
    