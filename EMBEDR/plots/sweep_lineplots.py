from EMBEDR.human_round import human_round
import EMBEDR.plotting_utility as putl
from EMBEDR.plotting_utility import make_border_axes as mbax
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np


class SweepLineplot(object):

    FIG_KWDS    = dict(figsize=(12, 5))
    GRID_KWDS   = dict(which='major', axis='both', alpha=0)
    THRESH_KWDS = dict(color='lightgrey', lw=3, zorder=-1)
    MEDIAN_KWDS = dict(marker='s', markersize=6, markeredgecolor='w', lw=1)
    PERC_KWDS   = dict(lw=2)

    def __init__(self,
                 hyperparam_array,
                 values_dict,
                 fig=None,
                 fig_kwds=None,
                 fig_pad=None,
                 fig_ppad=None,
                 axis=None,
                 show_border=True,
                 axis_kwds=None,
                 grid_kwds=None,
                 log_hyperparams=True,
                 log_values=True,
                 threshold=-3,
                 threshold_kwds=None,
                 line_color='k',
                 line_width=None,
                 line_alpha=None,
                 line_kwds=None,
                 plot_median=True,
                 median_kwds=None,
                 plot_percentiles=[90],
                 perc_kwds=None,
                 kEff_dict=None,
                 hp_2_xtick_map=None,
                 xticks=None,
                 xticklabels=None,
                 xlabel=None,
                 xlabel_size=16,
                 xlim=None,
                 yticks=None,
                 yticklabels=None,
                 ylabel=None,
                 ylabel_size=16,
                 ylim=None,
                 title=r"EMBEDR Sweep: per-sample $p$-Value",
                 title_size=16,
                 title_pad=None,
                 title_kwds=None,
                 cite_EMBEDR=True,
                 cite_kwds=None):

        self.hp_array = np.sort(hyperparam_array).squeeze()

        self.values = values_dict
        if isinstance(self.values, dict):
            self.values = np.asarray([self.values[key]
                                      for key in self.hp_array])
        self.n_hp, self.n_samples = self.values.shape

        self.fig      = fig
        self.fig_kwds = self.FIG_KWDS
        self.fig_kwds.update({} if fig_kwds is None else fig_kwds.copy())
        self.fig_pad  = fig_pad
        self.fig_ppad = fig_ppad

        self.axis        = axis
        self.show_border = show_border
        self.axis_kwds   = {} if axis_kwds is None else axis_kwds.copy()

        self.grid_kwds = self.GRID_KWDS
        self.grid_kwds.update({} if grid_kwds is None else grid_kwds.copy())

        self.log_hp     = log_hyperparams
        self.log_values = log_values

        self.threshold      = threshold
        self.threshold_kwds = self.THRESH_KWDS
        self.threshold_kwds.update({} if threshold_kwds is None 
                                   else threshold_kwds)

        self.line_color = line_color
        self.line_width = line_width
        self.line_alpha = line_alpha
        self.line_kwds  = {} if line_kwds is None else line_kwds.copy()

        self.plot_median = plot_median
        self.median_kwds  = self.MEDIAN_KWDS
        self.median_kwds['color'] = self.line_color
        self.median_kwds.update({} if median_kwds is None
                                else median_kwds.copy())

        self.plot_percentiles = np.asarray(plot_percentiles)
        self.perc_kwds         = self.PERC_KWDS
        self.perc_kwds['color'] = self.line_color
        self.perc_kwds.update({} if perc_kwds is None else perc_kwds.copy())

        if (kEff_dict is None) and (hp_2_xtick_map is None):
            self.hp_2_xtick_map = {hp: hp for hp in self.hp_array}
        elif kEff_dict is not None:
            self.hp_2_xtick_map = kEff_dict
        elif hp_2_xtick_map is not None:
            self.hp_2_xtick_map = hp_2_xtick_map
        else:
            if kEff_dict != hp_2_xtick_map:
                err_str  = f"Values provided for both `kEff_dict` and"
                err_str += f" `hp_2_xtick_map` but they do not match! Cannot"
                err_srt += f" set hyperparameter to xticklabel map!"
                raise ValueError(err_str)
            else:
                self.hp_2_xtick_map = kEff_dict

        self.xticks      = xticks
        self.xticklabels = xticklabels
        self.xlabel      = xlabel
        self.xlabel_size = xlabel_size
        self.xlim        = xlim

        self.yticks      = yticks
        self.yticklabels = yticklabels
        self.ylabel      = ylabel
        self.ylabel_size = ylabel_size
        self.ylim        = ylim

        if self.xlabel is None:
            self.xlabel = r"$k_{\mathrm{Eff}}$"

        if self.yticks is None:
            self.yticks = [0, -1, -2, -3, -5]

        if self.yticklabels is None:
            if self.log_values:
                self.yticklabels = [f"{10**yt:.2g}" for yt in self.yticks]

        self.title      = title
        self.title_size = title_size
        self.title_pad  = title_pad
        self.title_kwds = {} if title_kwds is None else title_kwds.copy()

        self.cite_EMBEDR = cite_EMBEDR
        self.cite_kwds   = {} if cite_kwds is None else cite_kwds.copy()

    def plot(self):

        if self.show_border:
            self.axis_kwds['spine_alpha'] = 1

        if (self.fig is None) and (self.axis is None):
            self.fig = plt.figure(**self.fig_kwds)
            self.axis = mbax(self.fig.add_subplot(111),
                             xticks=self.xticks,
                             xticklabels=self.xticklabels,
                             yticks=self.yticks,
                             yticklabels=self.yticklabels,
                             **self.axis_kwds)
        elif self.fig is None:
            self.fig = self.axis.figure
        elif self.axis is None:
            self.axis = mbax(self.fig.add_subplot(111),
                             xticks=self.xticks,
                             xticklabels=self.xticklabels,
                             yticks=self.yticks,
                             yticklabels=self.yticklabels,
                             **self.axis_kwds)

        self.axis.grid(**self.grid_kwds)

        self.axis = self._plot()

        if self.xticks is not None:
            self.axis.set_xticks(self.xticks)

        if self.xticklabels is None:
            if self.log_hp:
                self.xticklabels = [f"{10**xt:.2g}" for xt in self.xticks]
            else:
                self.xticklabels = self.xticks
        self.axis.set_xticklabels(self.xticklabels)

        self.axis.set_xlim(self.xlim)

        if self.yticks is not None:
            self.axis.set_yticks(self.yticks)
        if self.yticklabels is not None:
            self.axis.set_yticklabels(self.yticklabels)

        if self.xlabel is not None:
            self.axis.set_xlabel(self.xlabel, fontsize=self.xlabel_size)
        if self.ylabel is not None:
            self.axis.set_ylabel(self.ylabel, fontsize=self.ylabel_size)

        if self.title is not None:
            self.axis.set_title(self.title,
                                fontsize=self.title_size,
                                pad=self.title_pad,
                                **self.title_kwds)

        return self.axis

    def _plot(self):

        if self.threshold is not None:
            self.axis.axhline(self.threshold, **self.threshold_kwds)

        if self.line_width is None:
            self.line_width = 0.2 + 10 / self.n_samples
        if self.line_alpha is None:
            self.line_alpha = 0.2 + 10 / self.n_samples

        hp_array = self.hp_array
        if self.log_hp:
            hp_array = np.log10(self.hp_array)

        values = self.values
        if self.log_values:
            values = np.log10(self.values)

        _ = self.axis.plot(hp_array, 
                           values, 
                           color=self.line_color,
                           lw=self.line_width,
                           alpha=self.line_alpha,
                           **self.line_kwds)

        if self.plot_median:
            med_vals = np.median(values, axis=1)
            _ = self.axis.plot(hp_array, med_vals, **self.median_kwds)

        if len(self.plot_percentiles) > 0:
            perc_vals = np.percentile(values, 
                                      self.plot_percentiles,
                                      axis=1).T.squeeze()

            _ = self.axis.plot(hp_array, perc_vals, **self.perc_kwds)

        if self.xticks is None:
            if self.n_hp <= 5:
                self.xtick_idx = np.arange(self.n_hp)
            else:
                self.xtick_idx = np.unique(np.linspace(0, self.n_hp, 5))
                self.xtick_idx = np.clip(self.xtick_idx, 0, self.n_hp - 1)
            self.xticks = np.asarray([hp for ii, hp in enumerate(hp_array)
                                      if ii in self.xtick_idx])

        if self.xlim is None:
            xspan = hp_array.max() - hp_array.min()
            self.xlim = (hp_array.min() - xspan * 0.01, 
                         hp_array.max() + xspan * 0.01)

        return self.axis


class SweepLineplot_Category(object):

    GRID_KWDS   = dict(which='major', axis='both', alpha=0)
    THRESH_KWDS = dict(color='lightgrey', lw=3, zorder=-1)
    MEDIAN_KWDS = dict(marker='s', markersize=6, markeredgecolor='w', lw=1)
    PERC_KWDS   = dict(lw=2)

    def __init__(self,
                 hyperparam_array,
                 values_dict,
                 metadata,
                 label,
                 labels_2_show=None,
                 label_kwds=None,
                 fig=None,
                 fig_size_factors=(4, 3),
                 fig_pad=0.1,
                 fig_ppad=0.01,
                 axes=None,
                 n_cols=3,
                 n_rows=None,
                 show_border=True,
                 ax_sharex=False,
                 ax_sharey=True,
                 axes_kwds=None,
                 grid_kwds=None,
                 log_hyperparams=True,
                 log_values=True,
                 threshold=-3,
                 threshold_kwds=None,
                 line_color='k',
                 line_width=None,
                 line_alpha=None,
                 line_kwds=None,
                 plot_median=True,
                 median_kwds=None,
                 plot_percentiles=[90],
                 perc_kwds=None,
                 kEff_dict=None,
                 hp_2_xtick_map=None,
                 xticks=None,
                 xticklabels=None,
                 xlabel=None,
                 xlabel_size=16,
                 xlim=None,
                 yticks=None,
                 yticklabels=None,
                 ylabel=r"EMBEDR $p$-Value",
                 ylabel_size=16,
                 ylim=None,
                 title=None,
                 title_size=16,
                 title_pad=None,
                 title_kwds=None,
                 cite_EMBEDR=True,
                 cite_kwds=None):

        self.hp_array = np.sort(hyperparam_array).squeeze()

        self.values = values_dict
        if isinstance(self.values, dict):
            self.values = np.asarray([self.values[key]
                                      for key in self.hp_array])
        self.n_hp, self.n_samples = self.values.shape

        self.label_name    = label
        self.metadata      = metadata
        self.labels_2_show = labels_2_show
        self.label_kwds    = {} if label_kwds is None else label_kwds.copy()

        self.fig              = fig
        self.fig_size_factors = fig_size_factors
        self.fig_pad          = fig_pad
        self.fig_ppad         = fig_ppad

        self.axes        = axes
        self.n_cols      = n_cols
        self.n_rows      = n_rows
        self.show_border = show_border
        self.axes_kwds   = {} if axes_kwds is None else axes_kwds.copy()
        self.ax_sharex   = ax_sharex
        self.ax_sharey   = ax_sharey

        self.grid_kwds = self.GRID_KWDS
        self.grid_kwds.update({} if grid_kwds is None else grid_kwds.copy())

        self.log_hp     = log_hyperparams
        self.log_values = log_values

        self.threshold      = threshold
        self.threshold_kwds = self.THRESH_KWDS
        self.threshold_kwds.update({} if threshold_kwds is None 
                                   else threshold_kwds)

        self.line_color = line_color
        self.line_width = line_width
        self.line_alpha = line_alpha
        self.line_kwds  = {} if line_kwds is None else line_kwds.copy()

        self.plot_median = plot_median
        self.median_kwds = self.MEDIAN_KWDS.copy()
        self.median_kwds.update({} if median_kwds is None
                                else median_kwds.copy())

        self.plot_percentiles = np.asarray(plot_percentiles)
        self.perc_kwds        = self.PERC_KWDS.copy()
        self.perc_kwds.update({} if perc_kwds is None else perc_kwds.copy())

        if (kEff_dict is None) and (hp_2_xtick_map is None):
            self.hp_2_xtick_map = {hp: hp for hp in self.hp_array}
        elif kEff_dict is not None:
            self.hp_2_xtick_map = kEff_dict
        elif hp_2_xtick_map is not None:
            self.hp_2_xtick_map = hp_2_xtick_map
        else:
            if kEff_dict != hp_2_xtick_map:
                err_str  = f"Values provided for both `kEff_dict` and"
                err_str += f" `hp_2_xtick_map` but they do not match! Cannot"
                err_srt += f" set hyperparameter to xticklabel map!"
                raise ValueError(err_str)
            else:
                self.hp_2_xtick_map = kEff_dict

        self.xticks      = xticks
        self.xticklabels = xticklabels
        self.xlabel      = xlabel
        self.xlabel_size = xlabel_size
        self.xlim        = xlim

        self.yticks      = yticks
        self.yticklabels = yticklabels
        self.ylabel      = ylabel
        self.ylabel_size = ylabel_size
        self.ylim        = ylim

        if self.xlabel is None:
            self.xlabel = r"$k_{\mathrm{Eff}}$"

        if self.yticks is None:
            self.yticks = [0, -1, -2, -3, -5]

        if self.yticklabels is None:
            if self.log_values:
                self.yticklabels = [f"{10**yt:.2g}" for yt in self.yticks]

        self.title      = title
        self.title_size = title_size
        self.title_pad  = title_pad
        self.title_kwds = {} if title_kwds is None else title_kwds.copy()

        self.cite_EMBEDR = cite_EMBEDR
        self.cite_kwds   = {} if cite_kwds is None else cite_kwds.copy()

    def plot(self):

        if self.show_border:
            self.axes_kwds['spine_alpha'] = 1

        self.lab_proc = putl.process_categorical_label(self.metadata,
                                                       self.label_name,
                                                       **self.label_kwds)

        if self.labels_2_show is None:
            self.labels_2_show = self.lab_proc[1].index.values

        self.n_labs = len(self.labels_2_show)

        if self.n_rows is None:
            self.n_rows = int(np.ceil(self.n_labs / self.n_cols))

        self.fig_size = (self.n_cols * self.fig_size_factors[0],
                         self.n_rows * self.fig_size_factors[1])

        if (self.fig is None) and (self.axes is None):
            self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols,
                                               figsize=self.fig_size,
                                               sharex=self.ax_sharex,
                                               sharey=self.ax_sharey)
            # for rowNo in range(self.n_rows):
            #     for colNo in range(self.n_cols):
            #         axis = self.axes[rowNo][colNo]
            #         if rowNo * self.n_cols + colNo >= self.n_labs:
            #             axis = mbax(axis, spine_alpha=0)
            #         else:
            #             axis = mbax(axis, xticks=self.xticks,
            #                         xticklabels=self.xticklabels,
            #                         yticks=self.yticks,
            #                         yticklabels=self.yticklabels,
            #                         **self.axes_kwds)
            #         axis.grid(**self.grid_kwds)
            #         self.axes[rowNo][colNo] = axis

        elif (self.axes is not None):
            if self.fig is None:
                self.fig = self.axes[0][0].figure

            self.axes = np.asarray(self.axes)
            self.n_cols = self.axes.shape[1]
            self.n_rows = int(np.ceil(self.n_labs / self.n_cols))

        elif self.axes is None:
            gridspec = self.fig.add_gridspec(self.n_rows, self.n_cols)
            self.axes = gridspec.subplots(sharex=self.ax_sharex,
                                          sharey=self.ax_sharey)

        if self.axes.ndim != 2:
            self.axes = self.axes.reshape(self.n_rows, self.n_cols)

        for rowNo in range(len(self.axes)):
                for colNo in range(self.n_cols):
                    axis = self.axes[rowNo][colNo]
                    if rowNo * self.n_cols + colNo >= self.n_labs:
                        axis = mbax(axis, spine_alpha=0)
                    else:
                        axis = mbax(axis, xticks=self.xticks,
                                    xticklabels=self.xticklabels,
                                    yticks=self.yticks,
                                    yticklabels=self.yticklabels,
                                    **self.axes_kwds)
                    axis.grid(**self.grid_kwds)
                    self.axes[rowNo][colNo] = axis

        self._hp_array = self.hp_array
        if self.log_hp:
            self._hp_array = np.log10(self.hp_array)

        self._values = self.values
        if self.log_values:
            self._values = np.log10(self.values)

        for rowNo in range(self.n_rows):
            for colNo in range(self.n_cols):
                if rowNo * self.n_cols + colNo >= self.n_labs:
                    continue
                axis = self._plot(rowNo, colNo)

                axis = self._set_axis(axis, rowNo, colNo)

                self.axes[rowNo][colNo] = axis

        self.fig.tight_layout(pad=self.fig_pad)

        return self.axes

    def _plot(self, rowNo, colNo):

        axis = self.axes[rowNo][colNo]

        if self.threshold is not None:
            axis.axhline(self.threshold, **self.threshold_kwds)

        labelNo = rowNo * self.n_cols + colNo
        label = self.labels_2_show[labelNo]

        label_idx = self.lab_proc[0] == label
        label_values = self._values[:, label_idx]

        n_samples = len(label_idx) 

        line_color = self.lab_proc[4][self.lab_proc[3][label]]

        if self.line_width is None:
            line_width = 0.2 + 10 / n_samples
        if self.line_alpha is None:
            line_alpha = 0.2 + 10 / n_samples

        axis.plot(self._hp_array, 
                  label_values, 
                  color=line_color,
                  lw=line_width,
                  alpha=line_alpha,
                  **self.line_kwds)

        if self.plot_median:
            median_kwds = self.median_kwds.copy()
            if 'color' not in median_kwds:
                median_kwds['color'] = line_color

            med_vals = np.median(label_values, axis=1)

            _ = axis.plot(self._hp_array, med_vals, **median_kwds)

        if len(self.plot_percentiles) > 0:
            perc_kwds = self.perc_kwds.copy()
            if 'color' not in perc_kwds:
                perc_kwds['color'] = line_color

            perc_vals = np.percentile(label_values, 
                                      self.plot_percentiles,
                                      axis=1).T.squeeze()

            _ = axis.plot(self._hp_array, perc_vals, **perc_kwds)

        if self.xticks is None:
            if self.n_hp <= 5:
                self.xtick_idx = np.arange(self.n_hp)
            else:
                self.xtick_idx = np.unique(np.linspace(0, self.n_hp, 5))
                self.xtick_idx = np.clip(self.xtick_idx, 0, self.n_hp - 1)
            self.xticks = np.asarray([hp for ii,hp in enumerate(self._hp_array)
                                      if ii in self.xtick_idx])

        if self.xlim is None:
            xspan = self._hp_array.max() - self._hp_array.min()
            self.xlim = (self._hp_array.min() - xspan * 0.01, 
                         self._hp_array.max() + xspan * 0.01)

        if self.title is None:
            title = self.lab_proc[2][self.lab_proc[3][label]]
        else:
            title = self.title
        axis.set_title(title, fontsize=self.title_size,
                       pad=self.title_pad, **self.title_kwds)

        return axis

    def _set_axis(self, axis, rowNo, colNo):

        if self.xticks is not None:
            if rowNo * self.n_cols + colNo >= (self.n_labs - self.n_cols):
                axis.set_xticks(self.xticks)
            else:
                axis.set_xticks([])

        if self.xticklabels is None:
            if self.log_hp:
                self.xticklabels = [f"{10**xt:.4g}" 
                                    for xt in self.xticks]
            else:
                self.xticklabels = self.xticks

        if rowNo * self.n_cols + colNo >= (self.n_labs - self.n_cols):
            axis.set_xticklabels(self.xticklabels)
        else:
            axis.set_xticklabels([])

        axis.set_xlim(self.xlim)

        if self.yticks is not None:
            axis.set_yticks(self.yticks)
        if self.yticklabels is not None:
            if colNo == 0:
                axis.set_yticklabels(self.yticklabels)
            else:
                axis.set_yticklabels([])

        if self.xlabel is not None:
            if rowNo * self.n_cols + colNo >= (self.n_labs - self.n_cols):
                axis.set_xlabel(self.xlabel, fontsize=self.xlabel_size)
            else:
                axis.set_xlabel("")
        if self.ylabel is not None:
            if colNo == 0:
                axis.set_ylabel(self.ylabel, fontsize=self.ylabel_size)
            else:
                axis.set_ylabel("")

        return axis









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
                   median_kwds=None,
                   plot_percentiles=[90],
                   perc_kwds=None,
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
        axis = mbax(axis, spine_alpha=spine_alpha)

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
        if median_kwds is None:
            median_kwds = {'marker': 's',
                          'markersize': 6,
                          'markeredgecolor': 'w',
                          'color': line_color,
                          'lw': 1}
        _ = axis.plot(hyperparam_array, med_val, **median_kwds)

    if plot_percentiles:
        perc_val = np.percentile(values, plot_percentiles, axis=1).T.squeeze()

        if perc_kwds is None:
            perc_kwds = {'color': line_color, 'lw': 2}
        _ = axis.plot(hyperparam_array, perc_val, **perc_kwds)

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
            axes[rNo, cNo] = mbax(axis,
                                                   spine_alpha=spine_alpha)

    for lNo, label in enumerate(label_counts.index):

        if label not in labels_2_show:
            continue

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
                              title_pad=-8
                              **kwargs)

    fig.tight_layout()

    return axes
    
    