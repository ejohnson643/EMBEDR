from EMBEDR.human_round import human_round
import EMBEDR.plotting_utility as putl
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np


class Scatterplot(object):

    EMBEDR_text_defaults = {'x': 0.02,
                            'y': 0.02,
                            's': "Made with the EMBEDR package.",
                            'fontsize': 6,
                            'ha': 'left',
                            'va': 'bottom'}

    def __init__(self,
                 Y,
                 labels,
                 log_labels=False,
                 axis=None,
                 show_border=True,
                 scatter_sizes=3,
                 scatter_alpha=1,
                 scatter_kwds=None,
                 cmap=None,
                 cmap_kwds=None,
                 xticks=None,
                 xticklabels=None,
                 xlabel=None,
                 yticks=None,
                 yticklabels=None,
                 ylabel=None,
                 label_size=12,
                 plot_order=None,
                 title=None,
                 title_size=16,
                 title_pad=0,
                 show_cbar=True,
                 cbar_ax=None,
                 cbar_ticks=None,
                 cbar_ticklabels=None,
                 cbar_label=None,
                 cite_EMBEDR=True,
                 text_kwds=None,
                 **kwargs):

        self.Y = Y
        self.n_samples, self.n_components = Y.shape
        self.labels = labels

        self.log_labels = log_labels
        if self.log_labels:
            self.labels = np.log10(self.labels)
        
        self.axis        = axis
        self.show_border = show_border

        self.sct_s     = scatter_sizes
        if np.isscalar(self.sct_s):
            self.sct_s = np.ones((self.n_samples)) * self.sct_s
        self.sct_a     = scatter_alpha
        self.sct_kwds  = {} if scatter_kwds is None else scatter_kwds.copy()

        self.cmap      = cmap
        self.cmap_kwds = {} if cmap_kwds is None else cmap_kwds.copy()

        self.xticks      = xticks
        self.xticklabels = xticklabels
        self.xlabel      = xlabel
        self.yticks      = yticks
        self.yticklabels = yticklabels
        self.ylabel      = ylabel
        self.label_size  = label_size

        if plot_order is None:
            self.sort_idx = np.arange(self.n_samples)
        elif plot_order.lower() == 'asc':
            self.sort_idx = np.argsort(labels)
        elif plot_order.lower() == 'desc':
            self.sort_idx = np.argsort(labels)[::-1]
        else:
            raise ValueError(f"Unknown plot ordering '{plot_order}'.")

        self.title      = title
        self.title_size = title_size
        self.title_pad  = title_pad

        self.show_cbar       = show_cbar
        self.cax             = cbar_ax
        self.cbar_ticks      = cbar_ticks
        self.cbar_ticklabels = cbar_ticklabels
        self.cbar_label      = cbar_label

        self.cite_EMBEDR = cite_EMBEDR
        self.text_kwds = self.EMBEDR_text_defaults.copy()
        self.text_kwds.update({} if text_kwds is None else text_kwds.copy())

        return

    def plot(self, **kwargs):

        if self.axis is None:
            self.fig, self.axis = plt.subplots(1, 1, figsize=(8, 6))
        else:
            self.fig = self.axis.figure

        spine_alpha = 1 if self.show_border else 0
        self.axis = putl.make_border_axes(self.axis, spine_alpha=spine_alpha)

        self.axis = self._plot(**kwargs)

        if self.cite_EMBEDR:
            if 'transform' not in self.text_kwds:
                self.text_kwds['transform'] = self.axis.transAxes
            _ = self.axis.text(**self.text_kwds)

        self.fig.tight_layout()

        return self.axis

    def _plot(self, **kwargs):

        h_ax = self.axis.scatter(*self.Y[self.sort_idx, :2].T,
                                 c=self.labels[self.sort_idx],
                                 s=self.sct_s[self.sort_idx],
                                 alpha=self.sct_a,
                                 cmap=self.cmap,
                                 **self.sct_kwds)

        if self.show_cbar:
            self.cax = self.fig.colorbar(h_ax,
                                         ax=self.axis,
                                         cax=self.cax)

            if self.cbar_ticks is not None:
                self.cax.set_ticks(self.cbar_ticks)
                if self.cbar_ticklabels is not None:
                    self.cax.set_ticklabels(self.cbar_ticklabels)

            if self.cbar_label is not None:
                self.cax.set_label(self.cbar_label)
    
        return self.axis


class Scatter_by_pValue(Scatterplot):

    def __init__(self, *args, **kwargs):

        try:
            self.log_labels = kwargs.pop('log_labels')
        except KeyError:
            self.log_labels = True
        kwargs['log_labels'] = self.log_labels

        super(Scatter_by_pValue, self).__init__(*args, **kwargs)

        if self.log_labels:
            self.labels = -self.labels

        if self.xticks is None:
            self.xticks = []
            self.xticklabels = []

        if self.yticks is None:
            self.yticks = []
            self.yticklabels = []

        if self.cmap is None:
            self._cmap = putl.CategoricalFadingCMap(**self.cmap_kwds)
            self.cmap = self._cmap.cmap
            self.cnorm = self._cmap.cnorm
        elif isinstance(self.cmap, putl.CategoricalFadingCMap):
            self._cmap = self.cmap
            self.cmap = self._cmap.cmap
            self.cnorm = self._cmap.cnorm
        else:
            self.cnorm = None

        if (self.cbar_ticks is None):
            if isinstance(self._cmap, putl.CategoricalFadingCMap):
                self.cbar_ticks = self._cmap.change_points
            else:
                self.cbar_ticks = [0, 2, 3, 4, 5]

        if self.cbar_ticklabels is None:
            if self.log_labels:
                self.cbar_ticklabels = [f"{10.**(-ct):.1e}"
                                        for ct in self.cbar_ticks]
            else:
                self.cbar_ticklabels = [f"{ct:.1e}" for ct in self.cbar_ticks]

        if self.cbar_label is None:
            self.cbar_label = r"EMBEDR $p$-Value"

    def _plot(self, **kwargs):

        h_ax = self.axis.scatter(*self.Y[self.sort_idx, :2].T,
                                 c=self.labels[self.sort_idx],
                                 s=self.sct_s[self.sort_idx],
                                 alpha=self.sct_a,
                                 cmap=self.cmap,
                                 norm=self.cnorm,
                                 **self.sct_kwds)

        if self.show_cbar:
            bounds = self.cnorm.boundaries if self.cnorm is not None else None
            self.cax = self.fig.colorbar(h_ax,
                                         ax=self.axis,
                                         cax=self.cax,
                                         boundaries=bounds)
            self.cax.ax.invert_yaxis()

            self.cax.set_ticks(self.cbar_ticks)
            self.cax.set_ticklabels(self.cbar_ticklabels)

            self.cax.ax.tick_params(length=0)
            self.cax.set_label(self.cbar_label)
    
        return self.axis


class Scattergory(Scatterplot):

    LEGEND_KWDS_DEFAULT = dict(bbox_to_anchor=(1.04,1), loc="upper left")

    def __init__(self, Y, label, metadata, **kwargs):

        try:
            self.labels_2_show = kwargs.pop('labels_2_show')
        except KeyError:
            self.labels_2_show = None

        try:
            self.category_kwds = kwargs.pop('category_kwds')
        except KeyError:
            self.category_kwds = {}

        try:
            self.bkgd_label_kwds = kwargs.pop('bkgd_label_kwds')
        except KeyError:
            self.bkgd_label_kwds = {'color': 'lightgrey',
                                    's': 3,
                                    'alpha': 0.5}

        try:
            self.show_legend = kwargs.pop('show_legend')
        except KeyError:
            self.show_legend = True

        self.legend_kwds = self.LEGEND_KWDS_DEFAULT
        try:
            self.legend_kwds.update(kwargs.pop('legend_kwds'))
        except KeyError:
            pass

        super(Scattergory, self).__init__(Y, [], **kwargs)

        if self.cmap is None:
            self.cmap = 'husl'

        print(self.category_kwds)

        out = putl.process_categorical_label(metadata,
                                             label,
                                             cmap=self.cmap,
                                             **self.category_kwds)

        self._labels       = out[0]
        self._label_counts = out[1]
        self.long_labels   = [ll.title() for ll in out[2]]
        self._l2i_map      = out[3]
        self.label_cmap    = out[4]

        self._n_labels = len(self._label_counts)

        if self.labels_2_show is None:
            self.labels_2_show = self._label_counts.index.values

    def _plot(self, **kwargs):

        for lNo, label in enumerate(self._label_counts.index):
            
            good_idx = self._labels == label
            n_labs = sum(good_idx)
            # if verbose:
            #     print(f"There are {n_labs} samples with label = {label}")

            if label in self.labels_2_show:

                color = self.label_cmap[self._l2i_map[label]]

                if self.show_legend:
                    title = self.long_labels[self._l2i_map[label]]
                    if "Multipotent" in title:
                        title = " ".join(title.split(" Multipotent "))
                else:
                    title = None

                zorder = None

            else:
                color = self.bkgd_label_kwds['color']
                title = None
                zorder = -1

            self.axis.scatter(*self.Y[good_idx, :2].T,
                              color=color,
                              s=self.sct_s[good_idx],
                              alpha=self.sct_a,
                              label=title,
                              zorder=zorder,
                              **self.sct_kwds)

        if self.show_legend:
            self.axis.legend(**self.legend_kwds)

        return self.axis













