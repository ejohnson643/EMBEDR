from EMBEDR.human_round import human_round
import EMBEDR.plotting_utility as putl
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np


class SweepBoxplot(object):

    BOX_PATCHES = ['boxes', 'whiskers', 'fliers', 'caps', 'medians']
    GRID_KWDS = dict()

    def __init__(self,
                 hyperparam_array,
                 values_dict,
                 kEff_dict=None,
                 fig=None,
                 fig_size=(12, 5),
                 gridspec=None,
                 gridspec_idx=(0, 0),
                 show_outer_border=False,
                 show_inner_border=True,
                 back_wpad=0.0,
                 back_hpad=0.0,
                 fig_pad=0.4,
                 grid_kwds=None,
                 params_2_highlight=None,
                 box_color=None,
                 box_fliers=None,
                 box_props=None,
                 box_hl_color=None,
                 box_hl_props=None,
                 box_widths=None,
                 box_positions=None,
                 box_notch=True,
                 box_bootstrap=100,
                 box_whiskers=(1, 99),
                 xticks=None,
                 xticklabels=None,
                 hp_2_xtick_map=None,
                 xlabel=None,
                 xlabel_size=16,
                 xlim=None,
                 yticks=None,
                 yticklabels=None,
                 ylabel=None,
                 ylabel_size=16,
                 ylim=None,
                 verbose=False,
                 **kwargs):

        self.hp_array = np.sort(hyperparam_array).squeeze()
        self.n_hp = len(self.hp_array)

        self.values = values_dict

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

        if fig is None:
            fig = plt.figure(figsize=fig_size)
        self.fig = fig

        if gridspec is None:
            gridspec = fig.add_gridspec(1, 1)
        self.gridspec = gridspec
        self.outer_gs = self.gridspec[gridspec_idx]

        self.back_wpad, self.back_hpad = back_wpad, back_hpad
        self.fig_pad = fig_pad

        ## Set up large axes
        spine_alpha = 1 if show_outer_border else 0
        self.back_axis = self.fig.add_subplot(self.outer_gs)
        self.back_axis = putl.make_border_axes(self.back_axis, 
                                               xticks=[], yticks=[],
                                               spine_alpha=spine_alpha)

        ## Set up floating gridspec
        self.spine_alpha = 1 if show_inner_border else 0
        self.inner_gs = gs.GridSpec(nrows=1, ncols=1,
                                    wspace=self.back_wpad,
                                    hspace=self.back_hpad)

        if params_2_highlight is None:
            self.hp_2_hl = []
        else:
            self.hp_2_hl = params_2_highlight
        self.hp_2_hl_idx = np.array([ii for ii, hp in enumerate(self.hp_array)
                                     if hp in self.hp_2_hl]).astype(int)

        self.grid_kwds = self.GRID_KWDS.copy()
        if grid_kwds is not None:
            self.grid_kwds.update(grid_kwds)

        self.box_color     = box_color
        self.box_fliers    = box_fliers
        self.box_props     = box_props
        self.box_hl_color  = box_hl_color
        self.box_hl_props  = box_hl_props
        self.box_widths    = box_widths
        self.box_positions = box_positions
        self.box_notch     = box_notch
        self.box_bootstrap = box_bootstrap
        self.box_whiskers  = box_whiskers

        self.xticks = [] if xticks is None else xticks
        self.xticklabels = [] if xticklabels is None else xticklabels
        self.xlabel = xlabel
        self.xlab_s = xlabel_size
        self.xlim   = xlim

        if len(self.xticks) < 1:
            if len(self.hp_2_hl) > 0:
                self.xticks = [0] + self.hp_2_hl_idx.tolist() + [self.n_hp - 1]
            else:
                if self.n_hp <= 5:
                    self.xticks = np.arange(self.n_hp)
                else:
                    self.xticks = np.unique(np.linspace(0, self.n_hp, 5))
                    self.xticks = np.clip(self.xticks, 0, self.n_hp - 1)
        self.xticks = np.asarray(self.xticks).astype(int)

        if len(self.xticklabels) == 0:
            xtlabs = np.asarray([self.hp_2_xtick_map[self.hp_array[idx]]
                                 for idx in self.xticks])
            self.xticklabels = human_round(xtlabs).squeeze().astype(int)

        if self.xlabel is None:
            self.xlabel = r"$k_{\mathrm{Eff}}$"

        if self.xlim is None:
            self.xlim = [-1, self.n_hp]

        self.yticks = [] if yticks is None else yticks
        self.yticklabels = [] if yticklabels is None else yticklabels
        self.ylabel = ylabel
        self.ylab_s = ylabel_size
        self.ylim   = ylim
        
        self.verbose = verbose
        return

    def update_tight_bounds(self):
        putl.update_tight_bounds(self.fig,
                                 self.inner_gs,
                                 self.outer_gs,
                                 w_pad=self.back_wpad,
                                 h_pad=self.back_hpad,
                                 fig_pad=self.fig_pad)
        return

    def plot(self, **kwargs):

        self.axis = self.fig.add_subplot(self.inner_gs[0])
        self.axis = putl.make_border_axes(self.axis,
                                          spine_alpha=self.spine_alpha)

        self.axis = self._plot(**kwargs)

        self.axis.grid(which='major', axis='x', alpha=0)

        self.axis.set_xticks(self.xticks)
        self.axis.set_xticklabels(self.xticklabels)
        self.axis.set_xlabel(self.xlabel, fontsize=self.xlab_s, labelpad=0)
        self.axis.set_xlim(self.xlim)

        if len(self.yticks) > 0:
            self.axis.set_yticks(self.yticks)
            self.axis.set_yticklabels(self.yticklabels)

        if self.ylim is not None:
            self.axis.set_ylim(self.ylim)

        self.axis.tick_params(pad=-3)

        self.update_tight_bounds()

        return self.axis

    def _plot(self, **kwargs):

        return self.axis


class SweepBoxplot_pValues(SweepBoxplot):

    def __init__(self, *args, **kwargs):

        try:
            self.pVal_cmap = kwargs.pop('pVal_cmap')
        except KeyError:
            self.pVal_cmap = putl.CategoricalFadingCMap()

        try:
            self.cax_ticklabels = kwargs.pop('cax_ticklabels')
        except KeyError:
            self.cax_ticklabels = [f"{10.**(-cp):.1e}"
                                   for cp in self.pVal_cmap.change_points]

        try:
            self.clabel_size = kwargs.pop('clabel_size')
        except KeyError:
            self.clabel_size = 16

        try:
            self.cax_width_frac = kwargs.pop('cax_width_frac')
        except KeyError:
            self.cax_width_frac = 1.3

        try:
            self.cax_w2h_ratio = kwargs.pop('cax_w2h_ratio')
        except KeyError:
            self.cax_w2h_ratio = 0.1

        super(SweepBoxplot_pValues, self).__init__(*args, **kwargs)

        if self.yticks == []:
            self.yticks = -np.sort(self.pVal_cmap.change_points)

        if self.ylim is None:
            self.ylim = [-self.pVal_cmap.change_points.max(),
                         -self.pVal_cmap.change_points.min()]

        if self.box_color is None:
            self.box_color = 'gray'

        if self.box_fliers is None:
            self.box_fliers = {'marker': ".",
                               'markeredgecolor': self.box_color,
                               'markersize': 2,
                               'alpha': 0.5}

        if self.box_props is None:
            self.box_props = {'alpha': 0.5,
                              'color': self.box_color,
                              'fill': True}

        if self.box_hl_color is None:
            self.box_hl_color = 'gray'

        if self.box_hl_props is None:
            self.box_hl_props = self.box_props.copy()
            self.box_hl_props.update({"alpha": 0.9,
                                      "color": self.box_hl_color})

        if self.box_widths is None:
            self.box_widths = 0.8
        try:
            _ = [el for el in self.box_widths]
        except TypeError as err:
            self.box_widths = np.ones((self.n_hp)) * self.box_widths

        if self.box_positions is None:
            self.box_positions = np.arange(self.n_hp)

        return

    def plot(self, **kwargs):

        self.axis = super().plot(**kwargs)

        self._cAx = self._add_colorbar()

        self.update_tight_bounds()

        return self.axis

    def _plot(self, **kwargs):

        hl_boxes = {}
        for hpNo, hpVal in enumerate(self.hp_array):

            box_wid = self.box_widths[hpNo]
            box_pos = self.box_positions[hpNo]

            if hpVal in self.hp_2_hl:
                box_pps = self.box_hl_props.copy()
                box_col = self.box_hl_color
            else:
                box_pps = self.box_props.copy()
                box_col = self.box_color

            vals = np.log10(self.values[hpVal])
            box = self.axis.boxplot(vals,
                                    widths=box_wid,
                                    positions=[box_pos],
                                    notch=self.box_notch,
                                    bootstrap=self.box_bootstrap,
                                    patch_artist=True,
                                    whis=self.box_whiskers,
                                    boxprops=box_pps,
                                    flierprops=self.box_fliers)

            for item in self.BOX_PATCHES:
                plt.setp(box[item], color=box_col)

            if hpVal in self.hp_2_hl:
                hl_boxes[hpVal] = box['boxes'][0]

        self._highlighted_boxes = hl_boxes

        return self.axis

    def _add_colorbar(self, **kwargs):

        ## Get the transfiguration objects
        inv_ax_trans = self.axis.transAxes.inverted()
        fig_trans    = self.fig.transFigure

        ## Convert pValue bounds from data to display
        min_pVal = np.min([np.log10(self.values[hp].min())
                           for hp in self.hp_array])
        min_pVal = np.min([min_pVal, -self.pVal_cmap.change_points.max()])
        max_pVal = np.max([np.log10(self.values[hp].max())
                           for hp in self.hp_array])
        max_pVal = np.min([max_pVal, -self.pVal_cmap.change_points.min()])

        min_pVal_crds = self.axis.transData.transform([self.xlim[0], min_pVal])
        max_pVal_crds = self.axis.transData.transform([self.xlim[0], max_pVal])

        if self.verbose:
            print(f"min_pVal_crds: {min_pVal_crds}")
            print(f"max_pVal_crds: {max_pVal_crds}")

        ## Convert from display to figure coordinates
        cFigX0, cFigY0 = fig_trans.inverted().transform(min_pVal_crds)
        cFigX1, cFigY1 = fig_trans.inverted().transform(max_pVal_crds)

        if self.verbose:
            print(f"cFig0: {cFigX0:.4f}, {cFigY0:.4f}")
            print(f"cFig1: {cFigX1:.4f}, {cFigY1:.4f}")

        ## Get the height and width of the colorbar in Figure coordinates.
        cFig_height = np.abs(cFigY1 - cFigY0)
        cFig_width  = self.cax_w2h_ratio * cFig_height

        if self.verbose:
            print(f"The color bar will be {cFig_width:.4f} x"
                  f" {cFig_height:.4f}")

        ## Get the colorbar corners
        cAxX0, cAxY0 = cFigX0 - self.cax_width_frac * cFig_width, cFigY0
        cAxX1, cAxY1 = cAxX0 + cFig_width, cFigY0 + cFig_height

        ## Convert from Figure back into Axes
        [cAxX0,
         cAxY0] = inv_ax_trans.transform(fig_trans.transform([cAxX0, cAxY0]))
        [cAxX1,
         cAxY1] = inv_ax_trans.transform(fig_trans.transform([cAxX1, cAxY1]))

        if self.verbose:
            print(f"cAx0: {cAxX0:.4f}, {cAxY0:.4f}")
            print(f"cAx1: {cAxX1:.4f}, {cAxY1:.4f}")

        ## Get the colorbar height and width in axes coordinates
        cAx_height = np.abs(cAxY1 - cAxY0)
        cAx_width  = np.abs(cAxX1 - cAxX0)

        if self.verbose:
            print(f"The color bar will be {cAx_width:.4f} x {cAx_height:.4f}")

        ## Make an inset axis.
        caxIns = self.axis.inset_axes([cAxX0, cAxY0, cAx_width, cAx_height])
        caxIns = putl.make_border_axes(caxIns, spine_alpha=0)

        ## Create a dummy mappable then the colorbar.
        hax = self.axis.scatter([], [], c=[], s=[], cmap=self.pVal_cmap.cmap,
                                norm=self.pVal_cmap.cnorm)
        cAx = self.fig.colorbar(hax, cax=caxIns, ticks=[],
                                boundaries=self.pVal_cmap.cnorm.boundaries)

        cAx.set_ticks(self.pVal_cmap.change_points)
        cAx.set_ticklabels(self.cax_ticklabels)
        cAx.ax.tick_params(length=0)

        cAx.ax.invert_yaxis()
        cAx.ax.yaxis.set_ticks_position('left')
        cAx.ax.set_ylabel(r"EMBEDR $p$-Value",
                          fontsize=self.clabel_size,
                          labelpad=2)
        cAx.ax.yaxis.set_label_position('left')

        return cAx


class SweepBoxplot_EES(SweepBoxplot):

    def __init__(self, *args, **kwargs):

        super(SweepBoxplot_EES, self).__init__(*args, **kwargs)

        if self.ylabel is None:
            self.ylabel = r"Cell-Wise $D_{KL}$ (log scale)"

        if self.box_color is None:
            self.box_color = {'data': 'C0',
                              'null': 'C1'}

        if self.box_fliers is None:
            self.box_fliers = {'marker': ".",
                               'markersize': 2,
                               'alpha': 0.5}

        if self.box_props is None:
            self.box_props = {'alpha': 0.5,
                              'fill': True}

        if self.box_hl_color is None:
            self.box_hl_color = {'data': 'C0',
                                 'null': 'C1'}

        if self.box_hl_props is None:
            self.box_hl_props = self.box_props.copy()
            self.box_hl_props.update({"alpha": 0.9})

        if self.box_widths is None:
            self.box_widths = 0.8
        try:
            _ = [el for el in self.box_widths]
        except TypeError as err:
            self.box_widths = np.ones((self.n_hp)) * self.box_widths

        if self.box_positions is None:
            self.box_positions = np.arange(self.n_hp)

    def plot(self, **kwargs):

        self.axis = super().plot(**kwargs)

        self.update_tight_bounds()

        return self.axis

    def _plot(self, **kwargs):

        min_pVal, max_pVal = np.inf, -np.inf
        hl_boxes = {}
        legend_boxes = {}
        conditions = ['data', 'null']
        for condition in conditions:
            hl_boxes[condition] = {}

            for hpNo, hpVal in enumerate(self.hp_array):

                box_wid = self.box_widths[hpNo] / 2
                if condition == 'data':
                    box_pos = self.box_positions[hpNo] - box_wid / 2
                else:
                    box_pos = self.box_positions[hpNo] + box_wid / 2

                if hpVal in self.hp_2_hl:
                    box_col = self.box_hl_color[condition]
                    box_pps = self.box_hl_props.copy()
                else:
                    box_pps = self.box_props.copy()
                    box_col = self.box_color[condition]
                box_pps['color'] = box_col

                vals = np.log10(self.values[condition][hpVal]).ravel()

                if vals.min() < min_pVal:
                    min_pVal = vals.min()
                if vals.max() > max_pVal:
                    max_pVal = vals.max()

                box = self.axis.boxplot(vals,
                                        widths=box_wid,
                                        positions=[box_pos],
                                        notch=self.box_notch,
                                        bootstrap=self.box_bootstrap,
                                        patch_artist=True,
                                        whis=self.box_whiskers,
                                        boxprops=box_pps,
                                        flierprops=self.box_fliers)

                for item in self.BOX_PATCHES:
                    plt.setp(box[item], color=box_col)

                if hpVal in self.hp_2_hl:
                    hl_boxes[condition][hpVal] = box['boxes'][0]

            box_pps = self.box_props.copy()
            box_pps['color'] = self.box_color[condition]
            dummy_box = self.axis.boxplot([], widths=0.,
                                          notch=self.box_notch,
                                          bootstrap=self.box_bootstrap,
                                          patch_artist=True,
                                          whis=self.box_whiskers,
                                          boxprops=box_pps,
                                         flierprops=self.box_fliers)

            for item in self.BOX_PATCHES:
                plt.setp(dummy_box[item], color=self.box_color[condition])

            legend_boxes[condition] = dummy_box['boxes'][0]

        self.axis.legend(list(legend_boxes.values()),
                         [cond.title() for cond in conditions])

        self.yticks = np.linspace(min_pVal, max_pVal, 7)
        self.yticklabels = [f"{10**yt:.2f}" for yt in self.yticks]
        self.axis.set_yticks(self.yticks)
        self.axis.set_yticklabels(self.yticklabels)
        self.axis.set_ylabel(self.ylabel, fontsize=self.ylab_s)

        self._highlighted_boxes = hl_boxes

        self.update_tight_bounds()

        return self.axis








# def sweep_boxplots(hp_array,
#                    values_2_sweep,
#                    sweep_type,
#                    kEff_array,
#                    fig=None,
#                    fig_size=(12, 5),
#                    gridspec=None,
#                    show_borders=False,
#                    back_wpad=0.0,
#                    back_hpad=0.0,
#                    fig_pad=0.4,
#                    categ_cmap=None,
#                    values_2_highlight=[],
#                    box_color='grey',
#                    box_fliers=None,
#                    box_props=None,
#                    box_hl_color='grey',
#                    box_hl_props=None,
#                    box_widths=None,
#                    box_positions=None,
#                    box_notch=True,
#                    box_bootstrap=100,
#                    box_whiskers=(1, 99),
#                    xticks=None,
#                    xlabel=None,
#                    xlabel_size=16,
#                    xlim=None,
#                    ylim=None,
#                    cax_ticklabels=None,
#                    cax_width_frac=1.3,
#                    cax_w2h_ratio=0.1,
#                    verbose=False):

#     if fig is None:
#         fig = plt.figure(figsize=fig_size)

#     if gridspec is None:
#         gridspec = fig.add_gridspec(1, 1)

#     ## Set up large axes
#     spine_alpha = 1 if show_borders else 0
#     back_axis = fig.add_subplot(gridspec[0])
#     back_axis = putl.make_border_axes(back_axis, xticks=[], yticks=[],
#                                       spine_alpha=spine_alpha)

#     ## Set up floating gridspec
#     back_gs = gs.GridSpec(nrows=1, ncols=1,
#                           wspace=back_wpad, hspace=back_hpad)

#     def tight_bounds_updater():
#         putl.update_tight_bounds(fig, back_gs, gridspec[0], w_pad=back_wpad,
#                                  h_pad=back_hpad, fig_pad=fig_pad)

#     if sweep_type.lower() == 'pvalues':
#         axis = sweep_boxplots_pvalues(hp_array,
#                                       values_2_sweep,
#                                       fig, gridspec,
#                                       back_axis, back_gs,
#                                       tight_bounds_updater,
#                                       pVal_cmap=categ_cmap,
#                                       values_2_highlight=values_2_highlight,
#                                       box_color=box_color,
#                                       box_fliers=box_fliers,
#                                       box_props=box_props,
#                                       box_hl_color=box_hl_color,
#                                       box_hl_props=box_hl_props,
#                                       box_widths=box_widths,
#                                       box_positions=box_positions,
#                                       box_notch=box_notch,
#                                       box_bootstrap=box_bootstrap,
#                                       box_whiskers=box_whiskers,
#                                       cax_ticklabels=cax_ticklabels,
#                                       cax_width_frac=cax_width_frac,
#                                       cax_w2h_ratio=cax_w2h_ratio,
#                                       xlim=xlim,
#                                       ylim=ylim,
#                                       label_size=xlabel_size,
#                                       verbose=verbose)

#     elif sweep_type.lower() == 'ees':
#         axis = sweep_boxplots_EES(hp_array,
#                                   values_2_sweep,
#                                   fig, gridspec,
#                                   back_axis, back_gs,
#                                   tight_bounds_updater,
#                                   pVal_cmap=categ_cmap,
#                                   values_2_highlight=values_2_highlight,
#                                   box_color=box_color,
#                                   box_fliers=box_fliers,
#                                   box_props=box_props,
#                                   box_hl_color=box_hl_color,
#                                   box_hl_props=box_hl_props,
#                                   box_widths=box_widths,
#                                   box_positions=box_positions,
#                                   box_notch=box_notch,
#                                   box_bootstrap=box_bootstrap,
#                                   box_whiskers=box_whiskers,
#                                   cax_ticklabels=cax_ticklabels,
#                                   cax_width_frac=cax_width_frac,
#                                   cax_w2h_ratio=cax_w2h_ratio,
#                                   xlim=xlim,
#                                   ylim=ylim,
#                                   label_size=xlabel_size,
#                                   verbose=verbose)

#     if xticks is None:
#         if values_2_highlight:
#             xticks = [0] + hl_idx + [hpNo]
#         else:
#             if len(hp_array) <= 5:
#                 xticks = np.arange(len(hp_array))
#             else:
#                 xticks = np.unique(np.linspace(0, len(hp_array), 5))
#                 xticks = np.clip(xticks, 0, len(hp_array) - 1)
#     xticks = np.asarray(xticks).astype(int)

#     axis.set_xticks(xticks)

#     xticklabels = [f"{int(kEff_array[hp_array[idx]])}" for idx in xticks]
#     xticklabels = human_round(np.asarray(xticklabels).squeeze()).astype(int)
#     axis.grid(which='major', axis='x', alpha=0)
#     axis.set_xticklabels(xticklabels)

#     if xlabel is None:
#         xlabel = r"$k_{\mathrm{Eff}}$"
#     axis.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=0)

#     tight_bounds_updater()

#     return axis


# def sweep_boxplots_pvalues(hp_array,
#                            pValues,
#                            fig,
#                            gridspec,
#                            back_axis,
#                            back_gs,
#                            tight_bounds_updater,
#                            pVal_cmap=None, 
#                            values_2_highlight=[],
#                            box_color='grey',
#                            box_fliers=None,
#                            box_props=None,
#                            box_hl_color='grey',
#                            box_hl_props=None,
#                            box_widths=None,
#                            box_positions=None,
#                            box_notch=True,
#                            box_bootstrap=100,
#                            box_whiskers=(1, 99),
#                            cax_ticklabels=None,
#                            cax_width_frac=1.3,
#                            cax_w2h_ratio=0.1,
#                            xlim=None,
#                            ylim=None,
#                            label_size=16,
#                            verbose=False):

#     if pVal_cmap is None:
#         pVal_cmap = putl.CategoricalFadingCMap()

#     axis = putl.make_border_axes(fig.add_subplot(back_gs[0]),
#                                  yticklabels=[],
#                                  yticks=-np.sort(pVal_cmap.change_points),
#                                  spine_alpha=1)

#     if box_fliers is None:
#         box_fliers = {'marker': ".",
#                       'markeredgecolor': box_color,
#                       'markersize': 2,
#                       'alpha': 0.5}

#     if box_props is None:
#         box_props = {'alpha': 0.5,
#                      'color': box_color,
#                      'fill': True}

#     if box_hl_props is None:
#         box_hl_props = box_props.copy()
#         box_hl_props.update({"alpha": 0.9, "color": box_hl_color})

#     if values_2_highlight is None:
#         values_2_highlight = []

#     hl_boxes = {}
#     hl_idx = []
#     for hpNo, hpVal in enumerate(hp_array):

#         if box_widths is not None:
#             try:
#                 box_wid = box_widths[hpNo]
#             except TypeError as err:
#                 box_wid = box_widths
#         else:
#             box_wid = 0.8

#         if box_positions is not None:
#             try:
#                 box_pos = [box_positions[hpNo]]
#             except TypeError as err:
#                 box_pos = [box_positions]
#         else:
#             box_pos = [hpNo]

#         if hpVal in values_2_highlight:
#             box_pps = box_hl_props.copy()
#             box_col = box_hl_color
#             hl_idx.append(hpNo)
#         else:
#             box_pps = box_props.copy()
#             box_col = box_color

#         box = axis.boxplot(np.log10(pValues[hpVal]),
#                            widths=box_wid,
#                            positions=box_pos,
#                            notch=box_notch,
#                            bootstrap=box_bootstrap,
#                            patch_artist=True,
#                            whis=box_whiskers,
#                            boxprops=box_pps,
#                            flierprops=box_fliers)

#         for item in BOX_PATCHES:
#             plt.setp(box[item], color=box_col)

#         if hpVal in values_2_highlight:
#             hl_boxes[hpVal] = box['boxes'][0]

#     if xlim is None:
#         xlim = [-1, len(hp_array)]
#     axis.set_xlim(*xlim)

#     if ylim is None:
#         ylim = [-pVal_cmap.change_points.max(), -pVal_cmap.change_points.min()]
#     axis.set_ylim(*ylim)

#     axis.tick_params(pad=-3)

#     tight_bounds_updater()

#     ## Colorbar parameters
#     if cax_ticklabels is None:
#         cax_ticklabels = [f"{10.**(-cp):.1e}"
#                           for cp in pVal_cmap.change_points]

#     inv_ax_trans = axis.transAxes.inverted()
#     fig_trans    = fig.transFigure

#     ## Convert pValue bounds from data to display
#     min_pVal = np.min([np.log10(pValues[hp].min()) for hp in hp_array])
#     min_pVal = np.min([min_pVal, -pVal_cmap.change_points.max()])
#     max_pVal = np.max([np.log10(pValues[hp].max()) for hp in hp_array])
#     max_pVal = np.min([max_pVal, -pVal_cmap.change_points.min()])
#     min_pVal_crds = axis.transData.transform([xlim[0], min_pVal])
#     max_pVal_crds = axis.transData.transform([xlim[0], max_pVal])

#     if verbose:
#         print(f"min_pVal_crds: {min_pVal_crds}")
#         print(f"max_pVal_crds: {max_pVal_crds}")

#     ## Convert from display to figure coordinates
#     cFigX0, cFigY0 = fig_trans.inverted().transform(min_pVal_crds)
#     cFigX1, cFigY1 = fig_trans.inverted().transform(max_pVal_crds)

#     if verbose:
#         print(f"cFig0: {cFigX0:.4f}, {cFigY0:.4f}")
#         print(f"cFig1: {cFigX1:.4f}, {cFigY1:.4f}")

#     ## Get the height and width of the colorbar in Figure coordinates.
#     cFig_height = np.abs(cFigY1 - cFigY0)
#     cFig_width  = cax_w2h_ratio * cFig_height

#     if verbose:
#         print(f"The color bar will be {cFig_width:.4f} x {cFig_height:.4f}")

#     ## Get the colorbar corners
#     cAxX0, cAxY0 = cFigX0 - cax_width_frac * cFig_width, cFigY0
#     cAxX1, cAxY1 = cAxX0 + cFig_width, cFigY0 + cFig_height

#     ## Convert from Figure back into Axes
#     [cAxX0,
#      cAxY0] = inv_ax_trans.transform(fig_trans.transform([cAxX0, cAxY0]))
#     [cAxX1,
#      cAxY1] = inv_ax_trans.transform(fig_trans.transform([cAxX1, cAxY1]))

#     if verbose:
#         print(f"cAx0: {cAxX0:.4f}, {cAxY0:.4f}")
#         print(f"cAx1: {cAxX1:.4f}, {cAxY1:.4f}")

#     ## Get the colorbar height and width in axes coordinates
#     cAx_height = np.abs(cAxY1 - cAxY0)
#     cAx_width  = np.abs(cAxX1 - cAxX0)

#     if verbose:
#         print(f"The color bar will be {cAx_width:.4f} x {cAx_height:.4f}")

#     ## Make an inset axis.
#     caxIns = axis.inset_axes([cAxX0, cAxY0, cAx_width, cAx_height])
#     caxIns = putl.make_border_axes(caxIns, spine_alpha=0)

#     ## Create a dummy mappable then the colorbar.
#     hax = plt.scatter([], [], c=[], s=[], cmap=pVal_cmap.cmap,
#                       norm=pVal_cmap.cnorm)
#     cAx = fig.colorbar(hax, cax=caxIns, ticks=[],
#                        boundaries=pVal_cmap.cnorm.boundaries)

#     cAx.set_ticks(pVal_cmap.change_points)
#     cAx.set_ticklabels(cax_ticklabels)
#     cAx.ax.tick_params(length=0)

#     cAx.ax.invert_yaxis()
#     cAx.ax.yaxis.set_ticks_position('left')
#     cAx.ax.set_ylabel(r"EMBEDR $p$-Value",
#                       fontsize=label_size,
#                       labelpad=2)
#     cAx.ax.yaxis.set_label_position('left')

#     return axis
