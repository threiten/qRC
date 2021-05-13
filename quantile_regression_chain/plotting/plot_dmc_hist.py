from .plot_base import plotBase

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import numpy as np


class plot_dmc_hist(plotBase):

    def __init__(self, df_mc, var, weightstr_mc, label, type, df_data=None, **kwargs):

        super(plot_dmc_hist, self).__init__(df_mc, var, weightstr_mc, label, type, df_data=df_data, **kwargs)

        self.ratio_lim = kwargs['ratio_lim']
        self.bins = np.linspace(kwargs['xmin'], kwargs['xmax'], kwargs['bins']+1)

        if ('norm' in kwargs and kwargs['norm'] is True) or 'lumi' in kwargs:
            self.normalize_mc()

        if 'ratio' in kwargs:
            if df_data is not None:
                self.ratio = kwargs['ratio']
            elif df_data is None:
                self.ratio = False

        if 'logy' in kwargs:
            self.logy = kwargs['logy']

    @staticmethod
    def get_annot_pos(leg_pos, figsize):

        ret = [0, 0]

        if leg_pos.x0 >= 0.7*figsize[0]:
            lr = 'right'
            ret[0] = 1
        elif leg_pos.x1 <= 0.5*figsize[0]:
            lr = 'left'
            ret[0] = 0
        if leg_pos.y0 >= 0.7*figsize[1]:
            ret[1] = -0.3
            tb = 'top'
        elif leg_pos.y1 <= 0.5*figsize[1]:
            ret[1] = 1
            tb = 'bottom'

        return ret, lr, tb

    def draw(self):

        self.set_style()

        if self.ratio:
            fig, axes = plt.subplots(2, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            top = axes[0]
            bottom = axes[1]
            fig.tight_layout()
        else:
            fig = plt.figure(figsize=(8,6))
            axes = None
            top = plt

        xc = 0.5*(self.bins[1:]+self.bins[:-1])
        binw = xc[1] - xc[0]

        self.mc_labels = ['MC_{}'.format(var).replace(self.mc_vars[0], '').replace('_', ' ') for var in self.mc_vars]

        if self.ratio:
            bottom.grid()

        self.mc_hists = []
        mc_errs = []
        for var in self.mc_vars:
            i = self.mc_vars.index(var)
            hist, _, _ = top.hist(self.mc.loc[:, [var]].values, bins=self.bins, range=(self.bins[0], self.bins[-1]), histtype='step', alpha=1, weights=self.mc_weights, label=self.mc_labels[i], color=self.colors[i], linestyle='solid', linewidth=3)
            mc_err, _ = np.histogram(self.mc.loc[:, [var]].values, density=False, bins=self.bins, range=(self.bins[0],self.bins[-1]), weights=self.mc_weights**2)
            mc_err = np.sqrt(mc_err)
            mc_errs.append(mc_err)
            self.mc_hists.append(hist)

        if self.data is not None:
            if hasattr(self, 'data_weights'):
                self.data_hist, _ = np.histogram(self.data, density=False, bins=self.bins, weights=self.data_weights)
                data_err, _ = np.histogram(self.data, density=False, bins=self.bins, weights=self.data_weights**2)
                data_err = np.sqrt(data_err)
            else:
                self.data_hist, _ = np.histogram(self.data, density=False, bins=self.bins)
                data_err = np.sqrt(self.data_hist)
            (_, caps, _) = top.errorbar(xc, self.data_hist, ls='None', yerr=data_err, xerr=np.ones_like(self.data_hist)*binw*0.5, color='black', label=r'data', marker='.', markersize=8)
            for cap in caps:
                cap.set_markeredgewidth(0)

        if axes is None:
            axes = fig.axes
            top = axes[0]

        top.xaxis.grid(True)

        if self.ratio:
            for i in range(len(self.mc_vars)):
                with np.errstate(divide='ignore', invalid='ignore'):
                    rdatamc = np.divide(self.data_hist, self.mc_hists[i])
                    rdatamc_err = np.divide(1., self.mc_hists[i]) * np.sqrt(data_err**2 + rdatamc**2 * mc_errs[i]**2)
                (_, caps, _) = bottom.errorbar(xc, rdatamc, ls='None', xerr=np.ones_like(rdatamc)*binw*0.5, yerr=rdatamc_err, color=self.colors[i], marker='.', markersize=7)
                for cap in caps:
                    cap.set_markeredgewidth(0)

            bottom.plot((self.bins[0], self.bins[-1]), (1, 1), 'k--')
            bottom.set_ylabel(r'\textit{Data / MC}',fontsize=13)
            bottom.set_ylim(self.ratio_lim)

        if hasattr(self, 'logy'):
            if self.logy:
                top.set_yscale('log')
        top.set_xlim(self.bins[0], self.bins[-1])

        if self.var in self.tex_replace_dict:
            math, var, unit = self.parse_repl(self.tex_replace_dict[self.var])
            if unit == '':
                axes[-1].set_xlabel(r'$\boldsymbol{{{0}}}$'.format(var, unit), fontsize=16)
            else:
                axes[-1].set_xlabel(r'$\boldsymbol{{{0}}}\,\,\left[\textnormal{{{1}}}\right]$'.format(var, unit), fontsize=16)
        else:
            axes[-1].set_xlabel(r'\textit{{{0}}}'.format(self.var.replace('_','\_')), fontsize=16)

        axes[0].set_ylabel(r'\textit{{{0}}}'.format('Events / {0:.4f}'.format(binw)), fontsize=13)
        # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        # fig.suptitle(self.title, y=0.99)
        if hasattr(self, 'leg_loc'):
            top.legend(loc=self.leg_loc)
        else:
            top.legend(loc='best')

        if self.ratio:
            fig.text(0.055, .97, r'\textbf{CMS} \textit{Work in Progress}', fontsize=13)
        else:
            fig.text(0.13, .89, r'\textbf{CMS} \textit{Work in Progress}', fontsize=13)
        if hasattr(self, 'cut_str'):
            if self.cut_str or isinstance(self.cut_str,str):
                fig.canvas.draw()
                figsize = fig.get_size_inches()*fig.dpi
                pos = top.get_legend().get_window_extent()
                ann_pos, lr, tb = self.get_annot_pos(pos, figsize)
                lc = {'left': 'left', 'right': 'right'}

                if isinstance(self.cut_str, str):
                    cut_str_fig = self.cut_str
                else:
                    self.get_tex_cut()
                    cut_str_fig = self.cut_str_tex

                top.annotate(r'\begin{{{0}}}{1}\end{{{0}}}'.format('flush{}'.format(lr), cut_str_fig), tuple(ann_pos), fontsize=14, xycoords=top.get_legend(), bbox={'boxstyle': 'square', 'alpha': 0, 'fc': 'w', 'pad': 0}, ha=lr, va=tb)
            # else:
            #     self.get_tex_cut()
            #     top.annotate(r'\begin{{{0}}}{1}\end{{{0}}}'.format('flush{}'.format(lr), self.cut_str_tex), tuple(ann_pos), fontsize=14, xycoords=top.get_legend(), bbox={'boxstyle': 'square', 'alpha': 0, 'fc': 'w', 'pad': 0}, ha=lr, va=tb)

        # cut_box = AnchoredText('{0}'.format(self.cut_str), loc=2, frameon=False)
        # top.add_artist(cut_box)

        self.fig = fig
