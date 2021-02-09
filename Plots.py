import numpy as np
# pair_plot method libs:
from scipy.stats import gaussian_kde
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
class Plots:

    def __init__(self):
        self._plot_parameters()

    def pair_plot(self,
        data:dict,
        levels = [90],
        style="both",
        clabel=False,
        ):
        """Create a posterior distribuition plot for two parameters.

        Parameters
        ----------
        x : list
            first parameter sample.
        y : list
            secondi parameter sample.
        x_true : float
            first parameter true value.
        y_true : float
            second parameter true value.
        plot_color : str
            plot color.
        xlabel : str
            x axis label.
        ylabel : str
            y axis label.
        labels : list
            values labels.
        levels : list, optional
            Confidential intervals, by default [90]
        style : str, optional
            Choose to plot true values style {'both', 'point', 'lines'}, by default "both"
        clabel : bool, optional
            Show interval labels, by default False
        """
        # start plot figure
        fig, axScatter = plt.subplots(figsize=(10, 10))
        
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 1.5, pad=0., sharex=axScatter)
        axHisty = divider.append_axes("right", 1.5, pad=0., sharey=axScatter)

        # make some labels invisible
        # axHistx.axis('off')
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHistx.set_yticks([])


        axHisty.set_xticks([])
        axHisty.yaxis.set_tick_params(labelleft=False)
        
        # transform data to dataframe
        data = pd.DataFrame(data)

        # plot true values
        try:
            if style=='both':
                axScatter.scatter(data['true'].x, data['true'].y, marker = '+', color = 'k')
                axScatter.axvline(x = data['true'].x, lw=2, ls=':', color = 'k', alpha=.5)
                axScatter.axhline(y = data['true'].y, lw=2, ls=':', color = 'k', alpha=.5)
            elif style=='point':
                axScatter.scatter(data['true'].x, data['true'].y, marker = '+', color = 'k')
            elif style=='lines':
                axScatter.axvline(data['true'].x, lw=2, ls=':', color = 'k', alpha=.5)
                axScatter.axhline(data['true'].y, lw=2, ls=':', color = 'k', alpha=.5)
    
            # delete true key from data
            data.pop('true')
        except: pass
        
        try:
            axScatter.set_xlabel(data.loc['x']['label'])
            axScatter.set_ylabel(data.loc['y']['label'])
            data.pop('label')

        except: pass
        x_min = min(np.concatenate([x for x in data.loc['x']]))
        x_max = max(np.concatenate([x for x in data.loc['x']]))
        y_min = min(np.concatenate([y for y in data.loc['y']]))
        y_max = max(np.concatenate([y for y in data.loc['y']]))
        for key in data:
            
            x,y = data[key].x, data[key].y
            
            try:
                color = data[key].color
            except:
                color = None

            # compute density
            k = gaussian_kde(np.vstack([x, y]))
            
            # create grid
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            # if key=='2 modes, overtone':
            #     xi, yi = np.mgrid[x.min():1:x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]

            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            
            #set zi to 0-1 scale
            zi = (zi-zi.min())/(zi.max() - zi.min())
            zi =zi.reshape(xi.shape)


            # the scatter plot:
            # axScatter.scatter(x, y, alpha = 0.05, color = color, marker='p')
                
            #set up plot
            origin = 'lower'
            lvls = []
            for level in levels:
                lvls.append(1 - level/100)

            CS = axScatter.contour(xi, yi, zi,levels = lvls,
                    colors=(color,),
                    linewidths=(2,),
                    origin=origin)

            # fill inside
            lvls.append(1)

            axScatter.contourf(xi, yi, zi, levels = lvls,
                colors=(color,),
                alpha = 0.3,
                origin=origin,
                )
    
            if clabel == True:
                axScatter.clabel(CS, fmt=levels, colors = color)
                axScatter.ticklabel_format(axis='y', style='sci', scilimits=(0,0))



            # axHistx.hist(x, bins = 100, density = True, alpha = 0.3)
            xx = np.linspace(0, 1,500)

            axHistx.fill_between(xx,0,gaussian_kde(x)(xx)/x_max, color = color, alpha = 0.3)
            axHistx.plot(xx,gaussian_kde(x)(xx)/x_max, color = color, linewidth = 2, label=key)
            axHistx.axvline(x = np.percentile(x, 5), lw=2, ls='--', color = color)
            axHistx.axvline(x = np.percentile(x, 95), lw=2, ls='--', color = color)

            # first of all, the base transformation of the data points is needed
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(-90)

            # axHisty.hist(y, bins=100, orientation='horizontal', alpha = 0.2)
            yy = np.linspace(y_min, y_max,300)
            axHisty.fill(-yy,gaussian_kde(y)(yy)/y_max, color = color, alpha = 0.3, transform= rot + base)
            axHisty.plot(-yy, gaussian_kde(y)(yy)/y_max, color = color, linewidth = 2, transform= rot + base)
            axHisty.axhline(y = np.percentile(y, 5), lw=2, ls='--', color = color)
            axHisty.axhline(y = np.percentile(y, 95), lw=2, ls='--', color = color)
            
        # add this after calling the pair_plot function to remove gap between plots
        axScatter.set_xlim(min(np.concatenate([x for x in data.loc['x']])), max(np.concatenate([x for x in data.loc['x']])))
        axScatter.set_ylim(min(np.concatenate([y for y in data.loc['y']])), max(np.concatenate([y for y in data.loc['y']])))
        
        axHistx.set_ylim(bottom = 0)
        axHisty.set_xlim(left=0)

        axScatter.set_xlim(0.,1)
        # axScatter.set_ylim(0, 350)
        

        axHistx.legend(loc='upper left', bbox_to_anchor=(0,0))

        fig.tight_layout()


    def _plot_parameters(self):
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams["figure.figsize"] = [20, 8]  # plot image size

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 35

        plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)
