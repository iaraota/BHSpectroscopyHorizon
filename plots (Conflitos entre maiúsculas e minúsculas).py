import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# import getdist
# import getdist.plots

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

file = "/home/iaraota/Doutorado/Julia/mcmc_qnm/chains/z-1_M-500/qnm_data-220_model-220.txt"

posterior = np.genfromtxt(file, usecols = (2,3,4,5))
weight = np.genfromtxt(file, usecols = (0))
# plt.hist(posterior[:,1], color = "red", alpha = 0.4, bins = 100)
# plt.hist(posterior[:,1], weights = weight, bins = 500)
corner.corner(posterior, weights = weight)#, plot_contours = True,range=[(-1.4, -0.6), (3.4,6.0), (-1.2,0.3)], smooth = True)
# df = pd.DataFrame(posterior)
# df = df.sample(weights = weight)
# 
# fig = sns.pairplot(df, corner=True, diag_kind="kde", kind="hist", plot_kws=dict(rasterized = True))

# sns.distplot(posterior[:,0], hist_kws={'weights': weight}, kde=False)
# plt.plot(posterior[:,0]*weight)
plt.show()

# samples = getdist.loadMCSamples(file)
# print(samples.samples[:,])
# g = getdist.plots.getSubplotPlotter()
# g.settings.axes_fontsize=SMALL_SIZE
# g.settings.axes_labelsize=SMALL_SIZE
# g.settings.linewidth = 3
# g.settings.linewidth_contour = 3
# g.settings.num_plot_contours = 2
# theta_true = [0.415, 2.3, 0.3085, 34] 
# pars = []
# markers = {}
# for i in range(len(theta_true)):
#     pars.append('p'+str(i))
#     markers['p'+str(i)] = theta_true[i]

# g.triangle_plot(samples.samples, pars,
#     markers=markers)
# g.export('posterior2.pdf')
