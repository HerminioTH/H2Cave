import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import numpy as np
import os

hour = 60*60

def apply_grey_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.grid(True, color='0.92')
			ax.set_axisbelow(True)
			ax.spines['bottom'].set_color('black')
			ax.spines['top'].set_color('black')
			ax.spines['right'].set_color('black')
			ax.spines['left'].set_color('black')
			ax.tick_params(axis='x', colors='black', which='both')
			ax.tick_params(axis='y', colors='black', which='both')
			ax.yaxis.label.set_color('black')
			ax.xaxis.label.set_color('black')
			ax.set_facecolor("0.85")

def plot_smp(ax, folder):
	skip = 1
	eps_tot = pd.read_excel(os.path.join(folder, "smp", f"eps_tot.xlsx"))
	time = eps_tot["Time"].values[::skip]/hour
	eps_a = -100*eps_tot["22"].values[::skip]
	eps_r = -100*eps_tot["00"].values[::skip]
	ax.plot(time, eps_a, "-", color="black", label="SMP", linewidth=2.0)
	ax.plot(time, eps_r, "-", color="black", linewidth=2.0)

def plot_fem(ax, folder):
	skip = 2
	eps_tot = pd.read_excel(os.path.join(folder, "fem", "avg", f"eps_tot.xlsx"))
	time = eps_tot["Time"].values[::skip]/hour
	eps_a = -100*eps_tot["22"].values[::skip]
	eps_r = -100*eps_tot["00"].values[::skip]
	ax.plot(time, eps_a, ".", color="steelblue", label="FEM", linewidth=2.0)
	ax.plot(time, eps_r, ".", color="steelblue", linewidth=2.0)

def main():
	fig, ax = plt.subplots(1, 1, figsize=(4, 3))
	fig.subplots_adjust(top=0.975, bottom=0.155, left=0.140, right=0.980, hspace=0.2, wspace=0.32)

	results_folder = os.path.join("output", "case_0")
	plot_smp(ax, results_folder)
	plot_fem(ax, results_folder)

	ax.set_xlabel("Time (hours)", size=12, fontname="serif")
	ax.set_ylabel("Total strain (%)", size=12, fontname="serif")
	ax.grid(True)
	ax.legend(loc=0, shadow=True, fancybox=True)

	apply_grey_theme(fig, [ax], transparent=True)
	plt.show()

if __name__ == "__main__":
	main()