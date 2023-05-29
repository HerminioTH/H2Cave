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

def apply_white_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.grid(True, color='#c4c4c4ff')
			ax.set_axisbelow(True)
			ax.spines['bottom'].set_color('black')
			ax.spines['top'].set_color('black')
			ax.spines['right'].set_color('black')
			ax.spines['left'].set_color('black')
			ax.tick_params(axis='x', colors='black', which='both')
			ax.tick_params(axis='y', colors='black', which='both')
			ax.yaxis.label.set_color('black')
			ax.xaxis.label.set_color('black')
			ax.set_facecolor("#e9e9e9ff")

def plot_model(axis, folder, color, label_name):
	eps_tot = pd.read_excel(os.path.join("output", folder, "eps_tot.xlsx"))
	skip = 1
	time = eps_tot["Time"].values[::skip]/hour
	eps_a = 100*eps_tot["22"].values[::skip]
	eps_r = 100*eps_tot["00"].values[::skip]
	axis[0].plot(time, eps_a, "-", color=color, label=label_name, linewidth=2.0)
	axis[0].plot(time, eps_r, "-", color=color, linewidth=2.0)


def main():
	fig, axis = plt.subplots(1, 2, figsize=(8, 3))
	fig.subplots_adjust(top=0.985, bottom=0.160, left=0.115, right=0.940, hspace=0.2, wspace=0.275)

	# plot_exp(axis[0])
	plot_model(axis, os.path.join("case_no_damage", "avg"), "steelblue", "No damage")
	plot_model(axis, os.path.join("case_damage_0", "avg"), "lightcoral", "Damage")
	# plot_model(axis, os.path.join("case_0", "avg"), "lightcoral", "FEM")
	# plot_model(axis, "ANA", "steelblue", "ANA")
	# plot_model(ax, "", "gold", "ANA_0")



	axis[0].set_xlabel("Time (hours)", size=12, fontname="serif")
	axis[0].set_ylabel("Total strain (%)", size=12, fontname="serif")
	# ax.set_ylim(-1.5, 1.62)
	axis[0].grid(True)
	axis[0].legend(loc=3, shadow=True, fancybox=True)


	apply_white_theme(fig, axis.flatten(), transparent=True)
	plt.show()


if __name__ == "__main__":
	# main_0()
	main()