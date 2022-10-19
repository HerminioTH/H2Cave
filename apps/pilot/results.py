import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
month = 30*day
kPa = 1e3
MPa = 1e6
GPa = 1e9

def apply_grey_theme(fig, axes, transparent=True):
	# fig.patch.set_facecolor("#212121ff")
	# if transparent:
	# 	fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.spines['bottom'].set_color('white')
			ax.spines['top'].set_color('white')
			ax.spines['right'].set_color('white')
			ax.spines['left'].set_color('white')
			ax.tick_params(axis='x', colors='white', which='both')
			ax.tick_params(axis='y', colors='white', which='both')
			ax.yaxis.label.set_color('white')
			ax.xaxis.label.set_color('white')
			ax.title.set_color('white')
			ax.set_facecolor("#2b2b2bff")
			ax.grid(True, color='#414141ff')

def apply_dark_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#37474fff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.spines['bottom'].set_color('white')
			ax.spines['top'].set_color('white')
			ax.spines['right'].set_color('white')
			ax.spines['left'].set_color('white')
			ax.tick_params(axis='x', colors='white', which='both')
			ax.tick_params(axis='y', colors='white', which='both')
			ax.yaxis.label.set_color('white')
			ax.xaxis.label.set_color('white')
			ax.title.set_color('white')
			ax.set_facecolor("#424242ff")
			ax.grid(True, color='#5a5a5aff')

def apply_white_theme(fig, axes, transparent=True):
	# fig.patch.set_facecolor("#212121ff")
	# if transparent:
	# 	fig.patch.set_alpha(0.0)
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

def plot_exp(ax, axins=None):
	data = np.loadtxt(os.path.join("exp", "sigmaA_12_14_15MPa_sigmaR_0MPa.csv"), delimiter=",")
	ax.plot(data[:,0], data[:,1], ".", color="black", label="Exp.", ms=10, mfc="white")
	if axins != None:
		axins.plot(data[:,0], data[:,1], ".", color="black", label="Exp.", ms=10, mfc="white")

def plot_num(ax1, file_name):
	output_folder = os.path.join("output", f"{file_name}", "numeric")
	eps_tot = pd.read_excel(os.path.join(output_folder, "eps_tot.xlsx"))
	# eps_ve = pd.read_excel(os.path.join(output_folder, "eps_v.xlsx"))
	# eps_e = pd.read_excel(os.path.join(output_folder, "eps_e.xlsx"))
	# eps_cr = pd.read_excel(os.path.join(output_folder, "eps_cr.xlsx"))
	# eps_ve["22"] += eps_e["22"]

	step = 1
	ax1.plot(eps_tot["Time"][::step]/hour, 100*abs(eps_tot["22"][::step]), ".", color="0.15", label=r"$\varepsilon_{tot}$", ms=8, mfc="steelblue")
	# ax1.plot(eps_ve["Time"][::step]/hour, 100*abs(eps_ve["22"][::step]), ".", color="0.15", label=r"$\varepsilon_{v}$", ms=8, mfc="lightcoral")
	# ax1.plot(eps_e["Time"][::step]/hour, 100*abs(eps_e["22"][::step]), ".", color="0.15", label=r"$\varepsilon_{e}$", ms=8, mfc="gold")
	# ax1.plot(eps_cr["Time"][::step]/hour, 100*abs(eps_cr["22"][::step]), ".-", color="0.15", label=r"$\varepsilon_{cr}$", ms=8, mfc="#ac84cbff")
	ax1.set_xlabel("Time [hour]", size=12, fontname="serif")
	ax1.set_ylabel("Axial Strain [%]", size=12, fontname="serif")
	ax1.grid(True)




def main():
	fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
	fig.subplots_adjust(top=0.935, bottom=0.125, left=0.120, right=0.985, hspace=0.2, wspace=0.2)

	folder = "case_1"
	plot_num(ax1, folder)

	ax1.legend(loc=2, shadow=True, fancybox=True, ncol=1)
	ax1.set_title("Viscoelastic + creep", size=14, fontname="serif")

	apply_white_theme(fig, [ax1], transparent=False)

	fig.savefig(os.path.join("output", folder, f"fig.png"), dpi=200)

	plt.show()



if __name__ == "__main__":
	main()