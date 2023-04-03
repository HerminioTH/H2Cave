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


def plot_num(axis, output_folder):
	# Plot total strain
	eps_tot = pd.read_excel(os.path.join(output_folder, "avg", "eps_tot.xlsx"), index_col=0)
	step = 1
	axis[0,0].plot(eps_tot["Time"][::step]/hour, 100*(eps_tot["22"][::step]), ".-", color="0.15", label="Numeric", ms=8, mfc="steelblue")
	axis[0,0].plot(eps_tot["Time"][::step]/hour, 100*(eps_tot["00"][::step]), ".-", color="0.15", ms=8, mfc="steelblue")
	axis[0,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,0].set_ylabel("Total strain [%]", size=12, fontname="serif")
	axis[0,0].grid(True)
	axis[0,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot elastic strain
	eps_e = pd.read_excel(os.path.join(output_folder, "avg", "eps_e.xlsx"), index_col=0)
	step = 1
	axis[0,1].plot(eps_e["Time"][::step]/hour, 100*(eps_e["22"][::step]), ".-", color="0.15", label="Numeric", ms=8, mfc="steelblue")
	axis[0,1].plot(eps_e["Time"][::step]/hour, 100*(eps_e["00"][::step]), ".-", color="0.15", ms=8, mfc="steelblue")
	axis[0,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,1].set_ylabel("Elastic strain [%]", size=12, fontname="serif")
	axis[0,1].grid(True)
	axis[0,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot viscoplastic strain
	eps_vp = pd.read_excel(os.path.join(output_folder, "avg", "eps_ie.xlsx"), index_col=0)
	step = 1
	axis[0,2].plot(eps_vp["Time"][::step]/hour, 100*(eps_vp["22"][::step]), ".-", color="0.15", label="Numeric", ms=8, mfc="steelblue")
	axis[0,2].plot(eps_vp["Time"][::step]/hour, 100*(eps_vp["00"][::step]), ".-", color="0.15", ms=8, mfc="steelblue")
	axis[0,2].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,2].set_ylabel("Viscoplastic strain [%]", size=12, fontname="serif")
	axis[0,2].grid(True)
	axis[0,2].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot alpha
	alpha = pd.read_excel(os.path.join(output_folder, "avg", "alpha.xlsx"), index_col=0)
	step = 1
	axis[1,0].plot(alpha["Time"][::step]/hour, alpha["Scalar"][::step], ".-", color="0.15", label="Numeric", ms=8, mfc="steelblue")
	axis[1,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,0].set_ylabel(r"$\alpha$", size=12, fontname="serif")
	axis[1,0].grid(True)
	axis[1,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot Fvp
	Fvp = pd.read_excel(os.path.join(output_folder, "avg", "Fvp.xlsx"), index_col=0)
	step = 1
	axis[1,1].plot(Fvp["Time"][::step]/hour, Fvp["Scalar"][::step], ".-", color="0.15", label="Numeric", ms=8, mfc="steelblue")
	axis[1,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,1].set_ylabel(r"$F_{vp}$", size=12, fontname="serif")
	axis[1,1].grid(True)
	axis[1,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)


def plot_ana(axis, output_folder):
	data_alpha = pd.read_excel(os.path.join(output_folder, "alpha.xlsx"), index_col=0)
	alphas = data_alpha["00"]

	# Plot total strain
	eps_tot = pd.read_excel(os.path.join(output_folder, "eps_tot.xlsx"), index_col=0)
	step = 1
	axis[0,0].plot(eps_tot["Time"][1::step]/hour, 100*(eps_tot["22"][:-1:step]), ".-", color="0.15", label="Analytic", ms=8, mfc="lightcoral")
	axis[0,0].plot(eps_tot["Time"][1::step]/hour, 100*(eps_tot["00"][:-1:step]), ".-", color="0.15", ms=8, mfc="lightcoral")
	axis[0,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,0].set_ylabel("Total strain [%]", size=12, fontname="serif")
	axis[0,0].grid(True)
	axis[0,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot elastic strain
	eps_e = pd.read_excel(os.path.join(output_folder, "eps_e.xlsx"), index_col=0)
	step = 1
	axis[0,1].plot(eps_e["Time"][1::step]/hour, 100*(eps_e["22"][:-1:step]), ".-", color="0.15", label="Analytic", ms=8, mfc="lightcoral")
	axis[0,1].plot(eps_e["Time"][1::step]/hour, 100*(eps_e["00"][:-1:step]), ".-", color="0.15", ms=8, mfc="lightcoral")
	axis[0,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,1].set_ylabel("Elastic strain [%]", size=12, fontname="serif")
	axis[0,1].grid(True)
	axis[0,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot elastic strain
	eps_vp = pd.read_excel(os.path.join(output_folder, "eps_vp.xlsx"), index_col=0)
	step = 1
	axis[0,2].plot(eps_vp["Time"][1::step]/hour, 100*(eps_vp["22"][:-1:step]), ".-", color="0.15", label="Analytic", ms=8, mfc="lightcoral")
	axis[0,2].plot(eps_vp["Time"][1::step]/hour, 100*(eps_vp["00"][:-1:step]), ".-", color="0.15", ms=8, mfc="lightcoral")
	axis[0,2].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,2].set_ylabel("Viscoplastic strain [%]", size=12, fontname="serif")
	axis[0,2].grid(True)
	axis[0,2].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot alpha
	alpha = pd.read_excel(os.path.join(output_folder, "alpha.xlsx"), index_col=0)
	step = 1
	axis[1,0].plot(alpha["Time"][::step]/hour, alpha["00"][::step], ".-", color="0.15", label="Analytic", ms=8, mfc="lightcoral")
	axis[1,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,0].set_ylabel(r"$\alpha$", size=12, fontname="serif")
	axis[1,0].grid(True)
	axis[1,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot Fvp
	Fvp = pd.read_excel(os.path.join(output_folder, "Fvp.xlsx"), index_col=0)
	step = 1
	axis[1,1].plot(Fvp["Time"][::step]/hour, Fvp["00"][::step], ".-", color="0.15", label="Analytic", ms=8, mfc="lightcoral")
	axis[1,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,1].set_ylabel(r"$F_{vp}$", size=12, fontname="serif")
	axis[1,1].grid(True)
	axis[1,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)



def main():
	fig, axis = plt.subplots(2, 3, figsize=(12, 6))
	fig.subplots_adjust(top=0.975, bottom=0.125, left=0.080, right=0.985, hspace=0.2, wspace=0.2)

	plot_num(axis, os.path.join("output", "case_0", "num"))
	plot_ana(axis, os.path.join("output", "case_0", "ana"))

	apply_white_theme(fig, axis.flatten(), transparent=False)

	# fig.savefig(os.path.join("output", folder, f"fig.png"), dpi=200)

	plt.show()



if __name__ == "__main__":
	main()