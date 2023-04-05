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
	step = 1

	# Plot total strain
	eps_tot = pd.read_excel(os.path.join(output_folder, "avg", "eps_tot.xlsx"), index_col=0)
	axis[0,0].plot(eps_tot["Time"][::step]/hour, 100*(eps_tot["22"][::step]), ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[0,0].plot(eps_tot["Time"][::step]/hour, 100*(eps_tot["00"][::step]), ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[0,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,0].set_ylabel("Total strain [%]", size=12, fontname="serif")
	axis[0,0].grid(True)
	axis[0,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot elastic strain
	eps_e = pd.read_excel(os.path.join(output_folder, "avg", "eps_e.xlsx"), index_col=0)
	axis[0,1].plot(eps_e["Time"][::step]/hour, 100*(eps_e["22"][::step]), ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[0,1].plot(eps_e["Time"][::step]/hour, 100*(eps_e["00"][::step]), ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[0,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,1].set_ylabel("Elastic strain [%]", size=12, fontname="serif")
	axis[0,1].grid(True)
	axis[0,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot viscoelastic strain
	eps_ve = pd.read_excel(os.path.join(output_folder, "avg", "eps_ve.xlsx"), index_col=0)
	axis[0,2].plot(eps_ve["Time"][::step]/hour, 100*(eps_ve["22"][::step]), ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[0,2].plot(eps_ve["Time"][::step]/hour, 100*(eps_ve["00"][::step]), ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[0,2].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[0,2].set_ylabel("Visco-elastic strain [%]", size=12, fontname="serif")
	axis[0,2].grid(True)
	axis[0,2].legend(loc=0, shadow=True, fancybox=True, ncol=1)


	# Plot dislocation creep strain
	eps_d = pd.read_excel(os.path.join(output_folder, "avg", "eps_d.xlsx"), index_col=0)
	axis[1,0].plot(eps_d["Time"][::step]/hour, 100*(eps_d["22"][::step]), ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[1,0].plot(eps_d["Time"][::step]/hour, 100*(eps_d["00"][::step]), ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[1,0].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,0].set_ylabel("Dislocation creep strain [%]", size=12, fontname="serif")
	axis[1,0].grid(True)
	axis[1,0].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot pressure solution creep strain
	eps_p = pd.read_excel(os.path.join(output_folder, "avg", "eps_p.xlsx"), index_col=0)
	axis[1,1].plot(eps_p["Time"][::step]/hour, 100*(eps_p["22"][::step]), ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[1,1].plot(eps_p["Time"][::step]/hour, 100*(eps_p["00"][::step]), ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[1,1].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,1].set_ylabel("Pressure creep strain [%]", size=12, fontname="serif")
	axis[1,1].grid(True)
	axis[1,1].legend(loc=0, shadow=True, fancybox=True, ncol=1)

	# Plot stresses
	stress = pd.read_excel(os.path.join(output_folder, "avg", "stress.xlsx"), index_col=0)
	axis[1,2].plot(stress["Time"][::step]/hour, (stress["22"][::step])/MPa, ".-", color="0.15", label="Axial strain", ms=8, mfc="steelblue")
	axis[1,2].plot(stress["Time"][::step]/hour, (stress["00"][::step])/MPa, ".-", color="0.15", label="Radial strain", ms=8, mfc="lightcoral")
	axis[1,2].set_xlabel("Time [hour]", size=12, fontname="serif")
	axis[1,2].set_ylabel("Stress [MPa]", size=12, fontname="serif")
	axis[1,2].grid(True)
	axis[1,2].legend(loc=0, shadow=True, fancybox=True, ncol=1)





def main():
	fig, axis = plt.subplots(2, 4, figsize=(16, 6))
	fig.subplots_adjust(top=0.975, bottom=0.09, left=0.065, right=0.985, hspace=0.225, wspace=0.26)

	folder = os.path.join("output", "case_0")
	# folder = "case_plain"
	plot_num(axis, folder)

	# ax1.legend(loc=0, shadow=True, fancybox=True, ncol=1)
	# ax1.set_title("Creep", size=14, fontname="serif")

	apply_white_theme(fig, axis.flatten(), transparent=False)

	# fig.savefig(os.path.join("output", folder, f"fig.png"), dpi=200)

	plt.show()



if __name__ == "__main__":
	main()