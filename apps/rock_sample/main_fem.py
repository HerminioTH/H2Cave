import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from Simulators import H2CaveSimulator
from Utils import read_json

def main():
	# Read input_model
	input_model = read_json("input_model.json")

	# Read input_bc
	input_bc = read_json("input_bc_fem.json")

	# Change output folder name (just add "fem" folder)
	output_folder = input_model["Paths"]["Output"]
	input_model["Paths"]["Output"] = os.path.join(output_folder, "fem")

	# Build simulation and run
	H2CaveSimulator(input_model, input_bc)


if __name__ == "__main__":
	main()