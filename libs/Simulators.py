

class Simulator():
	def __init__(self, time_handler):
		self.time_handler = time_handler
		self.models = []
		self.events = []
		self.controllers = []

	def execute_models_pre(self):
		for model in self.models:
			model.execute_model_pre(self.time_handler)

	def execute_models_post(self):
		for model in self.models:
			model.execute_model_post(self.time_handler)

	def execute_iterative_procedures(self):
		for model in self.models:
			model.execute_iterative_procedure(self.time_handler)

	def update_solution_vector(self):
		for model in self.models:
			model.update_solution_vector()

	def add_model(self, model):
		self.models.append(model)

	def initialize_models(self):
		for model in self.models:
			model.initialize(self.time_handler)

	def add_controller(self, controller):
		self.controllers.append(controller)

	def reset_controllers(self):
		for controller in self.controllers:
			controller.reset()

	def execute_controllers(self):
		for controller in self.controllers:
			controller.execute()

	def check_controllers(self):
		if len(self.controllers) == 0:
			return False
		else:
			for controller in self.controllers:
				if not controller.check():
					return False
			return True

	def add_event(self, event):
		self.events.append(event)

	def initialize_events(self):
		for event in self.events:
			event.initialize()

	def execute_events(self):
		for event in self.events:
			event.execute()

	def finalize_events(self):
		for event in self.events:
			event.finalize()

	def is_final_time_reached(self):
		return self.time_handler.is_final_time_reached()

	def advance_time(self):
		self.time_handler.advance_time()

	def run(self):
		self.initialize_models()
		self.initialize_events()

		# Time marching
		while not self.is_final_time_reached():
			self.advance_time()
			self.execute_models_pre()
			self.reset_controllers()

			# Begin iterative loop
			while self.check_controllers():
				self.update_solution_vector()
				self.execute_iterative_procedures()
				self.execute_controllers()

			self.execute_models_post()
			self.execute_events()

		# Write results
		self.finalize_events()
		



def H2CaveSimulator(input_model, input_bc, grid=None):
	import os
	import time
	import shutil
	from Grid import GridHandler
	from Events import VtkSaver, AverageSaver, AverageScalerSaver, ScreenOutput, TimeLevelCounter, TimeCounter
	from Controllers import TimeController, IterationController, ErrorController
	from Time import TimeHandler
	from FiniteElements import FemHandler
	from BoundaryConditions import MechanicsBoundaryConditions
	from Simulators import Simulator
	from Models import MaxwellModel, BurgersModel, ElasticModel, ViscoelasticModel
	from Elements import DashpotElement, DislocationCreep, PressureSolutionCreep, Damage, ViscoplasticDesaiElement
	from Utils import sec, save_json

	# Define folders
	output_folder = os.path.join(*input_model["Paths"]["Output"].split("/"))
	grid_folder = os.path.join(*input_model["Paths"]["Grid"].split("/"))

	# Load grid
	if grid == None:
		grid = GridHandler("geom", grid_folder)

	# Define time handler
	time_handler = TimeHandler(input_bc["Time"])

	# Define finite element handler (function spaces, normal vectors, etc)
	fem_handler = FemHandler(grid)

	# Define boundary condition handler
	bc_handler = MechanicsBoundaryConditions(fem_handler, input_bc)

	# Build model
	inelastic_elements = input_model["Model"].copy()
	if "Spring" not in input_model["Model"]:
		raise Exception("Model must have a Spring element.")

	elif "KelvinVoigt" not in input_model["Model"]:
		if len(input_model["Model"]) == 1:
			model = ElasticModel(fem_handler, bc_handler, input_model)
		else:
			model = MaxwellModel(fem_handler, bc_handler, input_model)

	elif "KelvinVoigt" in input_model["Model"]:
		if len(input_model["Model"]) == 2:
			model = ViscoelasticModel(fem_handler, bc_handler, input_model)
		else:
			model = BurgersModel(fem_handler, bc_handler, input_model)
		inelastic_elements.remove("KelvinVoigt")
	inelastic_elements.remove("Spring")

	# Add inelastic elements to the model
	ELEMENT_DICT = {"Dashpot": DashpotElement,
					"DislocationCreep": DislocationCreep,
					"PressureSolutionCreep": PressureSolutionCreep,
					"ViscoplasticDesai": ViscoplasticDesaiElement,
					"Damage": Damage}
	for element_name in inelastic_elements:
		model.add_inelastic_element(ELEMENT_DICT[element_name](fem_handler, input_model, element_name=element_name))

	# Build simulator
	sim = Simulator(time_handler)

	# Add models
	sim.add_model(model)

	print(model.inelastic_elements)

	# Save displacement field
	sim.add_event(VtkSaver("displacement", model.u, time_handler, output_folder))

	# Save stress field
	if input_model["Elements"]["Spring"]["save_stress_vtk"] == True:
		field_name = input_model["Elements"]["Spring"]["stress_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.stress, time_handler, output_folder))
	if input_model["Elements"]["Spring"]["save_stress_avg"] == True:
		field_name = input_model["Elements"]["Spring"]["stress_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.stress, time_handler, output_folder))

	# Save total strain field
	if input_model["Elements"]["Spring"]["save_total_strain_vtk"] == True:
		field_name = input_model["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.eps_tot, time_handler, output_folder))
	if input_model["Elements"]["Spring"]["save_total_strain_avg"] == True:
		field_name = input_model["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_tot, time_handler, output_folder))
	
	# Save elastic strain field
	if input_model["Elements"]["Spring"]["save_strain_vtk"] == True:
		field_name = input_model["Elements"]["Spring"]["strain_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.eps_e, time_handler, output_folder))
	if input_model["Elements"]["Spring"]["save_strain_avg"] == True:
		field_name = input_model["Elements"]["Spring"]["strain_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_e, time_handler, output_folder))

	# Save viscoelastic strain field
	if "KelvinVoigt" in input_model["Model"]:
		if input_model["Elements"]["KelvinVoigt"]["save_strain_vtk"] == True:
			field_name = input_model["Elements"]["KelvinVoigt"]["strain_name"]
			sim.add_event(VtkSaver(field_name, model.elastic_element.eps_v, time_handler, output_folder))
		if input_model["Elements"]["KelvinVoigt"]["save_strain_avg"] == True:
			field_name = input_model["Elements"]["KelvinVoigt"]["strain_name"]
			sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_v, time_handler, output_folder))

	# Save inelastic strain fields
	for element_name in inelastic_elements:
		index = inelastic_elements.index(element_name)
		if input_model["Elements"][element_name]["save_strain_vtk"] == True:
			field_name = input_model["Elements"][element_name]["strain_name"]
			sim.add_event(VtkSaver(field_name, model.inelastic_elements[index].eps_ie, time_handler, output_folder))
		if input_model["Elements"][element_name]["save_strain_avg"] == True:
			field_name = input_model["Elements"][element_name]["strain_name"]
			sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.inelastic_elements[index].eps_ie, time_handler, output_folder))

	# If Desai viscoplastic model is included, save Fvp and alpha fields
	if "ViscoplasticDesai" in inelastic_elements:
		i_desai = inelastic_elements.index("ViscoplasticDesai")
		if input_model["Elements"]["ViscoplasticDesai"]["save_Fvp_vtk"] == True:
			vtk_Fvp_saver = VtkSaver("Fvp", model.inelastic_elements[i_desai].F_vp, time_handler, output_folder)
			sim.add_event(vtk_Fvp_saver)
		if input_model["Elements"]["ViscoplasticDesai"]["save_Fvp_avg"] == True:
			avg_Fvp_saver = AverageScalerSaver(fem_handler.dx(), "Fvp", model.inelastic_elements[i_desai].F_vp, time_handler, output_folder)
			sim.add_event(avg_Fvp_saver)
		if input_model["Elements"]["ViscoplasticDesai"]["save_alpha_vtk"] == True:
			vtk_alpha_saver = VtkSaver("alpha", model.inelastic_elements[i_desai].alpha, time_handler, output_folder)
			sim.add_event(vtk_alpha_saver)
		if input_model["Elements"]["ViscoplasticDesai"]["save_alpha_avg"] == True:
			avg_alpha_saver = AverageScalerSaver(fem_handler.dx(), "alpha", model.inelastic_elements[i_desai].alpha, time_handler, output_folder)
			sim.add_event(avg_alpha_saver)

	# If damage model is included, save damage field
	if "Damage" in inelastic_elements:
		i_damage = inelastic_elements.index("Damage")
		name_damage = input_model["Elements"]["Damage"]["damage_name"]
		vtk_damage_saver = VtkSaver(name_damage, model.inelastic_elements[i_damage].D, time_handler, output_folder)
		avg_damage_saver = AverageScalerSaver(fem_handler.dx(), name_damage, model.inelastic_elements[i_damage].D, time_handler, output_folder)
		sim.add_event(vtk_damage_saver)
		sim.add_event(avg_damage_saver)

	# Build time counters
	time_level_counter = TimeLevelCounter(time_handler)
	time_counter = TimeCounter(time_handler, "Time (h)", "hours")
	sim.add_event(time_level_counter)
	sim.add_event(time_counter)

	# Build controllers
	if type(model) == MaxwellModel or type(model) == BurgersModel:
		iteration_controller = IterationController("Iterations", max_ite=50)
		error_controller = ErrorController("Error", model, tol=1e-6)
		sim.add_controller(iteration_controller)
		sim.add_controller(error_controller)

	# Build screen monitor
	screen_monitor = ScreenOutput()
	if type(model) == ElasticModel or type(model) == ViscoelasticModel:
		screen_monitor.add_controller(time_level_counter, width=10, align="center")
		screen_monitor.add_controller(time_counter, width=20, align="center")
	else:
		screen_monitor.add_controller(time_level_counter, width=10, align="center")
		screen_monitor.add_controller(time_counter, width=20, align="center")
		screen_monitor.add_controller(iteration_controller, width=10, align="center")
		screen_monitor.add_controller(error_controller, width=30, align="center")
	sim.add_event(screen_monitor)

	# Run simlation
	start = time.time()
	sim.run()
	end = time.time()
	print("Elapsed time: %.3f"%((end-start)/sec))

	# Copy .msh mesh to output_folder
	try:
		shutil.copy(os.path.join(grid_folder, "geom.msh"), os.path.join(output_folder, "vtk"))
	except:
		print("File geom.msh could not be saved.")
	# shutil.copy(__file__, os.path.join(output_folder, "copy.py"))
	save_json(input_model, os.path.join(output_folder, "input_model.json"))
	save_json(input_bc, os.path.join(output_folder, "input_bc_fem.json"))




def SmpSimulator(input_model, input_bc):
	'''
		This is a simulator for the Simplified Model, which has nothing to do with the Finite Elements.
	'''
	from RockSampleSolutions import Elastic, Viscoelastic, DislocationCreep, PressureSolutionCreep, Damage, ViscoplasticDesai, ViscoPlasticDruckerPrager, TensorSaver
	from Utils import save_json
	import numpy as np
	import os

	# Output folder
	output_folder = input_model["Paths"]["Output"]

	# Build model
	if "Spring" not in input_model["Model"]:
		raise Exception("Model must have a Spring element.")
	else:
		model_elements = [Elastic(input_model, input_bc)]
	if "KelvinVoigt" in input_model["Model"]:
		model_elements.append(Viscoelastic(input_model, input_bc))
	if "DislocationCreep" in input_model["Model"]:
		model_elements.append(DislocationCreep(input_model, input_bc))
	if "PressureSolutionCreep" in input_model["Model"]:
		model_elements.append(PressureSolutionCreep(input_model, input_bc))
	if "Damage" in input_model["Model"]:
		model_elements.append(Damage(input_model, input_bc))
	if "ViscoplasticDesai" in input_model["Model"]:
		model_elements.append(ViscoplasticDesai(input_model, input_bc))
	if "ViscoPlasticDruckerPrager" in input_model["Model"]:
		model_elements.append(ViscoPlasticDruckerPrager(input_model, input_bc))

	# Compute total strain
	eps_tot = 0
	for element in model_elements:
		element.compute_strains()
		eps_tot += element.eps.copy()

	# Save fields
	ELEMENT_DICT = {
						Elastic : "Spring",
						Viscoelastic : "KelvinVoigt",
						DislocationCreep : "DislocationCreep",
						PressureSolutionCreep : "PressureSolutionCreep",
						Damage : "Damage",
						ViscoplasticDesai : "ViscoplasticDesai",
						ViscoPlasticDruckerPrager : "ViscoPlasticDruckerPrager"
	}
	for element in model_elements:
		element_name = ELEMENT_DICT[type(element)]
		if input_model["Elements"][element_name]["save_strain_smp"] == True:
			strain_name = input_model["Elements"][element_name]["strain_name"]
			saver_eps = TensorSaver(output_folder, strain_name)
			saver_eps.save_results(element.time_list, element.eps)

	# Save total strain
	if input_model["Elements"]["Spring"]["save_total_strain_smp"] == True:
		strain_name = input_model["Elements"]["Spring"]["total_strain_name"]
		saver_eps = TensorSaver(output_folder, strain_name)
		saver_eps.save_results(model_elements[0].time_list, eps_tot)

	# Save internal parameters of Desai viscoplastic model
	if "ViscoplasticDesai" in input_model["Model"]:
		print(input_model["Model"])
		index = input_model["Model"].index("ViscoplasticDesai")
		element = model_elements[index]
		if input_model["Elements"]["ViscoplasticDesai"]["save_Fvp_smp"] == True:
			saver_Fvp = TensorSaver(output_folder, "Fvp")
			Fvps = np.zeros(element.eps.shape)
			Fvps[:,0,0] = element.Fvp_list
			saver_Fvp.save_results(element.time_list, Fvps)
		if input_model["Elements"]["ViscoplasticDesai"]["save_alpha_smp"] == True:
			saver_alpha = TensorSaver(output_folder, "alpha")
			alphas = np.zeros(element.eps.shape)
			alphas[:,0,0] = element.alphas
			saver_alpha.save_results(element.time_list, alphas)

	# Save internal parameters of Durcker-Prager viscoplastic model
	if "ViscoPlasticDruckerPrager" in input_model["Model"]:
		print(input_model["Model"])
		index = input_model["Model"].index("ViscoPlasticDruckerPrager")
		element = model_elements[index]
		if input_model["Elements"]["ViscoPlasticDruckerPrager"]["save_Fvp_smp"] == True:
			saver_Fvp = TensorSaver(output_folder, "Fvp")
			Fvps = np.zeros(element.eps.shape)
			Fvps[:,0,0] = element.Fvp_list
			saver_Fvp.save_results(element.time_list, Fvps)
		if input_model["Elements"]["ViscoPlasticDruckerPrager"]["save_alpha_smp"] == True:
			saver_alpha = TensorSaver(output_folder, "alpha")
			alphas = np.zeros(element.eps.shape)
			alphas[:,0,0] = element.alphas
			saver_alpha.save_results(element.time_list, alphas)



	# Save settings
	save_json(input_bc, os.path.join(output_folder, "input_bc_smp.json"))
	save_json(input_model, os.path.join(output_folder, "input_model.json"))