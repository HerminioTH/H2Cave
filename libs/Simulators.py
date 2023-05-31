

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
		



def H2CaveSimulator(settings):
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
	from Models import MaxwellModel_2, BurgersModel, ElasticModel, ViscoelasticModel
	from Elements import DashpotElement, DislocationCreep, PressureSolutionCreep, Damage
	from Utils import sec, save_json

	# Define folders
	output_folder = os.path.join(*settings["Paths"]["Output"].split("/"))
	grid_folder = os.path.join(*settings["Paths"]["Grid"].split("/"))

	# Load grid
	grid = GridHandler("geom", grid_folder)

	# Define time handler
	time_handler = TimeHandler(settings["Time"])

	# Define finite element handler (function spaces, normal vectors, etc)
	fem_handler = FemHandler(grid)

	# Define boundary condition handler
	bc_handler = MechanicsBoundaryConditions(fem_handler, settings)

	# Build model
	inelastic_elements = settings["Model"].copy()
	if "Spring" not in settings["Model"]:
		raise Exception("Model must have a Spring element.")

	elif "KelvinVoigt" not in settings["Model"]:
		if len(settings["Model"]) == 1:
			model = ElasticModel(fem_handler, bc_handler, settings)
		else:
			model = MaxwellModel_2(fem_handler, bc_handler, settings)

	elif "KelvinVoigt" in settings["Model"]:
		if len(settings["Model"]) == 2:
			model = ViscoelasticModel(fem_handler, bc_handler, settings)
		else:
			model = BurgersModel(fem_handler, bc_handler, settings)
		inelastic_elements.remove("KelvinVoigt")
	inelastic_elements.remove("Spring")

	# Add inelastic elements to the model
	ELEMENT_DICT = {"Dashpot": DashpotElement,
					"DislocationCreep": DislocationCreep,
					"PressureSolutionCreep": PressureSolutionCreep,
					"Damage": Damage}
	for element_name in inelastic_elements:
		model.add_inelastic_element(ELEMENT_DICT[element_name](fem_handler, settings, element_name=element_name))

	# Build simulator
	sim = Simulator(time_handler)

	# Add models
	sim.add_model(model)

	# Save displacement field
	sim.add_event(VtkSaver("displacement", model.u, time_handler, output_folder))

	# Save stress field
	if settings["Elements"]["Spring"]["save_stress_vtk"] == True:
		field_name = settings["Elements"]["Spring"]["stress_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.stress, time_handler, output_folder))
	if settings["Elements"]["Spring"]["save_stress_avg"] == True:
		field_name = settings["Elements"]["Spring"]["stress_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.stress, time_handler, output_folder))

	# Save total strain field
	if settings["Elements"]["Spring"]["save_total_strain_vtk"] == True:
		field_name = settings["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.eps_tot, time_handler, output_folder))
	if settings["Elements"]["Spring"]["save_total_strain_avg"] == True:
		field_name = settings["Elements"]["Spring"]["total_strain_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_tot, time_handler, output_folder))
	
	# Save elastic strain field
	if settings["Elements"]["Spring"]["save_strain_vtk"] == True:
		field_name = settings["Elements"]["Spring"]["strain_name"]
		sim.add_event(VtkSaver(field_name, model.elastic_element.eps_e, time_handler, output_folder))
	if settings["Elements"]["Spring"]["save_strain_avg"] == True:
		field_name = settings["Elements"]["Spring"]["strain_name"]
		sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_e, time_handler, output_folder))

	# Save viscoelastic strain field
	if "KelvinVoigt" in settings["Model"]:
		if settings["Elements"]["KelvinVoigt"]["save_strain_vtk"] == True:
			field_name = settings["Elements"]["KelvinVoigt"]["strain_name"]
			sim.add_event(VtkSaver(field_name, model.elastic_element.eps_v, time_handler, output_folder))
		if settings["Elements"]["KelvinVoigt"]["save_strain_avg"] == True:
			field_name = settings["Elements"]["KelvinVoigt"]["strain_name"]
			sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.elastic_element.eps_v, time_handler, output_folder))

	# Save inelastic strain fields
	for element_name in inelastic_elements:
		index = inelastic_elements.index(element_name)
		if settings["Elements"][element_name]["save_strain_vtk"] == True:
			field_name = settings["Elements"][element_name]["strain_name"]
			sim.add_event(VtkSaver(field_name, model.inelastic_elements[index].eps_ie, time_handler, output_folder))
		if settings["Elements"][element_name]["save_strain_avg"] == True:
			field_name = settings["Elements"][element_name]["strain_name"]
			sim.add_event(AverageSaver(fem_handler.dx(), field_name, model.inelastic_elements[index].eps_ie, time_handler, output_folder))

	# If damage model is included, save damage field
	if "Damage" in inelastic_elements:
		i_damage = inelastic_elements.index("Damage")
		name_damage = settings["Elements"]["Damage"]["damage_name"]
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
	if type(model) == MaxwellModel_2 or type(model) == BurgersModel:
		iteration_controller = IterationController("Iterations", max_ite=20)
		error_controller = ErrorController("Error", model, tol=1e-8)
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
	shutil.copy(os.path.join(grid_folder, "geom.msh"), os.path.join(output_folder, "vtk"))
	# shutil.copy(__file__, os.path.join(output_folder, "copy.py"))
	save_json(settings, os.path.join(output_folder, "settings.json"))