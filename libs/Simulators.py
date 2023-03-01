

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
		