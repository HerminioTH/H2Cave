

sec = 1.
minute = 60*sec
hour = 60*minute
day = 24*hour
month = 30*day
kPa = 1e3
MPa = 1e6
GPa = 1e9

class Simulator():
	def __init__(self, model, time_handler):
		self.model = model
		self.time_handler = time_handler
		self.savers = []

	def add_saver(self, saver):
		self.savers.append(saver)

	def savers_record(self):
		for saver in self.savers:
			saver.record()

	def savers_save(self):
		for saver in self.savers:
			saver.save()

	def run(self):
		self.model.update_BCs(self.time_handler)
		self.model.solve_elastic_model(self.time_handler)
		self.model.compute_total_strain()
		self.model.compute_elastic_strain()
		self.model.compute_stress()
		self.model.update_matrices(self.time_handler)
		self.model.compute_viscous_strain()
		self.model.model_v.update()
		self.model.assemble_matrix()
		self.savers_record()

		# Time marching
		while not self.time_handler.is_final_time_reached():

			# Update time
			self.time_handler.advance_time()
			print()
			print(self.time_handler.time/hour)

			self.model.update_matrices(self.time_handler)
			self.model.assemble_matrix()
			self.model.compute_creep(self.time_handler)
			self.model.update_BCs(self.time_handler)

			# Iteration settings
			ite = 0
			tol = 1e-9
			error = 2*tol
			error_old = error

			while error > tol:
				self.model.solve_mechanics()
				error = self.model.compute_error()
				print(ite, error)
				self.model.compute_total_strain()
				self.model.compute_elastic_strain()
				self.model.compute_stress()
				self.model.compute_creep(self.time_handler)

				# Increase iteration
				ite += 1

			self.model.compute_viscous_strain()
			self.model.update_old_strains()

			# Save results
			self.savers_record()

		# Write results
		self.savers_save()
		