from fenics import Mesh, MeshFunction
import numpy as np
import meshio
import os

class GridHandler(object):
	def __init__(self, geometry_name, grid_folder):
		self.grid_folder = grid_folder
		self.geometry_name = geometry_name

		self.load_mesh()
		self.get_tags()
		self.get_grid_dimensions()
		self.load_subdomains()
		self.load_boundaries()
		self.get_box_dimensions()

	def load_mesh(self):
		# Load grid and mesh tags
		file_name_xml = os.path.join(self.grid_folder, self.geometry_name+".xml")
		self.mesh = Mesh(file_name_xml)

	def get_tags(self):
		file_name_msh = os.path.join(self.grid_folder, self.geometry_name+".msh")
		grid = meshio.read(file_name_msh)
		self.tags = {1:{}, 2:{}, 3:{}}
		for key, value in grid.field_data.items():
			self.tags[value[1]][key] = value[0]
		self.dolfin_tags = self.tags

	def get_grid_dimensions(self):
		self.domain_dim = self.mesh.topology().dim()
		self.boundary_dim = self.domain_dim - 1

	def load_subdomains(self):
		file_name_physical_region_xml = os.path.join(self.grid_folder, self.geometry_name+"_physical_region.xml")
		subdomains0 = MeshFunction("size_t", self.mesh, file_name_physical_region_xml)
		self.subdomains = MeshFunction("size_t", self.mesh, self.domain_dim)
		self.subdomains.set_all(0)
		for i, value in enumerate(self.tags[self.domain_dim].items()):
			self.subdomains.array()[subdomains0.array() == value[1]] = i + 1
			self.dolfin_tags[self.domain_dim][value[0]] = i + 1

	def load_boundaries(self):
		file_name_facet_region_xml = os.path.join(self.grid_folder, self.geometry_name+"_facet_region.xml")
		boundaries0 = MeshFunction("size_t", self.mesh, file_name_facet_region_xml)
		self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
		self.boundaries.set_all(0)
		for i, value in enumerate(self.tags[self.boundary_dim].items()):
			self.boundaries.array()[boundaries0.array() == value[1]] = i + 1
			self.dolfin_tags[self.boundary_dim][value[0]] = i + 1

	def get_box_dimensions(self):
		# Get geometrical data
		self.Lx = self.mesh.coordinates()[:,0].max() - self.mesh.coordinates()[:,0].min()
		self.Ly = self.mesh.coordinates()[:,1].max() - self.mesh.coordinates()[:,1].min()
		self.Lz = self.mesh.coordinates()[:,2].max() - self.mesh.coordinates()[:,2].min()

