import numpy as np
import numpy.linalg as lin
import scipy as sp
import scipy.linalg as lin
import scipy.sparse as spar
from numba import jit
from scipy.optimize import minimize
from scipy.spatial import KDTree

class NodeFinder:
	def __init__(self, r_1, r_2, twist_num, h):
		
		self.r_1 = r_1
		self.r_2 = r_2
		self.del_r = (r_2 - r_1)/twist_num
		self.lam = self.del_r/(2*np.pi)     # Using the notation from the paper; not lambda because it's a keyword
		self.h = h
		self.twist_num = twist_num
		self.init_boundary_nodes()
		self.init_interior_nodes()

	def get_node_info(self):
		output = {}
		output['nodes'] = self.nodes
		output['dirichlet_num'] = self.dirichlet_num
		output['outlet_num'] = self.outlet_num
		output['pair_offset'] = self.pair_offset
		output['boundary_num'] = self.boundary_num
		output['lambda'] = self.lam
		output['width'] = self.r_2 - self.r_1
		return output

	def init_boundary_nodes(self):
		#Initialize inlet and outlet nodes--dirichlet boundary conditions
		self.nodes = []
#		n = np.ceil(2*self.del_r / self.h)       #Boundary nodes are twice as densely packed as interior nodes
#		a, b = self.r_1, self.r_1 + self.del_r
#		self.nodes = [[0.0, s] for s in np.linspace(a, b, n)]		

#		a, b = self.r_2 - 2*self.del_r, self.r_2 - self.del_r
#		self.nodes.extend([ [2*np.pi, s] for s in np.linspace(a, b, n)])

		self.outlet_num = len(self.nodes)

		#Initialize top and bottom nodes--dirichlet boundary conditions
		n = np.ceil(2*(self.r_2 - self.r_1 - self.del_r)/self.h)
		a, b = 0.0, 2*np.pi
		self.nodes.extend([ [theta, self.r_1] for theta in np.linspace(a, b, n)])

		a, b = 0.0, 2*np.pi
		self.nodes.extend([ [theta, self.r_2 - self.del_r] for theta in np.linspace(a, b, n)])		

		self.dirichlet_num = len(self.nodes)

		#Initialize wrap-around nodes, these nodes are paired. Pairs must always have the same partial derivatives.
		#n = int(np.ceil(2*(self.r_2 - self.r_1 - 2*self.del_r)/self.h))
		#self.pair_offset = n - 2
		self.pair_offset = 0
		#left pairs
		#a, b = self.r_1 + self.del_r, self.r_2 - self.del_r
		#self.nodes.extend([ [0.0, s] for s in np.linspace(a, b, n)[1:-1]])
		#right pairs
		#a, b = self.r_1, self.r_2 - 2*self.del_r
		#self.nodes.extend([ [2*np.pi, s] for s in np.linspace(a, b, n)[1:-1]])

		self.boundary_num = len(self.nodes)
		self.pair_num = int(self.boundary_num - self.dirichlet_num)

	def init_interior_nodes(self):

		#Initialize hexagonal lattice
		t_1, t_2 = np.pi/12, 5*np.pi/12
		e_1, e_2 = np.array( [np.cos(t_1), np.sin(t_1)])*self.h/(self.r_2 - self.r_1), np.array([np.cos(t_2), np.sin(t_2)])*self.h/(self.r_2 - self.r_1)
		k_num = int(np.ceil(3.0/e_1[0]))

		#insert interior nodes
		interior_nodes = []
		for i in range(-k_num, k_num):
			for j in range(-k_num, k_num):
				x = i*e_1 + j*e_2
				frac = 0*self.twist_num
				if x[1] > 1 - frac:
					x[1] = 1 - frac + (x[1] + frac - 1)/20
				if x[1] < frac:
					x[1] =  frac - (frac - x[1])/10
				if np.all(np.abs(x - .5) < .5):
					x = -np.cos(x*np.pi)/2 + .5
					x[1] = x[1]*(self.r_2 - self.r_1 - self.del_r) + self.r_1
					x[0] = 2*np.pi*x[0]
					self.nodes.append(x)
		self.nodes = np.array(self.nodes)

class NodeFinderSimple(NodeFinder):
	# node finder for the special case of a domain whose inlet/outlet covers the entire left and right boundaries
	def __init__(self, r_1, r_2, h):
		NodeFinder.__init__(self, r_1, r_2, 1e999, h)
		self.nodes[:,0] *= .5

	def init_boundary_nodes(self):
		self.del_r = 0
		n = np.ceil(2*(self.r_2 - self.r_1)/self.h)
		self.nodes = [[0.0, s] for s in np.linspace(self.r_1, self.r_2, n)[1:-1]]
		self.nodes.extend([2*np.pi, s] for s in np.linspace(self.r_1, self.r_2, n)[1:-1])
		self.outlet_num = len(self.nodes)

		self.nodes.extend([alpha, self.r_1] for alpha in np.linspace(0, 2*np.pi, n))
		self.nodes.extend([alpha, self.r_2] for alpha in np.linspace(0, 2*np.pi, n))

		self.dirichlet_num = len(self.nodes)

		self.pair_offset = 0
		self.pair_num = 0
		self.boundary_num = self.dirichlet_num
