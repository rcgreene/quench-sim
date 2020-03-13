import numpy as np
import numpy.linalg as lin
from scipy.linalg import null_space
from scipy.sparse.linalg import gmres
from rbf_utils import rbf, rbf_grad, rbf_d, rbf_2d

def poly_count(degree):
	return (degree + 2)*(degree + 1)//2

def get_max_degree(p_num):
	i = 0
	p_count = 0
	while p_count < p_num:
		i += 1
		p_count = poly_count(i)
	return i-1

def get_k(p_num):
	return poly_count(get_max_degree(p_num))

class MonomialEvaluator:
	def __init__(self, dim, degree):
		# Class for enumerating polynomials up to a given degree and evaluating them
		self.dim = dim
		self.degree = degree
		self.set_up_list()

	def set_up_list(self):
		# make a list representing all monomials up to a given degree, represented by a list of the powers of the dim first degree monomials
		self.terms = [tuple([0,]*self.dim)]
		mono = np.array([0,]*(self.dim + 1))
		for i in range(self.degree):
			mono[:-1] = 0
			mono[-1] = i + 1
			while 1:
				self.terms.append(tuple(np.flip(np.array([mono[j + 1] - mono[j] for j in range(self.dim)]), axis=0)))
				if mono[1] == mono[-1]:
					break
				for j in range(self.dim - 1):
					if mono[j + 1] < mono[j + 2]:
						mono[j + 1] += 1
						break
					else:
						mono[j + 1] = 0
		self.term_num = len(self.terms)
		self.terms_ref = [u for u in self.terms if (u[1]%2) == 1]
		self.term_num_ref = len(self.terms_ref)


	def evaluate_monomials(self, vec):
		#evaluate the monomial list and all of the first two derivatives
		mono = np.ones(self.term_num)
		#derivs = np.zeros((self.term_num, t_ind(self.dim, self.dim)))
		power_mat = np.array([vec**i for i in range(self.degree + 1)])
		for i in range(self.term_num):
			for j in range(self.dim):
				mono[i] *= power_mat[self.terms[i][j], j]

		return mono

	def evaluate_monomials_ref(self, vec, rotation):
		mono = np.ones(self.term_num_ref)
		output = []
		rot_vec = rotation @ vec
		power_mat = np.array([(rot_vec)**i for i in range(self.degree + 1)])
		for i in range(self.term_num_ref):
			for j in range(self.dim):
				mono[i] *= power_mat[self.terms_ref[i][j], j]

		return mono

	def evaluate_mono_derivs(self, vec, direct):
		# Evaluate the first derivatives in both directions, add based on weight provided by vector 'direct'
		mono = np.ones((2, self.term_num))
		power_mat = np.array([vec**i for i in range(self.degree + 1)])
		for k in range(2):
			for i in range(self.term_num):
				for j in range(self.dim):
					if j == k:
						mono[k, i] *= self.terms[i][j]
						if self.terms[i][j] == 0:
							break
						else:
							mono[k, i] *= power_mat[self.terms[i][j] - 1, j]
					else:
						mono[k, i] *= power_mat[self.terms[i][j], j]

		return mono[0]*direct[0] + mono[1]*direct[1]

	def evaluate_mono_derivs_ref(self, vec, rotation):
		mono = np.ones((5, self.term_num_ref))
		rot_vec = rotation @ vec
		power_mat = np.array([vec**i for i in range(self.degree + 1)])
		for k in range(2):
			for i in range(self.term_num_ref):
				for j in range(self.dim):
					if j == k:
						mono[k, i] *= self.terms_ref[i][j]
						mono[k + 2, i] *= (self.terms_ref[i][j]**2 - self.terms_ref[i][j])
						if self.terms[i][j] == 0:
							break
						else:
							mono[k, i] *= power_mat[self.terms_ref[i][j] - 1, j]
							if self.terms[i][j] != 1:
								mono[k + 2, i] *= power_mat[self.terms_ref[i][j] - 2, j]
					else:
						mono[k, i] *= power_mat[self.terms_ref[i][j], j]
		for i in range(self.term_num_ref):
			for j in range(self.dim):
				mono[4, i] *= self.terms_ref[i][j]
				if self.terms[i][j] == 0:
					break
				else:
					mono[4, i] *= power_mat[self.terms_ref[i][j] - 1, j]
		mono[2, :] += mono[3, :]
		mono[:2, :] = lin.inv(rotation) @ mono[:2, :]
		new_rot = lin.inv(rotation)
		for u in mono[2:, :].T:
			matrix = np.array([[u[0], u[2]],[u[2], u[1]]])
			matrix = new_rot @ matrix @ new_rot.T
			u[0], u[1] = matrix[0,0], matrix[1,1]
		return mono[:4, :]


class stencilMaker:
	def __init__(self, stencil_size):
		self.evaluator = MonomialEvaluator(2, get_max_degree(stencil_size))

	def rbf_ga(self, coord_list, boundary_list, eps=.01):
		p_num = coord_list.shape[0]
		p_mat = np.ones((get_k(p_num), p_num))
		eval_mat = np.zeros((p_num, p_num))
		eval_mat[0] = [np.exp(-eps*np.sum((u - coord_list[0])**2)) for u in coord_list]
		rhs_mat = np.zeros((p_num, 4))
		rhs_mat[0, :2] = [0 for i in range(2)]
		rhs_mat[0, 2:] = [-2*eps for i in range(2)]

		for i, u in enumerate(coord_list):
			p_mat[:, i] = self.evaluator.evaluate_monomials(u)
		vec_matrix = np.zeros((p_num, p_num))
		vec_matrix[0,0] = 1.0
		i, k = 1, 3
		degree = 0
		while i < p_num:
			degree += 1
			ns = null_space(p_mat[:i, :k]).T
			print(lin.cond(p_mat[:i, :k]))
			print('----')
			vec_matrix[i:k, :k] = ns
			storage_mat = np.array([[rbf(degree, coord_list[m], coord_list[n], eps) 
				if m not in boundary_list else rbf_grad(degree, coord_list[m], coord_list[n], boundary_list[m], eps) 
				for m in range(p_num)] for n in range(k)])
			eval_mat[i:k] = (vec_matrix[i:k, :k] @ storage_mat)/eps**degree

			d_mat = np.array([[rbf_d(degree, coord_list[0], coord_list[n], m, eps) for m in range(2)] for n in range(k)])/eps**degree
			rhs_mat[i:k, :2] = vec_matrix[i:k, :k] @ d_mat
			d_mat = np.array([[rbf_2d(degree, coord_list[0], coord_list[n], m, eps) for m in range(2)] for n in range(k)])/eps**degree
			rhs_mat[i:k, 2:] = vec_matrix[i:k, :k] @ d_mat

			i = k
			k = min(p_num, k + degree + 2)

		for i in range(p_num):
			eval_mat[:, i] *= np.exp(-eps*(coord_list[i] @ coord_list[i]))
		weights = np.zeros(rhs_mat.shape)
		for i in range(4):
			weights[:, i] = gmres(A=eval_mat, b=rhs_mat[:,i], tol=1e-8)[0]
		print(eval_mat.shape, rhs_mat.shape, weights.shape)
		print(np.max(np.abs(eval_mat @ weights - rhs_mat)))
		return weights.T