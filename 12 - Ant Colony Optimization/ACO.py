import numpy as np

class AntColonyOptimization:
	def __default_h_calc(dist_mtx:np.ndarray) -> np.ndarray:
		heuristic_mtx = np.ones_like(dist_mtx)
		heuristic_mtx /= dist_mtx

		heuristic_mtx[dist_mtx <= 0.0] = 0
		for i in range(dist_mtx.shape[0]):
			heuristic_mtx[i][i] = 0

		return heuristic_mtx
	#  __default_h_calc end -------------------------------|
	def __init__(self,
		dist_mtx: np.ndarray,
		n_ants:int,
		max_it:int=100,
		evaporation_rate:float = 0.7,
		alpha:float = 1.0,
		betha:float = 1.0,
		heuristic_calc_func = None
	) -> None:
		self.dist_mtx = dist_mtx
		self.n_ants   = n_ants
		self.max_it   = max_it
		self.evapr    = evaporation_rate
		self.alpha    = alpha
		self.betha    = betha

		self.heuristic_calc_func = self.__default_h_calc
		if heuristic_calc_func is not None:
			self.heuristic_calc_func = heuristic_calc_func

		self.pher_mxt = None
		self.hstc_mtx = None
		self.best_ant = None
	#  __init__ end ---------------------------------------|
	def __init_var(self):
		self.hstc_mtx = AntColonyOptimization.__default_h_calc(self.dist_mtx)
		self.pher_mxt = np.ones_like(self.dist_mtx) * 0.01
		self.best_ant = np.zeros((self.dist_mtx.shape[0]))
	#  __init_var end -------------------------------------|
	def solve(self):
		self.__init_var()
		ant_path     = np.zeros((self.n_ants, self.best_ant.shape[0]))
		ant_path_len = np.zeros((self.n_ants))

		for k in range(self.max_it):
			# Hacer caminar las hormigas
			# Depositar feromonas
			self.pher_mxt *= (1.0 - self.evapr)
	# solve -----------------------------------------------|
