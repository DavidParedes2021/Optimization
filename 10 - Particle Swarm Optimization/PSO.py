import numpy  as np
import NSGAII as nsga
from copy import deepcopy
from copy import copy

class ParticleSwarmOptimization:
	def __init__(self,
		fitness_func,
		bounds:np.ndarray,

		# Optional parameters
		function_args  = (),
		pop_size:int   = 50,
		max_it:int     = 1000,
		inertia:float  = 0.9,
		lazyness:float = 0.9,
		enviness:float = 0.9,

		# Really optional parameters
		min_vel:float = 0.01,
		max_vel:float = 1.0
	) -> None:
		self.func   = fitness_func
		self.args   = function_args
		self.bounds = bounds
		self.n_var  = bounds.shape[0]

		self.p_size = pop_size
		self.max_it = max_it
		self.k      = 0

		self.min_v  = min_vel
		self.max_v  = max_vel

		# Hyperparameters
		self.w   = inertia
		self.laz = lazyness
		self.env = enviness

		self.w_delta   = (self.w - 0.1) / self.max_it
		self.laz_delta = (self.laz-0.1) / self.max_it
		self.env_delta = (self.env-0.1) / self.max_it

		# Optimize function variables
		self.X      = None
		self.V      = None
		self.Fit    = None
		self.P_best = None
		self.elite  = None
	# end __init__
	def __init_population(self):
		self.X      = None
		self.V      = None
		self.Fit    = None
		self.P_best = None
		self.elite  = None

		self.X = np.zeros((self.p_size, self.n_var))
		self.V = np.zeros((self.p_size, self.n_var))
		self.P_best = np.zeros_like(self.X)

		for i, bound in enumerate(self.bounds):
			self.X[:, i] = np.random.uniform(bound[0], bound[1], (self.p_size))
			self.V[:, i] = np.random.uniform(self.min_v, self.max_v, (self.p_size))
		self.P_best = np.array(deepcopy(self.X))
	# end __init_population
	def __fitness(self, individual):
		return self.func(individual, *self.args)
	# end __fitness
	def __calc_fitness(self):
		if self.Fit is None:
			self.Fit = np.zeros((self.p_size))
		for i in range(self.p_size):
			self.Fit[i] = self.__fitness(self.X[i])
	# end __calc_fitness
	def __get_elite(self, copy:bool=True):
		elite_idx = np.argmin(self.Fit)

		if copy:
			return elite_idx, np.array(deepcopy(self.X[elite_idx]))
		return elite_idx
	# end __get_elite

	def __update_velocity(self, index:int, w:float, laz:float, env:float):
		delta_v   = w * self.V[index]
		nostalgia = laz*np.random.uniform()*(self.P_best[index]-self.X[index])
		envy      = env*np.random.uniform()*(self.elite - self.X[index])

		self.V[index] = delta_v + nostalgia + envy
	# end __update_velocity

	def optimize(self):
		w   = copy(self.w)
		laz = copy(self.laz)
		env = copy(self.env)

		self.__init_population()
		self.__calc_fitness()
		elite_idx, self.elite = self.__get_elite()
		elite_fit = self.Fit[elite_idx]

		for self.k in range(1, self.max_it+1):
			for i in range(self.p_size):
				self.__update_velocity(i, w, laz, env)
				np.clip(
					self.X[i]+self.V[i],
					self.bounds[:, 0], self.bounds[:, 1],
					out=self.X[i]
				)
				# self.X[i] += self.V[i]

				fit = self.__fitness(self.X[i])

				if fit < self.Fit[i]:
					self.Fit[i] = fit
					self.P_best[i] = np.copy(self.X[i])
				if fit < elite_fit:
					elite_fit  = fit
					elite_idx  = i
					self.elite = np.array(deepcopy(self.X[i]))
			#end for i
			w   -= self.w_delta
			laz -= self.laz_delta
			env -= self.env_delta
		# end for k in max_it

		return deepcopy(self.elite)

class MultiObjectiveParticleSwarmOptimization:
	def __init__(self,
		fitness_func:list,
		bounds:np.ndarray,

		# Optional parameters
		function_args:list,
		pop_size:int   = 50,
		max_it:int     = 1000,
		inertia:float  = 0.9,
		lazyness:float = 0.9,
		enviness:float = 0.9,

		# Really optional parameters
		min_vel:float = 0.5,
		max_vel:float = 10.0
	) -> None:
		self.func   = fitness_func
		self.args   = function_args
		self.bounds = bounds
		self.n_tar  = len(fitness_func)
		self.n_var  = bounds.shape[0]

		self.p_size = pop_size
		self.max_it = max_it
		self.k      = 0

		self.min_v  = min_vel
		self.max_v  = max_vel

		# Hyperparameters
		self.w   = inertia
		self.laz = lazyness
		self.env = enviness

		self.w_delta   = (self.w - 0.1) / self.max_it
		self.laz_delta = (self.laz-0.1) / self.max_it
		self.env_delta = (self.env-0.1) / self.max_it

		# Optimize function variables
		self.X      = None
		self.V      = None
		self.Fit    = None
		self.Lev    = None
		self.P_best = None
		self.elite  = None
	# end __init__
	def __init_population(self):
		self.X      = None
		self.V      = None
		self.Fit    = None
		self.Lev    = None
		self.P_best = None
		self.elite  = None

		self.X = np.zeros((self.p_size, self.n_var))
		self.V = np.zeros((self.p_size, self.n_var))
		self.P_best = np.zeros_like(self.X)

		for i, bound in enumerate(self.bounds):
			self.X[:, i] = np.random.uniform(bound[0], bound[1], (self.p_size))
			self.V[:, i] = np.random.uniform(self.min_v, self.max_v, (self.p_size))
		self.P_best = np.array(deepcopy(self.X))
	# end __init_population
	def __fitness(self, individual):
		return np.array(
			[self.func[i](individual, *self.args[i]) for i in range(self.n_tar)]
		)
	# end __fitness
	def __calc_fitness(self):
		if self.Fit is None:
			self.Fit = np.zeros((self.p_size, self.n_tar))
		for i in range(self.p_size):
			self.Fit[i] = self.__fitness(self.X[i])
		self.Lev = nsga.NSGA2(self.Fit)
	# end __calc_fitness
	def __get_elite(self):
		leaders = self.Lev == 0
		return deepcopy(self.X[leaders])
	# end __get_elite

	def __update_velocity(self, index:int, w:float, laz:float, env:float):
		delta_v   = w * self.V[index]
		nostalgia = laz*np.random.uniform()*(self.P_best[index]-self.X[index])
		random_elite_idx = np.random.randint(0, len(self.elite))
		envy      = env*np.random.uniform()*(self.elite[random_elite_idx] - self.X[index])

		self.V[index] = delta_v + nostalgia + envy
	# end __update_velocity

	def optimize(self): #TODO: put the multi-objective changes in the algorithm
		w   = copy(self.w)
		laz = copy(self.laz)
		env = copy(self.env)

		self.__init_population()
		self.__calc_fitness()
		self.elite = self.__get_elite()

		for self.k in range(1, self.max_it+1):
			for i in range(self.p_size):
				self.__update_velocity(i, w, laz, env)
				np.clip(
					self.X[i]+self.V[i],
					self.bounds[:, 0], self.bounds[:, 1],
					out=self.X[i]
				)

				fit = self.__fitness(self.X[i])

				if nsga.compare(fit, self.Fit[i]) == 1:
					self.Fit[i] = fit
					self.P_best[i] = np.copy(self.X[i])
			#end for i
			## TODO: update elite
			self.elite = self.__get_elite()
			w   -= self.w_delta
			laz -= self.laz_delta
			env -= self.env_delta
		# end for k in max_it

		return deepcopy(self.elite)
