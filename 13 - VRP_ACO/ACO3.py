import numpy as np

class AntColonyOptimization:

    def __default_h_calc(self, dist_mtx: np.ndarray) -> np.ndarray:
        heuristic_mtx = np.ones_like(dist_mtx, dtype=np.float64)
        nonzero_indices = dist_mtx > 0.0

        heuristic_mtx[nonzero_indices] /= dist_mtx[nonzero_indices]

        for i in range(dist_mtx.shape[0]):
            heuristic_mtx[i][i] = 0

        return heuristic_mtx

    def __init__(self,
        dist_mtx: np.ndarray,
        n_ants: int,
        plot_function,
        max_it: int = 100,
        evaporation_rate: float = 0.7,
        alpha: float = 1.0,
        betha: float = 1.0,
        heuristic_calc_func=None,
        verbose=False
    ) -> None:
        self.dist_mtx = dist_mtx
        self.n_ants = n_ants
        self.max_it = max_it
        self.evapr = evaporation_rate
        self.alpha = alpha
        self.betha = betha
        self.verbose = verbose
        self.plot_function = plot_function

        self.heuristic_calc_func = self.__default_h_calc
        if heuristic_calc_func is not None:
            self.heuristic_calc_func = heuristic_calc_func

        self.pher_mxt = None
        self.hstc_mtx = None
        self.best_ant = None
        self.best_ant_length = None
        self.n_nodes         = dist_mtx.shape[0]
        self.max_path_length = self.n_nodes

    def __init_var(self):
        self.hstc_mtx = self.__default_h_calc(self.dist_mtx)
        self.pher_mxt = np.ones_like(self.dist_mtx) * 0.01
        self.best_ant = np.zeros((self.max_path_length))
        
    def test_print(self):
        print(self.hstc_mtx)
        print(self.pher_mxt)

    def solve(self):
        self.__init_var()
        ant_path = np.zeros((self.n_ants, self.max_path_length))
        ant_path_len = np.zeros((self.n_ants))
        for k in range(self.max_it):
            for i in range(self.n_ants):
                ant_path[i][0] = np.random.randint(self.n_nodes)
                for j in range(1, self.max_path_length):
                    probs = self.__move_prob(ant_path[i][:j], int(ant_path[i][j - 1]))
                    ant_path[i][j] = np.random.choice(np.arange(self.n_nodes), p=probs)

                ant_path_len[i] = self.__path_len(ant_path[i])

                if (self.best_ant_length == None) or ant_path_len[i] < self.best_ant_length:
                    self.best_ant = ant_path[i].copy()
                    self.best_ant_length = ant_path_len[i]

                self.__update_pheromones(ant_path[i], ant_path_len[i])

            self.pher_mxt *= (1.0 - self.evapr)

            if k%10 and self.verbose == True:
                print("it: ",k)
                self.plot_function(self)

        return self.best_ant, self.best_ant_length

    def __move_prob(self, visited, current):
        pheromones = np.copy(self.pher_mxt[current])
        heuristics = self.hstc_mtx[current]

        # Cast the values in 'visited' to integers and set corresponding pheromones to 0
        visited_indices = np.array(visited, dtype=np.int64)
        pheromones[visited_indices] = 0

        probabilities = pheromones ** self.alpha * heuristics ** self.betha
        probabilities /= np.sum(probabilities)
        return probabilities

    def __path_len(self, path):
        length = 0
        for i in range(self.max_path_length - 1):
            length += self.dist_mtx[int(path[i])][int(path[i + 1])]
        return length

    def __update_pheromones(self, path, length):
        for i in range(self.max_path_length - 1):
            self.pher_mxt[int(path[i])][int(path[i + 1])] += 1.0 / length
            self.pher_mxt[int(path[i + 1])][int(path[i])] += 1.0 / length


class NTargetsACO(AntColonyOptimization):
    def __init__(self,
        dist_mtx: np.ndarray,
        n_ants:   int,
        n_paths:  int,
        plot_function,
        max_capacity:     int = None,
        starting_node:    int = 0,
        max_it:           int = 100,
        evaporation_rate: float = 0.7,
        alpha: float = 1, betha: float = 1,
        heuristic_calc_func=None,
        verbose=False
    ) -> None:
        super().__init__(
            dist_mtx,
            n_ants,
            plot_function,
            max_it,
            evaporation_rate,
            alpha,
            betha,
            heuristic_calc_func,
            verbose
        )
        self.n_paths       = n_paths
        self.starting_node = starting_node

        self.max_capacity  = max_capacity
        if max_capacity is None:
            self.max_capacity = round(self.n_nodes / n_paths)

        self.max_path_length += n_paths - 1
    # end __init__
    def solve(self):
        self.__init_var()
        ant_path = np.zeros((self.n_ants, self.max_path_length))
        ant_path_len = np.zeros((self.n_ants))
        for k in range(self.max_it):
            for i in range(self.n_ants):
                ant_path[i][0] = self.starting_node

                for j in range(1, self.max_path_length):
                    probs = self.__move_prob(ant_path[i][:j], int(ant_path[i][j - 1]))
                    ant_path[i][j] = np.random.choice(np.arange(self.n_paths), p=probs)

                ant_path_len[i] = self.__path_len(ant_path[i])

                if (self.best_ant_length == None) or ant_path_len[i] < self.best_ant_length:
                    self.best_ant = ant_path[i].copy()
                    self.best_ant_length = ant_path_len[i]

                self.__update_pheromones(ant_path[i], ant_path_len[i])

            self.pher_mxt *= (1.0 - self.evapr)

            if k%10 and self.verbose == True:
                print("it: ",k)
                self.plot_function(self)

        return self.best_ant, self.best_ant_length
    # end solve