import numpy             as np
import matplotlib.pyplot as plt

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
        self.hstc_mtx = self.heuristic_calc_func(self.dist_mtx)
        self.pher_mxt = np.ones_like(self.dist_mtx) * 0.01
        self.best_ant = np.zeros((self.max_path_length), dtype=np.int32)
        
    def test_print(self):
        print(self.hstc_mtx)
        print(self.pher_mxt)

    def solve(self):
        self.__init_var()
        ant_path = np.zeros((self.n_ants, self.max_path_length), dtype=np.int32)
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

            if k%10 == 0 and self.verbose == True:
                print("it: ",k)
                self.plot_function(self)

        return self.best_ant, self.best_ant_length

    def __move_prob(self, visited, current):
        pheromones = np.copy(self.pher_mxt[current])
        heuristics = self.hstc_mtx[current]

        # Cast the values in 'visited' to integers and set corresponding pheromones to 0
        pheromones[visited] = 0

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
    def __init_var(self):
        self.hstc_mtx = self.heuristic_calc_func(self.dist_mtx)
        self.pher_mxt = np.ones_like(self.dist_mtx) * 0.01
        self.best_ant = np.zeros((self.max_path_length), dtype=np.int32)
        
        self.best_ant_fit    = np.inf
        self.best_ant_cost   = np.inf
        self.best_ant_length = np.inf
    def __path_len(self, path):
        length = 0
        for i in range(self.max_path_length - 1):
            length += self.dist_mtx[int(path[i])][int(path[i + 1])]
        return length
    def __update_pheromones(self, path, length):
        for i in range(self.max_path_length - 1):
            self.pher_mxt[path[i]][path[i + 1]] += 1.0 / length
            self.pher_mxt[path[i + 1]][path[i]] += self.dir_balance / length

    def __init__(self,
        dist_mtx: np.ndarray,
        n_ants:   int,
        n_paths:  int,
        plot_function,
        max_capacity:     int          = None,
        demand:           np.ndarray   = None,
        starting_node:    int          = 0,
        max_it:           int          = 100,
        evaporation_rate: float        = 0.7,
        overwork_penalty: float        = 500.0,
        dir_balance:      float        = 0.8,
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
        self.overwork_pen  = overwork_penalty
        self.dir_balance   = dir_balance
        self.best_ant_cost = None
        self.best_ant_fit  = None

        self.n_paths       = n_paths
        self.starting_node = starting_node
        self.demand        = np.append([0], demand)
        if demand is None:
            self.demand = np.ones((self.n_nodes))

        self.max_capacity  = max_capacity
        if max_capacity is None:
            self.max_capacity = round(self.n_nodes / n_paths)

        self.max_path_length += n_paths - 1

    # end __init__
    def solve(self, print_int:int=None):
        if print_int is None:
            print_int = int(self.max_it / 10)
        self.__init_var()
        ant_path = np.zeros((self.n_ants, self.max_path_length), dtype=np.int32)
        ant_path_len  = np.zeros((self.n_ants))
        ant_path_cost = np.zeros((self.n_ants))
        ant_path_fit  = np.zeros((self.n_ants))

        for k in range(self.max_it):
            for i in range(self.n_ants):
                ant_capacity   = self.max_capacity
                n_stops        = 1
                overwork       = 0.0
                cant_return    = False
                ant_path[i][0] = self.starting_node

                for j in range(1, self.max_path_length):
                    dest = None
                    if n_stops >= self.n_paths:
                        cant_return = True

                    if ant_capacity <= 0 and not cant_return:
                        dest  = self.starting_node
                    else:
                        probs = self.__move_prob(ant_path[i][:j], ant_path[i][j - 1], cant_return)
                        try:
                            dest = np.random.choice(np.arange(self.n_nodes), p=probs)
                        except ValueError:
                            dest = self.starting_node


                    if dest == self.starting_node:
                        n_stops     += 1
                        overwork    += abs(ant_capacity)
                        ant_capacity = self.max_capacity
                    if cant_return:
                        overwork    += self.demand[dest]

                    ant_capacity  -= self.demand[dest]
                    ant_path[i][j] = dest

                ant_path_len [i] = self.__path_len(ant_path[i])
                ant_path_cost[i] = self.path_cost(ant_path[i])

                ant_path_fit [i] = ant_path_len[i] + overwork*self.overwork_pen

                if self.best_ant_fit is None or ant_path_fit[i] <= self.best_ant_fit:
                    self.best_ant        = np.copy(ant_path     [i])
                    self.best_ant_cost   = np.copy(ant_path_cost[i])
                    self.best_ant_length = np.copy(ant_path_len [i])
                    self.best_ant_fit    = np.copy(ant_path_fit [i])

                self.__update_pheromones(ant_path[i], ant_path_fit[i])

            self.pher_mxt *= (1.0 - self.evapr)

            if k%print_int == 0 and self.verbose == True:
                print("it: ",k)
                self.plot_function(self)

        return self.best_ant, self.best_ant_length
    # end solve
    def path_cost(self, path) -> float:
        extra_sum  = 0.0
        travel_sum = 0.0
        for i in range(path.shape[0]):
            if path[i] == self.starting_node:
                travel_sum = 0.0
            travel_sum += self.demand[path[i]]
            if travel_sum >= self.max_capacity:
                extra_sum += self.demand[path[i]]
        return extra_sum
    # end path_cost
    def __move_prob(self, visited, current, exclude_start):
        pheromones = np.copy(self.pher_mxt[current])
        heuristics = self.hstc_mtx[current]

        # Cast the values in 'visited' to integers and set corresponding pheromones to 0
        visited_indices = visited[visited != self.starting_node]
        pheromones[visited_indices] = 0
        if exclude_start:
            pheromones[self.starting_node] = 0

        probabilities = pheromones ** self.alpha * heuristics ** self.betha
        probabilities /= np.sum(probabilities)
        return probabilities

class ACOPlotter:
    def __init__(self, points) -> None:
        self.points = points
    # end __init__
    def path_plot(self, aco_instance:AntColonyOptimization):
        plt.figure(figsize=(12, 6))

        # Plot 1: Pheromone levels on edges
        plt.subplot(1, 2, 1)
        plt.scatter(self.points[:, 0], self.points[:, 1], c='red', marker='o', label='Points')
        
        max_pheromone_level = np.max(aco_instance.pher_mxt)  # Get the maximum pheromone level for normalization
        
        for i in range(aco_instance.pher_mxt.shape[0]):
            for j in range(i+1, aco_instance.pher_mxt.shape[1]):
                pheromone_level = aco_instance.pher_mxt[i, j]
                normalized_alpha = pheromone_level / max_pheromone_level  # Normalize alpha to [0, 1]
                plt.plot(
                    [self.points[i, 0], self.points[j, 0]],
                    [self.points[i, 1], self.points[j, 1]],
                    color='gray', alpha=normalized_alpha, linewidth=2
                )
        
        plt.title('Pheromone Levels on Edges')
        plt.legend()

        # Plot 2: Best path
        plt.subplot(1, 2, 2)
        
        # Visualize the original points
        plt.scatter(
            self.points[:, 0], self.points[:, 1],
            c='blue', marker='o', label='Original Points'
        )
        # Visualize the best path found by the Ant Colony Optimization
        best_path = np.append(aco_instance.best_ant, aco_instance.best_ant[0])  # Closing the loop
        best_path = best_path.astype(int)  # Convert to integers
        plt.plot(self.points[best_path, 0], self.points[best_path, 1], c='red', linestyle='-', linewidth=2, label='Best Path')
        
        # Annotate each point with its index
        for i, point in enumerate(self.points):
            plt.text(point[0], point[1], str(i), fontsize=8, ha='right', va='bottom')
        plt.title('Ant Colony Optimization - Best Path')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()

        plt.tight_layout()
        plt.show()
    def draw_path(customer_x, customer_y, path):
        # Extract coordinates for the path
        path_coordinates_x = [customer_x[i] for i in path]
        path_coordinates_y = [customer_y[i] for i in path]

        # Find the indices where the path returns to customer 0
        return_indices = [i for i, value in enumerate(path) if value == 0]

        # Plot the path, changing color when returning to customer 0
        plt.figure(figsize=(8, 8))
        for i in range(len(return_indices) - 1):
            start_index = return_indices[i]
            end_index = return_indices[i + 1]
            plt.plot(path_coordinates_x[start_index:end_index+1], path_coordinates_y[start_index:end_index+1], marker='o', linestyle='-')

        # Mark customer locations with indices
        for i, (x, y) in enumerate(zip(customer_x, customer_y)):
            plt.text(x, y, str(i), ha='center', va='center', color='red', fontweight='bold')

        plt.title('Best Solution Path with Customer Indices')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.show()
    def draw_path_i(self, aco_instance:NTargetsACO):
        path = aco_instance.best_ant

        # Extract coordinates for the path
        path_coordinates_x = [self.points[i][0] for i in path]
        path_coordinates_y = [self.points[i][1] for i in path]

        # Find the indices where the path returns to customer 0
        return_indices = [i for i, value in enumerate(path) if value == 0]

        # Plot the path, changing color when returning to customer 0
        # plt.figure(figsize=(8, 8))
        for i in range(len(return_indices) - 1):
            start_index = return_indices[i]
            end_index = return_indices[i + 1]
            plt.plot(
                path_coordinates_x[start_index:end_index+1],
                path_coordinates_y[start_index:end_index+1],
                marker='o',
                linestyle='-'
            )

        # Mark customer locations with indices
        for i, (x, y) in enumerate(self.points):
            plt.text(x, y, str(i), ha='center', va='center', color='black', fontweight='bold')

        plt.title('Best Solution Path with Customer Indices')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.show()