import numpy as np
import copy

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
                 demand,
                 max_cap,
                 n_nodes,
                 n_ants: int,
                 plot_function,
                 max_it: int = 100,
                 evaporation_rate: float = 0.1,
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
        self.demand = np.array(demand)
        self.max_cap = max_cap
        self.n_nodes = n_nodes
        self.pending_customers_all = set(range(self.n_nodes))

        self.heuristic_calc_func = self.__default_h_calc
        if heuristic_calc_func is not None:
            self.heuristic_calc_func = heuristic_calc_func

        self.pher_mxt = None
        self.hstc_mtx = None
        self.best_ant_global = None
        self.best_ant_global_length = None

    def __init_var(self):
        self.hstc_mtx = self.__default_h_calc(self.dist_mtx)
        self.pher_mxt = np.ones_like(self.dist_mtx) 
        #self.best_ant = np.zeros((self.dist_mtx.shape[0]))
        
    def test_print(self):
        print(self.hstc_mtx)
        print(self.pher_mxt)

    def solve(self):
        self.__init_var()
        ant_path_len = np.zeros((self.n_ants))
        count_best = 0
        prev_best = None
        for k in range(self.max_it):
            ant_paths = []
            for i in range(self.n_ants):
                ant_path = []
                route = [0]
                ant_capacity = self.max_cap
                pending_customers = set(range(self.n_nodes))
                visited_customers = set([0])
                pending_customers.remove(0)
                j = 1 #current path position
                while len(pending_customers)>0:
                    probs, flag = self.__move_prob(visited_customers, int(route[j - 1]),ant_capacity )
                    if  flag == 0:
                        route.append(0)
                        j=0
                        ant_path.append(route.copy())
                        route = [0]
                        ant_capacity = self.max_cap
                    else:
                        route.append( np.random.choice(np.arange(self.dist_mtx.shape[0]), p=probs) )
                        visited_customers.add(route[j])
                        pending_customers.remove(route[j])
                        ant_capacity -= self.demand[route[j]]
                    j = j+1
                
                if route[j-1] != 0: #returns to origin
                    route.append(0)   
                    ant_path.append(route.copy())

                ant_paths.append(copy.deepcopy(ant_path))
                ant_path_len[i] = 0
                for z in range(len(ant_path)):
                    ant_path_len[i]  += self.__path_len(ant_path[z])
       
                if (self.best_ant_global_length == None) or ant_path_len[i] < self.best_ant_global_length:
                    self.best_ant_global = ant_path.copy()
                    self.best_ant_global_length = ant_path_len[i]
                    
                #actualizacion feromonas
                for d in range(len(ant_path)):
                    self.__update_pheromones(ant_path[d], ant_path_len[i])
            
            # evaporación
            self.pher_mxt *= (1- self.evapr)

            if k%100==0 and self.verbose == True:
                print("it: ",k)
                self.plot_function(self)

        return self.concatenate_best(), self.best_ant_global_length  
    
    def mutate2(self, ant_path):
        prob_demand = np.ones((len(ant_path)))
        routes_demand = []
        for i in ant_path:
            routes_demand.append(self.calculate_demand(i))
        routes_demand = np.array(routes_demand)   
        prob_demand /= routes_demand
        prob_demand = prob_demand / np.sum(prob_demand)
        
        rnd_route_idx = np.random.choice(np.arange(len(ant_path)), p=prob_demand)
        rnd_route_demand = self.calculate_demand(ant_path[rnd_route_idx])
        rnd_customer = np.random.randint(1, (len(ant_path[rnd_route_idx])-1))
        
        heur = np.copy(self.hstc_mtx[rnd_customer])
        heur[ant_path[rnd_route_idx]] = 0
        
        infalible_idx = np.where(self.demand + rnd_route_demand > self.max_cap)

        heur[infalible_idx] = 0
        if np.sum(heur)==0:
            return ant_path
        probs = heur/np.sum(heur)
        
        customer_selected = np.random.choice(np.arange(self.dist_mtx.shape[0]), p=probs)
        for i in range(len(ant_path)):
            if customer_selected in ant_path[i]:
                index = np.where(ant_path[i] == customer_selected)[0]
                ant_path[i] = np.delete(ant_path[i], index)
                ant_path[rnd_route_idx][len(ant_path[rnd_route_idx])-1] = customer_selected
                ant_path[rnd_route_idx] = np.append(ant_path[rnd_route_idx],0)
                break
        
        return ant_path
    
    def print_final_capacities(self):
        for p in range(len(self.best_ant_global)):
            print("route ",p," : ", self.calculate_demand(self.best_ant_global[p]))
        
    def calculate_demand(self, path):
        return np.sum(self.demand[path])
        
    def two_opt_heuristic(self, path):
        path_len = self.__path_len(path)
        #print("orig: ", path_len )
        for i in range(1, len(path)-2):
            for j in range(i+1, len(path)-1):
                new_path = np.copy(path)
                aux = new_path[i]
                new_path[i] = new_path[j]
                new_path[j] = aux
                #print("new path ", new_path)
                new_path_len = self.__path_len(new_path)
                if new_path_len < path_len:
                    path = np.copy(new_path)
                    path_len = new_path_len
        #print("final: ", path_len )
        return path

    def test(self):
        arr = np.array([20  ,5 ,25 ,10, 15 , 9 ,22 , 8 ,18, 29 ])
        best_route = self.two_opt_heuristic(arr)
        print(best_route)
    
    def __move_prob(self, visited, current, ant_capacity):
        visited = np.array(list(visited))
        pheromones = np.copy(self.pher_mxt[current])
        #print("pher size ", len(pheromones))
        heuristics = self.hstc_mtx[current]
        # Cast the values in 'visited' to integers and set corresponding pheromones to 0
        visited_indices = np.array(visited, dtype=np.int64)
        pheromones[visited_indices] = 0
    
        #infalible indices
        infalible_indices = np.where( (ant_capacity - self.demand) < 0)[0]
        pheromones[infalible_indices] = 0
        probabilities = pheromones ** self.alpha * heuristics ** self.betha
        den = np.sum(probabilities)
        if den == 0.0:
            return (0 , 0)
        probabilities = probabilities / den
        return (probabilities, 1)
    
    def concatenate_best(self):
        path = self.best_ant_global[0]
        for i in range(1,len(self.best_ant_global)):
            path = np.concatenate((path, self.best_ant_global[i][1:] ))
            
        return path

    def __path_len(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.dist_mtx[int(path[i])][int(path[i + 1])]
        return length

    def __update_pheromones(self, path, length):
        for i in range(len(path) - 1):
            self.pher_mxt[int(path[i])][int(path[i + 1])] += 1.0 / length
            self.pher_mxt[int(path[i + 1])][int(path[i])] += 1.0 / length