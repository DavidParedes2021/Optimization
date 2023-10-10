import numpy as np
class DiferentialEvolution:
    def __init__(self,
        func,
        bounds,
        args=(),
        popsize:int         = 100,
        mutation=(0.5,2),
        crossover:float     = 0.7,
        maxit:int           = 1000,
        report_interval:int = None,
        eps:float           = 0.01,
        n_rep:int           = None,
        verbose:bool        = False,
        print_elite:bool    = False
    ):
        self.verbose     = verbose
        self.print_elite = print_elite
        self.func        = func
        self.bounds      = bounds
        self.args        = args
        self.popsize     = popsize
        self.mutation    = mutation
        self.crossover   = crossover

        self.population = None
        self.fitness    = None

        self.nvar  = len(self.bounds)
        self.maxit = maxit
        self.eps   = eps
        self.n_rep = n_rep
        self.n_rep     = n_rep
        if self.n_rep is None:
            self.n_rep = int(self.maxit * 0.10)
        self.report_interval = report_interval
        if self.report_interval is None:
            self.report_interval = int(self.maxit * 0.05)
    # end __init__
        
    def __init_population(self):
        self.population = np.zeros((self.popsize,self.nvar))
        for i in range(self.nvar):
            self.population[:,i] = np.random.uniform(
                self.bounds[i,0],self.bounds[i,1],
                (self.popsize)
            )
    
    def __fitness_population(self):
        self.fitness = np.zeros((self.popsize))
        for i in range(self.popsize):
            self.fitness[i] = self.func( self.population[i,:], *self.args )
            
    def __fitness(self, ind):
        return self.func( ind , *self.args )
            
    def __mutation(self):
        r1,r2,r3 = np.random.randint(0, self.popsize, 3)
        f = np.random.uniform(self.mutation[0],self.mutation[1])
        return np.clip(
            self.population[r1] + f*(self.population[r2]-self.population[r3]),
            self.bounds[:, 0], self.bounds[:, 1]
        )
    
    def __crossover(self, xi, vi):
        rnd = np.random.randint(0, self.nvar)
        for i in range(self.nvar):
            if i==rnd: continue
            cr = np.random.uniform(0,1)
            if cr > self.crossover:
                vi[i] = xi[i]
        
    def __get_elite(self):
        i = np.argmin(self.fitness)
        return self.population[i], self.fitness[i], i
    
    def __print_progress(self, k, n_rep):
        elite, elite_fit, elite_idx = self.__get_elite()

        print(f'{(k/self.maxit)*100: 0.2f}%', end=' ')
        print(f'elite_idx: {elite_idx} n_rep {n_rep}/{self.n_rep}', end=' ')
        print(f'elite fitness: {elite_fit: 0.4f}', end=' ')
        if self.print_elite:
            print(f'elite: {elite}', end=' ')
        print('', end='\r')
        
    def optimize(self):
        self.__init_population()
        self.__fitness_population()
        k = 0

        vi  = np.zeros((self.nvar))
        # rep = np.zeros((self.n_rep))
        n_rep = 0
        old_elite_i = elite_i = self.__get_elite()[2]
        success = False
        
        while k < self.maxit:
            for i,xi in enumerate(self.population):
                vi = self.__mutation()
                self.__crossover(xi,vi)
                ui_fitness = self.__fitness(vi)

                # self.__selection(i, xi, self.fitness[i],vi,ui_fitness )
                if ui_fitness < self.fitness[i]:
                    self.fitness[i] = ui_fitness
                    self.population[i] = vi
            
            elite_i = self.__get_elite()[2]
            if(elite_i == old_elite_i):
                n_rep += 1
            else:
                n_rep = 0
            old_elite_i = elite_i

            if(n_rep == self.n_rep):
                success = True
                break

            if k % self.report_interval == 0:
                if self.verbose:
                    self.__print_progress(k, n_rep)
            k += 1
        if self.verbose:
            self.__print_progress(k, n_rep)
            print()

        elite, elite_fit, elite_idx = self.__get_elite()
        return {'success':success,'sol':elite, 'fitness':elite_fit, 'niter':k}
    
def differential_evolution(func,bounds,args=(),popsize=100,mutation=(0.5,1),crossover=0.7, maxit=1000, verbose=False):
    de = DiferentialEvolution(
        func,
        bounds,
        args=args,
        popsize=popsize,
        mutation=mutation,
        crossover=crossover,
        maxit=maxit,
        verbose=verbose
    )
    return de.optimize()