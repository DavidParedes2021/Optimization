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
        report_interval:int = 10,
        eps:float           = 0.01,
        n_rep:int           = None,
        verbose:bool        = False
    ):
        self.verbose   = verbose
        self.func      = func
        self.bounds    = np.array(bounds)
        self.args      = args
        self.popsize   = popsize
        self.mutation  = mutation
        self.crossover = crossover

        self.population = None
        self.fitness    = None

        self.nvar  = len(self.bounds)
        self.maxit = maxit
        self.eps   = eps
        self.n_rep = n_rep
        self.report_interval = report_interval
        self.n_rep     = n_rep
        if self.n_rep is None:
            self.n_rep = int(self.maxit * 0.10)
    # end __init__
        
    def __init_population(self):
        self.population = np.zeros((self.popsize,self.nvar))
        for i in range(self.nvar):
            self.population[:,i] = np.random.uniform(self.bounds[i,0],self.bounds[i,1],(self.popsize))
    
    def __fitness_population(self):
        self.fitness = np.zeros((self.popsize))
        for i in range(self.popsize):
            self.fitness[i] = self.func( self.population[i,:], *self.args )
            
    def __fitness(self, ind):
        return self.func( ind , *self.args )
            
    def __mutation(self):
        r1,r2,r3 = np.random.randint(0, self.popsize, 3)
        f = np.random.uniform(self.mutation[0],self.mutation[1])
        return self.population[r1] + f*(self.population[r2]-self.population[r3])
    
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
        
    def optimize(self):
        self.__init_population()
        self.__fitness_population()
        k = 0

        vi  = np.zeros((self.nvar))
        rep = np.zeros((self.n_rep))
        elite_idx = 0
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
            
            rep[k%self.n_rep] = elite_idx = self.__get_elite()[2]

            if k % self.report_interval == 0:
                n_rep = np.count_nonzero(rep == elite_idx)
                if n_rep == self.n_rep:
                    success = True
                    break
                if self.verbose:
                    elite, elite_fit, elite_idx = self.__get_elite()
                    print(f'{k} elite_idx: {elite_idx} n_rep {n_rep} elite fitness: {elite_fit} elite: {elite}')
            k += 1

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