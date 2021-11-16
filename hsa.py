from matplotlib import markers
import matplotlib.pyplot as plt 
import random
import copy
import math

H, W = (50, 50)
R = [5, 10]
UE = [x/2 for x in R]
cell_W, cell_H = (10, 10)

no_cells = H/cell_H * W/cell_W
targets = []
for h in range(int(abs(H/cell_H))):
    for w in range(int(abs(W/cell_W))):
        targets.append((w*cell_W + cell_W/2, h*cell_H + cell_H/2))


lower = [(ue, ue) for ue in UE]
upper = [(H-l[0], W-l[1]) for l in lower]

min_noS = [int((W*H)/(36*ue**ue)) for ue in UE]
max_noS = [int((W*H)/(4*ue*ue)) for ue in UE]



class ObjectiveFunction(object):
    def __init__(self, targets, alpha1=1, alpha2=0, beta1=1, beta2=0.5, threshold=0.9):
        self.targets = targets
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold

    def _distance(self, x, y):
        return math.sqrt((x[0]-y[0])**2 + (x[1] - y[1]) ** 2)
    
    def _psm(self, x, y, type):
        distance = self._distance(x, y)

        if distance < R[type] - UE[type] : 
            return 1
        elif distance > R[type] + UE[type]:
            return 0
        else:
            lambda1 = UE[type] - R[type] + distance
            lambda2 = UE[type] + R[type] - distance
            lambda1 = math.pow(lambda1, self.beta1)
            lambda2 = math.pow(lambda2, self.beta2)
            return math.exp(-(self.alpha1*lambda1/lambda2 + self.alpha2))

    def get_fitness(self, x):
        
        used = []
        for sensor in x:
            if sensor[0] == -1 or sensor[1] == -1:
                continue
            else:
                used.append(sensor)
        
        covered = []
        for t in targets:
            pt = 1
            for sensor in used:
                p = self._psm(sensor, t, type=0)
                if p==0:
                    continue
                pt = pt*p
            pt = 1-pt

            if pt >= self.threshold:
                covered.append(t)
        

        obj = (len(used)/no_cells ) * (len(covered)-min_noS[0])/(max_noS[0]-min_noS[0])  


        return obj

            
                        
    
class HarmonySearch(object):
    def __init__(self, objective_function, hms=20, size=10, hcmr=0.9, par=0.1):
        
        self._obj_fun = objective_function

        self.hms = hms
        self.size = size
        self.hcmr = hcmr
        self.par = par 

    def run(self, step=100):

        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()

        self._initialize(self.hms, self.size)
        generation = 0
        for i in range(step):
            # generate new harmony
            new_harmony = list()
            for i in range(0, self.size):
                if random.random() < self.hcmr:
                    new_harmony.append(self._memory_consideration(i))
                    if random.random() < self.par:
                        _amount = self._pitch_adjustment(i)
                        if random.random() < 0.5:
                            new_harmony[i][0] += _amount[0]
                            new_harmony[i][1] += _amount[1]
                        else:
                            new_harmony[i][0] -= _amount[0]
                            new_harmony[i][1] -= _amount[1]
                else:
                    new_harmony.append(self._random_selection())
            
            new_fitness = self._obj_fun.get_fitness(new_harmony)
            best = self._update_harmony_memory(new_harmony, new_fitness)
          
            
            generation += 1
            
            self._harmony_history.append({'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)})
            

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf')
        for harmony, fitness in self._harmony_memory:
            if fitness > best_fitness:
                best_harmony = harmony
                best_fitness = fitness
            
        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def _initialize(self, hms, size):
     
        for i in range(0, hms):
            harmony = list()
            for j in range(0, size):
                harmony.append(self._random_selection())

            fitness = self._obj_fun.get_fitness(harmony)

            self._harmony_memory.append((harmony, fitness))

        self._harmony_history.append({'gen': 0, 'harmonies': self._harmony_memory})

        
    def _memory_consideration(self, i):

        memory_index = random.randint(0, self.hms - 1)
        return self._harmony_memory[memory_index][0][i]


    def _pitch_adjustment(self,  i):

        return [random.random(), random.random()]

    
    def _random_selection(self):
        
        if random.random() < 0.8:
            x = lower[0][0] + (upper[0][0] - lower[0][0])*random.random()
            y = lower[0][1] + (upper[0][1] - lower[0][1])*random.random()
        else: # unused position
            x = -1 
            y = -1
        return [x, y]

    def _update_harmony_memory(self, considered_harmony, considered_fitness):
        if (considered_harmony, considered_fitness) not in self._harmony_memory:
            worst_index = None
            worst_fitness = float('+inf')
            best_index = None 
            best_fitness = float('-inf')
            for i, (harmony, fitness) in enumerate(self._harmony_memory):
                if fitness < worst_fitness:
                    worst_index = i
                    worst_fitness = fitness
                if fitness > best_fitness:
                    best_index = i 
                    best_fitness = fitness
            
            if (considered_fitness > worst_fitness):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness)

            return best_index, best_fitness
obj = ObjectiveFunction(1000)
hsa = HarmonySearch(obj)

best_harmony, best_fitness, bank, history = hsa.run(30)

print(best_harmony)
print(best_fitness)


xtarget = [x[0] for x in targets]
ytarget = [x[1] for x in targets]
xsensor = [x[0] for x in best_harmony]
ysensor = [x[1] for x in best_harmony]
s = [R[0]*20*20*4 for i in range(len(xsensor))]

plt.scatter(xtarget, ytarget)
plt.scatter(xsensor, ysensor, s=s)

plt.show()