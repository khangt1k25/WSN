import matplotlib.pyplot as plt 
import random
import copy
import math
from itertools import product
import os

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


min_noS = [int((W*H)/(36*ue*ue)) for ue in UE]
max_noS = [int((W*H)/(4*ue*ue)) for ue in UE]



class ObjectiveFunction(object):
    def __init__(self, targets, types=2, alpha1=1, alpha2=0, beta1=1, beta2=0.5, threshold=0.9):
        self.targets = targets
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.types = types 
        self.type_sensor = range(types)
        self.diagonal = [self._distance([W, H], [R[i]+UE[i], R[i]+UE[i]]) for i in range(len(R))]
    
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
            return math.exp(-(self.alpha1*lambda1/lambda2)) + self.alpha2

    def get_fitness(self, x):
        
        used = []
        for sensor in x:
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            else:
                used.append(sensor)
        
        if len(used) < min(min_noS):
            return (float('-inf'), 0, 0), [0]*len(used)
        
        # 2^n cal
        best_obj = float('-inf')
        best_covered = None
        best_case = None
        random_cases = [[random.randint(0, self.types-1) for j in range(len(used))] for i in range(10)]
        for case in random_cases:
            covered = []
            for t in targets:
                nov = 0
                pt = 1
                for index, sensor in enumerate(used):
                    if self._distance(sensor, t) > R[case[index]]:
                        continue
                    p = self._psm(sensor, t, type=case[index])
                    pt = pt*p
                    nov += 1
                pt = 1.-pt
                if pt >= self.threshold or (nov==1 and (1.-pt)>=self.threshold):
                    covered.append(t)
            
            min_dist_sensor = float('+inf')
            if len(used) == 1:
                min_dist_sensor = 0.001
            else:
                for ia, a in enumerate(used):
                    for ib, b in enumerate(used):
                        if a!=b:
                            min_dist_sensor = min(min_dist_sensor, self._distance(a, b))
                            # *(R[case[ia]]*R[case[ib]])
            
            ## 3s/3 objective 
            obj = (len(covered)/no_cells)**10 * 1/((len(used)-min(min_noS))*0.999/(max(max_noS)-min(min_noS)) + 0.001) * (min_dist_sensor/max(self.diagonal))
            
            if obj > best_obj:
                best_obj = obj
                best_covered = len(covered)
                best_case = case

        return (best_obj, best_covered, len(used)), best_case
    
    def get_best_fitness(self, x):
        used = []
        for sensor in x:
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            else:
                used.append(sensor)
        
        if len(used) < min(min_noS):
            return float('-inf'), 0, 0, 0
        
        # 2^n cal
        best_obj = float('-inf')
        best_case = None
        best_covered = None
        for case in product(self.type_sensor, repeat=len(used)):
            covered = []
            for t in targets:
                nov = 0
                pt = 1
                for index, sensor in enumerate(used):
                    
                    if self._distance(sensor, t) > R[case[index]]:
                        continue
                    p = self._psm(sensor, t, type=case[index])
                    pt = pt*p
                    nov += 1
                
                pt = 1.-pt
                if pt >= self.threshold or (nov==1 and (1.-pt)>=self.threshold):
                    covered.append(t)
            
            min_dist_sensor = float('+inf')
            if len(used) == 1:
                min_dist_sensor = 0.001
            else:
                for ia, a in enumerate(used):
                    for ib, b in enumerate(used):
                        if a!=b:
                            min_dist_sensor = min(min_dist_sensor, self._distance(a, b)/(R[case[ia]]*R[case[ib]]))
            
            

            ## 3s/3 objective 
            obj = (len(covered)/no_cells)* 1/((len(used)-min(min_noS))*0.999/(max(max_noS)-min(min_noS)) + 0.001) * min_dist_sensor/max(self.diagonal)
            
            
            if obj > best_obj:
                best_obj = obj
                best_case = case
                best_covered = len(covered)
        
        
        return best_obj, best_covered, len(used), best_case

class HarmonySearch(object):
    def __init__(self, objective_function, hms=30, hmv=35, hcmr=0.9, par=0.3, BW=0.2, save_dir='./log'):
        
        self._obj_fun = objective_function

        self.hms = hms
        self.size = hmv
        self.hcmr = hcmr
        self.par = par
        self.BW = BW
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            print('Make log dir')
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'config.txt'), 'w') as f:
            
            f.write("HMS: {}".format(self.hms))
            f.write("\n")
            f.write("HMV: {}".format(self.size))
            f.write("\n")
            f.write("HCMR: {}".format(self.hcmr))
            f.write("\n")
            f.write("PAR: {}".format(self.par))
            f.write("\n")
            f.write("BW: {}".format(self.BW))


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

                    if random.random() < self.par and (new_harmony[i][0] != -1 and new_harmony[i][1]!= -1):
                        while True:
                            _amount = self._pitch_adjustment(i)
                            new_harmony[i][0] += _amount[0]
                            new_harmony[i][1] += _amount[1]
                            if self._checkValidPosition(new_harmony[i]):
                                break 
                            new_harmony[i][0] -= _amount[0]
                            new_harmony[i][1] -= _amount[1]
                else:
                    new_harmony.append(self._random_selection())
            
            new_fitness, new_case = self._obj_fun.get_fitness(new_harmony)
            

            best_index = self._update_harmony_memory(new_harmony, new_fitness, new_case)
            best_fitness = self._harmony_memory[best_index][1]
            best_case = self._harmony_memory[best_index][2]
            
            self._save_log(best_index, best_fitness, best_case)

            print("gen", generation, " with best =", best_fitness[0], "cover: ", best_fitness[1], " used: ", best_fitness[2], " case:", best_case)

            generation += 1
            
            self._harmony_history.append({'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)})
        
    
        # return best harmony
        best_harmony = None
        best = float('-inf')
        for harmony, fitness, _ in self._harmony_memory:
            if fitness[0] > best:
                best = fitness[0]
                best_harmony = harmony

        return best_harmony
    
    def _save_log(self, index, fitness, case):
        with open(os.path.join(self.save_dir, 'index.txt'), 'a') as f:
            f.write(str(index))
            f.write('\n')
        with open(os.path.join(self.save_dir, 'obj.txt'), 'a') as f:
            f.write(str(fitness[0]))
            f.write('\n')
        with open(os.path.join(self.save_dir, 'covered.txt'), 'a') as f:
            f.write(str(fitness[1]))
            f.write('\n')
        with open(os.path.join(self.save_dir, 'used.txt'), 'a') as f:
            f.write(str(fitness[2]))
            f.write('\n')
        with open(os.path.join(self.save_dir, 'case.txt'), 'a') as f:
            f.write(str(case))
            f.write('\n')   

    def _checkValidPosition(sel, position):
        if(position[0] >= 0 and position[0] <= 50 and position[1] >= 0 and position[1] <= 50):
            return True
        if(position[0] == -1 and position[1] == -1):
            return True
        return False

    def _initialize(self, hms, size):
     
        for i in range(0, hms):
            harmony = list()
            # case  = list()
            for j in range(0, size):
                harmony.append(self._random_selection())
                # case.append(random.randint(0, len(R)-1))

            fitness, case = self._obj_fun.get_fitness(harmony)

            self._harmony_memory.append((harmony, fitness, case))

        self._harmony_history.append({'gen': 0, 'harmonies': self._harmony_memory})

        
    def _memory_consideration(self, i):

        memory_index = random.randint(0, self.hms - 1)
        return self._harmony_memory[memory_index][0][i]


    def _pitch_adjustment(self,  i):

        return [(2*random.random()-1)*self.BW, (2*random.random()-1)*self.BW]

    
    def _random_selection(self):
        
        if random.random() < 0.8:
            choice = random.randint(0, 1)
            if choice == 0:
                x = lower[0][0] + (upper[0][0] - lower[0][0])*random.random()
                y = lower[0][1] + (upper[0][1] - lower[0][1])*random.random()
            else:
                x = lower[1][0] + (upper[1][0] - lower[1][0])*random.random()
                y = lower[1][1] + (upper[1][1] - lower[1][1])*random.random()
        else: # unused position
            x = -1 
            y = -1
        return [x, y]

    def _update_harmony_memory(self, considered_harmony, considered_fitness, considered_case):
       
        if (considered_harmony, considered_fitness, considered_case) not in self._harmony_memory:
            best_index = 0
            best = float('-inf')
            worst_index = None
            worst = float('+inf')
            for i, (harmony, fitness, _) in enumerate(self._harmony_memory):
                if fitness[0] <= worst:
                    worst = fitness[0]
                    worst_index = i
                if fitness[0] >= best:
                    best = fitness[0]
                    best_index = i 
            
            if (considered_fitness[0] > worst):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness, considered_case)
        
        else:
            best_index = 0 
            best = float('-inf')
            for i, (harmony, fitness, _) in enumerate(self._harmony_memory):
                if fitness[0] >= best:
                    best = fitness[0]
                    best_index = i 
        
        return best_index


obj = ObjectiveFunction(targets)
hsa = HarmonySearch(obj, save_dir='./log')

best_harmony= hsa.run(100)

best_global = obj.get_fitness(best_harmony)

print("Best Harmony ", best_harmony)
print("Best Global", best_global[1])





xtarget = [x[0] for x in targets]
ytarget = [x[1] for x in targets]
xsensor = [x[0] for x in best_harmony]
ysensor = [x[1] for x in best_harmony]


fig, ax = plt.subplots()
plt.scatter(xtarget, ytarget, marker='x')
plt.scatter(xsensor, ysensor, marker='^')
for s in range(len(xsensor)):
    if xsensor[s] == -1:
        continue
    ax.add_patch(plt.Circle((xsensor[s], ysensor[s]), 5, color='r', alpha=0.5, fill=False))
ax.set_aspect('equal', adjustable='datalim')
ax.plot()

plt.grid()
plt.show()


