import matplotlib.pyplot as plt 
import random
import copy
import math
from itertools import product
import os
import numpy as np

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
    def __init__(self, targets, types=2, alpha1=1, alpha2=0, beta1=1, beta2=0.5, threshold=0.9, coeff=(1, 1, 1)):
        self.targets = targets
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.types = types 
        self.type_sensor = range(types)
        self.coeff = coeff
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
        best_true_covered = None
        random_cases = [[random.randint(0, self.types-1) for j in range(len(used))] for i in range(1)]
        for case in random_cases:
            covered = []
            true_covered = []
            for t in targets:
                nov = 0
                pt = 1
                for index, sensor in enumerate(used):
                    if self._distance(sensor, t) > R[case[index]]:
                        continue
                
                    if t not in true_covered:
                        true_covered.append(t)
                    
                    p = self._psm(sensor, t, type=case[index])
                    pt = pt*p
                    nov += 1
                pt = 1.-pt
                # if pt >= self.threshold or (nov==1 and (1.-pt)>=self.threshold):
                if pt>=self.threshold or (nov==1):
                    covered.append(t)
            
            min_dist_sensor = float('+inf')
            if len(used) == 1:
                min_dist_sensor = 0.001
            else:
                for ia, a in enumerate(used):
                    for ib, b in enumerate(used):
                        if a!=b:
                            min_dist_sensor = min(min_dist_sensor, self._distance(a, b))*(R[case[ia]]*R[case[ib]])
                            
            
            obj1 = (len(covered)/no_cells)
            obj2 = (len(used)-min(min_noS))/(max(max_noS)-min(min_noS))
            obj3 =  min_dist_sensor/(max(self.diagonal)*max(R)*max(R))
            
            obj1 = obj1**self.coeff[0]
            obj2 = (1/(10*obj2+1))**self.coeff[1]
            obj3 = obj3**self.coeff[2] 
            

            ## 3s/3 objective 
            obj = obj1 * obj2 * obj3

            
            if obj > best_obj:
                best_obj = obj
                best_covered = len(covered)
                best_case = case
                best_true_covered = len(true_covered)

        return (best_obj, best_covered, len(used), best_true_covered),  best_case
    
   

class HarmonySearch(object):
    def __init__(self, hms=30, hmv=25, hcmr=0.9, par=0.3, BW=0.2, threshold=0.9, coeff=(1, 1, 1), root_dir='./log'):
        
        self.threshold = threshold
        self.coeff = coeff

        self._obj_fun = ObjectiveFunction(targets=targets, coeff=self.coeff)

        self.hms = hms
        self.size = hmv
        self.hcmr = hcmr
        self.par = par
        self.BW = BW
        self.root_dir = root_dir
        self.best_obj_dir = os.path.join(self.root_dir, 'bestobjlog')
        self.best_cov_dir = os.path.join(self.root_dir, 'bestcovlog')
        if not os.path.exists(self.root_dir):
            print('Make log dir')
            os.makedirs(self.root_dir)
            os.makedirs(self.best_obj_dir)
            os.makedirs(self.best_cov_dir)
        else:
            raise ValueError('Save in another dir')

        with open(os.path.join(self.root_dir, 'config.txt'), 'w') as f:     
            f.write("HMS: {}".format(self.hms))
            f.write("\n")
            f.write("HMV: {}".format(self.size))
            f.write("\n")
            f.write("HCMR: {}".format(self.hcmr))
            f.write("\n")
            f.write("PAR: {}".format(self.par))
            f.write("\n")
            f.write("BW: {}".format(self.BW))
            f.write("\n")
            f.write("Coeff: {}".format(self.coeff))
            f.write("\n")
            f.write("Threshold: {}".format(self.threshold))



    def run(self, step=10000):
        
        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()

        # save the best cov
        best_harmony_cov = None
        best_fitness_cov = None 
        best_case_cov = None 
        best_cov = float('-inf')
        best_harmony = None
        best_case = None
        best_fitness = None 
        best = float('-inf')


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
            

            best_index, best_index_cov = self._update_harmony_memory(new_harmony, new_fitness, new_case)
            
            best_harmony_step, best_fitness_step, best_case_step = self._harmony_memory[best_index]
            best_harmony_cov_step, best_fitness_cov_step, best_case_cov_step = self._harmony_memory[best_index_cov]

            ## Update global best obj and best cov

            if best_fitness_cov_step[1] > best_cov or (best_fitness_cov_step[1]==best_cov and best_fitness_cov_step[2] < best_fitness_cov[2]):
                best_cov = best_fitness_cov_step[1]
                best_harmony_cov = best_harmony_cov_step
                best_case_cov = best_case_cov_step
                best_fitness_cov = best_fitness_cov_step
                
                self._save_result(self.best_cov_dir, best_harmony_cov_step, best_fitness_cov_step, best_case_cov_step)

            
            if best_fitness_step[0] > best:
                best = best_fitness_step[0]
                best_harmony = best_harmony_step
                best_case = best_case_step
                best_fitness = best_fitness_step

                self._save_result(self.best_obj_dir, best_harmony_step, best_fitness_step, best_case_step)
            
            self._save_log(self.best_obj_dir, best_harmony_step, best_fitness_step, best_case_step)
            self._save_log(self.best_cov_dir, best_harmony_cov_step, best_fitness_cov_step, best_case_cov_step)

            print("gen", generation, " with best =", best_fitness_step[0], "cover: ", best_fitness_step[1], "true covered:", best_fitness_step[3], " used: ", best_fitness_step[2], " case:", best_case_step)

            generation += 1
            
            self._harmony_history.append({'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)})
        

        return best_harmony, best_fitness, best_case, best_harmony_cov, best_fitness_cov, best_case_cov
    
    def _save_result(self, dir, harmony, fitness, case):
        with open(os.path.join(dir, 'result.txt'), 'w') as f:
            f.write(str(harmony))
            f.write('\n')
            f.write(str(fitness))
            f.write('\n')
            f.write(str(case))
            f.write('\n')

    def _save_log(self, dir, harmony, fitness, case):
        with open(os.path.join(dir, 'harmony.txt'), 'a') as f:
            f.write(str(harmony))
            f.write('\n')
        with open(os.path.join(dir, 'obj.txt'), 'a') as f:
            f.write(str(fitness[0]))
            f.write('\n')
        with open(os.path.join(dir, 'covered.txt'), 'a') as f:
            f.write(str(fitness[1]))
            f.write('\n')
        with open(os.path.join(dir, 'truecovered.txt'), 'a') as f:
            f.write(str(fitness[3]))
            f.write('\n')
        with open(os.path.join(dir, 'used.txt'), 'a') as f:
            f.write(str(fitness[2]))
            f.write('\n')
        with open(os.path.join(dir, 'case.txt'), 'a') as f:
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
            best_cov_index = 0
            best_cov = float('-inf')
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
                if fitness[1] >= best_cov:
                    best_cov = fitness[1]
                    best_cov_index = i
            
            if (considered_fitness[0] >= worst):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness, considered_case)
        
        else:
            best_cov_index = 0
            best_cov = float('-inf')
            best_index = 0 
            best = float('-inf')
            for i, (harmony, fitness, _) in enumerate(self._harmony_memory):
                if fitness[0] >= best:
                    best = fitness[0]
                    best_index = i 
                if fitness[1] >= best_cov:
                    best_cov = fitness[1]
                    best_cov_index = i
        
        return best_index, best_cov_index


root_dir = './main/base4'

hsa = HarmonySearch(root_dir=root_dir, coeff=(1, 1, 1))

best_harmony, best_fitness, best_case, best_harmony_cov, best_fitness_cov, best_case_cov = hsa.run(60000)



xtarget = [x[0] for x in targets]
ytarget = [x[1] for x in targets]

xsensor_obj = [x[0] for x in best_harmony]
ysensor_obj = [x[1] for x in best_harmony]
xsensor_obj = [x for x in xsensor_obj if x!=-1]
ysensor_obj = [x for x in ysensor_obj if x!=-1]


xsensor_cov = [x[0] for x in best_harmony_cov]
ysensor_cov = [x[1] for x in best_harmony_cov]
xsensor_cov = [x for x in xsensor_cov if x!=-1]
ysensor_cov = [x for x in ysensor_cov if x!=-1]

fig, (ax1, ax2) = plt.subplots(2)

ax1.scatter(xtarget, ytarget, marker=".")
ax1.scatter(xsensor_obj, ysensor_obj, marker="*")
for s in range(len(xsensor_obj)):
    if xsensor_obj[s] == -1:
        continue
    ax1.add_patch(plt.Circle((xsensor_obj[s], ysensor_obj[s]), R[best_case[s]], color='r', alpha=0.5, fill=False))
ax1.set_aspect('equal', adjustable='datalim')
ax1.plot()
ax1.grid()
# ax1.savefig(os.path.join(root_dir, 'fig_obj.png'))

ax2.scatter(xtarget, ytarget, marker=".")
ax2.scatter(xsensor_cov, ysensor_cov, marker="*")
for s in range(len(xsensor_cov)):
    if xsensor_cov[s] == -1:
        continue
    ax2.add_patch(plt.Circle((xsensor_cov[s], ysensor_cov[s]), R[best_case_cov[s]], color='r', alpha=0.5, fill=False))
ax2.set_aspect('equal', adjustable='datalim')
ax2.plot()
ax2.grid()
# ax2.savefig(os.path.join(root_dir, 'fig_cov.png'))
plt.savefig(os.path.join(root_dir, 'fig.png'))

plt.show()


