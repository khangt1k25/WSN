import random
import logging
from tqdm import tqdm
from visualize import draw
import numpy as np 
import os 
class HarmonySearch():
    def __init__(self, objective_function, AoI, cell_size, hms=30, hmv=7, hmcr=0.9, par=0.3, BW=0.2, lower=[], upper=[], min_no = 0, savedir = './baseline'):
        """
            param explaination
            
            :param hms: harmony memory size, number of vectors stored in harmony memory
            :param hmv: harmony vector size
            :param hmcr: probability for each node considering
            :param par: pitch adjustment rate
            :param BW: distance bandwidth, used for adjust node position when pich adjustment is applied
            :param lower: list contains coordinates for bottom corners
            :param upper: list contains coordinates for upper corners
        """
        self.root_dir = savedir
        self.image_dir = os.path.join(self.root_dir, 'plot')
        self.log_dir = os.path.join(self.root_dir, 'log')
        if not os.path.exists(self.root_dir):
            print('Make log dir')
            os.makedirs(self.root_dir)
            os.makedirs(self.image_dir)
            os.makedirs(self.log_dir)
        else:
            raise ValueError('Save in another dir')
        
        self._obj_function = objective_function
        self.radius = self._obj_function.get_radius()
        self.hms = hms
        self.hmv = hmv
        self.hmcr = hmcr
        self.par = par
        self.BW = BW
        self.lower = lower
        self.upper = upper
        self.min_no = min_no
        self.AoI = AoI
        self.cell_size = cell_size

        
        self.logger2 = logging.getLogger(name='best maximum coverage ratio')
        self.logger2.setLevel(logging.INFO)
        handler2 = logging.FileHandler(os.path.join(self.log_dir, 'best_maximum_coverage_ratio.log'))
        handler2.setLevel(logging.INFO)
        formatter2 = logging.Formatter('%(levelname)s: %(message)s')
        handler2.setFormatter(formatter2)
        self.logger2.addHandler(handler2)
        self.best_coverage = 0

    def _random_selection(self, min_valid):
        harmony = []
        for each_node in range(self.hmv):
            type_ = random.choice([0, 1])
            if type_ == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            harmony.append([x, y])
        return harmony
    
    def _centroid_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        # harmony = [[-1, -1]] * self.hmv
        # type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        # valid_nodes = random.sample(range(self.hmv), random.randrange(min_valid, num_width_cell*num_height_cell + 1))
        # id_valid_cell = random.sample(range(num_width_cell*num_height_cell), len(valid_nodes))
        id_valid_cell = list(range(self.hmv))
        random.shuffle(id_valid_cell)
        harmony = []
        type_trace = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0] / 2
            y = height_coor * self.cell_size[1] + self.cell_size[1] / 2
            # harmony[each_node] = [x, y]
            # type_trace[each_node] = type_
            harmony.append([x, y])
            type_trace.append(type_)
        return harmony, type_trace
        
    def _cell_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        # harmony = [[-1, -1]] * self.hmv
        # type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        # valid_nodes = random.sample(range(self.hmv), random.randrange(min_valid, num_width_cell*num_height_cell + 1))
        # id_valid_cell = random.sample(range(num_width_cell*num_height_cell), len(valid_nodes))
        id_valid_cell = list(range(self.hmv))
        random.shuffle(id_valid_cell)
        harmony = []
        type_trace = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0]*random.random()
            y = height_coor * self.cell_size[1] + self.cell_size[1]*random.random()
            # harmony[each_node] = [x, y]
            # type_trace[each_node] = type_
            harmony.append([x, y])
            type_trace.append(type_)
        return harmony, type_trace

    def _initialize_harmony(self, type = "default", min_valid=14, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix containing solution vectors (harmonies)
        """
        if initial_harmonies is not None:
            # assert len(initial_harmonies) == self._obj_function.get_hms(),\
            #     "Size of harmony memory and objective function is not compatible"
            # assert len(initial_harmonies[0]) == self._obj_function.get_num_parameters(),\
            #     "Number of params in harmony memory and objective function is not compatible"
            for each_harmony, type_trace in initial_harmonies:
                self._harmony_memory.append((each_harmony, self._obj_function.get_fitness(each_harmony, type_trace)[0]))
        else:
            assert type in ["default", "centroid", "cell"], "Unknown type of initialization"
            self._harmony_memory = []
            if type == "default":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony = self._random_selection(min_valid)
                    fitness, type_trace = self._obj_function.get_fitness(harmony)
                    self._harmony_memory.append((harmony, type_trace, fitness[0]))
            elif type == "centroid":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony, type_trace = self._centroid_selection(min_valid)
                    self._harmony_memory.append((harmony,type_trace, self._obj_function.get_fitness((harmony, type_trace))[0]))
            elif type == "cell":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony, type_trace = self._cell_selection(min_valid)
                    self._harmony_memory.append((harmony,type_trace, self._obj_function.get_fitness((harmony, type_trace))[0]))

    def _memory_consideration(self):
        """
            Generate new harmony from previous harmonies in harmony memory
            Apply pitch adjustment with par probability
        """
        harmony = []
        for i in range(self.hmv):
            p_hmcr = random.random()
            if p_hmcr < self.hmcr:
                id = random.choice(range(self.hms))
                [x, y] = self._harmony_memory[id][0][i]
                [x, y] = self._pitch_adjustment([x, y])
            else:
                type_ = random.choice([0, 1])
                if type_ == 0:
                    x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                    y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
                else:
                    x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                    y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()

            if x > self.upper[1][0] or x < self.lower[0][0]:
                x = -1
            if y > self.upper[1][1] or y < self.lower[0][1]:
                y = -1
            harmony.append([x, y])
        
        return harmony

    def _pitch_adjustment(self, position):
        """
            Adjustment for generating completely new harmony vectors
        """
        p_par = random.random()
        if p_par < self.par:
            bw_rate = random.uniform(-1,1)
            position[0] = self.BW*bw_rate + position[0]
            position[1] = self.BW*bw_rate + position[1]
        return position

    def _new_harmony_consideration(self, harmony):
        """
            Update harmony memory
        """
        (fitness, _), type_trace = self._obj_function.get_fitness(harmony)
        
        worst_fitness = float("+inf")
        worst_ind = -1
        best_fitness = float("-inf")
        best_ind = -1
        for ind, (_, _x, each_fitness) in enumerate(self._harmony_memory):
            if each_fitness < worst_fitness:
                worst_fitness = each_fitness
                worst_ind = ind
            
            if each_fitness > best_fitness:
                best_fitness = each_fitness
                best_ind = ind
        
        if fitness >= worst_fitness:
            self._harmony_memory[worst_ind] = (harmony, type_trace, fitness)
        
        return best_ind

    def _get_best_fitness(self):
        """
            Gest best fitness and corresponding harmony vector in harmony memory
        """
        best_fitness = float("-inf")
        best_harmony = []
        for each_harmony, type_trace, each_fitness in self._harmony_memory:
            if each_fitness > best_fitness:
                best_fitness = each_fitness
                best_harmony = each_harmony
                type_ = type_trace
        return best_harmony, type_, best_fitness

    def _get_best_coverage_ratio(self):
        best_harmony, type_trace = self._get_best_fitness()[0:2]
        (_, coverage_ratio), _ = self._obj_function.get_fitness(best_harmony)
        return coverage_ratio

    def _evaluation(self, threshold):
        coverage_ratio = self._get_best_coverage_ratio()
        best_harmony, type_, best_fitness = self._get_best_fitness()
        if coverage_ratio > self.best_coverage and coverage_ratio >= threshold:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.best_coverage = coverage_ratio
            self.logger.info("This harmony is sastified")
        elif coverage_ratio >= threshold:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.logger.info("This harmony is sastified")
        elif coverage_ratio > self.best_coverage:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.best_coverage = coverage_ratio
        return False

    def _count_sensor(self, harmony):
        count_ = 0
        for item in harmony:
            if item[0] >= 0 and item[1] >= 0:
                count_ += 1
        return count_

    def run(self, type_init="default", min_valid=14,steps=100, threshold=0.9,order=0, logger=None):
        
        print("Start run:")
        self._initialize_harmony(type_init, min_valid)

        best_ind = -1
        for i in tqdm(range(steps)):
            new_harmony = self._memory_consideration()

            new_best_ind = self._new_harmony_consideration(new_harmony)

            best_harmony, best_type, best_fitness = self._harmony_memory[best_ind]
            
            if new_best_ind != best_ind:
                best_ind = new_best_ind
                logger.info(f'Step: {str(i)} Best harmony: {str(best_harmony)} Type: {str(best_type)} Best_fitness: {str(best_fitness)}')
                logger.info('------------------------------------------------------------------------------------')

        best_harmony, best_type, best_fitness = self._harmony_memory[best_ind]
        
   
        used_node = []
        type_trace = best_type
        for ind, node in enumerate(best_harmony):
            if node[0] > 0 and node[1] > 0:
                used_node.append(node)
        coverage = self._obj_function.get_coverage_ratio(used_node, type_trace)
        no_used = len(used_node)
        no_used_convert = sum(type_trace) + (len(type_trace)-sum(type_trace))/2

        draw(used_node, type_trace, os.path.join(self.image_dir, './fig{}.png'.format(str(order))))

        # save the best for1 runs
        self.logger2.info(f'Best harmony: {str(best_harmony)}\nType: {str(type_trace)}\nBest_fitness: {str(best_fitness)}\nCoressponding coverage: {str(coverage)} \nCoressponding sensors: {str(no_used)} and {str(no_used_convert)}')
        self.logger2.info('------------------------------------------------------------------------------------')
        
        
        
        return best_fitness, coverage, no_used, no_used_convert
    
    def test(self, type_init="default", min_valid=14, steps=60000, threshold=0.9, file='logging.txt', num_run=12):
        coverage = []
        fitness = []
        used = []
        cost = []
        corr = []
        for i in range(num_run):
            logger = logging.getLogger(name='harmony{}'.format(str(i)))
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_dir, "output{}.log".format(i)))
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            ifitness, icover, iused, iusedconvert = self.run(type_init, min_valid, steps, threshold, i, logger)
            
            coverage.append(icover)
            fitness.append(ifitness)
            used.append(iused)
            corr.append(iusedconvert/25)
            cost.append(icover - iusedconvert/25)
        

        self.logger2.info('------------------------------------------------------------------------------------') 
        self.logger2.info('------------------------------------------------------------------------------------') 
        self.logger2.info(f'Coverage mean, std : {str(np.mean(coverage))} and {str(np.std(coverage))}')
        self.logger2.info(f'Used mean, std : {str(np.mean(used))} and {str(np.std(used))}')
        self.logger2.info(f'Corr Used mean, std : {str(np.mean(corr))} and {str(np.std(corr))}')
        self.logger2.info(f'Cost mean, std : {str(np.mean(cost))} and {str(np.std(cost))}')
        self.logger2.info(f'Fitness mean, std : {str(np.mean(fitness))} and {str(np.std(fitness))}')