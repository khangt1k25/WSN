import math
from itertools import product
import random
import numpy as np


class ObjectiveFunction():
    def __init__(self, hmv, hms, targets, types=2, radius=[0,0], alpha1=1, alpha2=0, beta1=1, beta2=0.5,\
                threshold=0.9, w=50, h=50, cell_h=10, cell_w=10):
        """
            :param hmv: harmony vector size
            :param hms: harmony memory size
            :param targets: position of target points
            :param types: number of different node types
            :param radius: radius for each node type
            :param alpha1, alpha2, beta1, beta2: parameter for calculating Pov
            :param threshold: threshold for Pov
            :param h, w: height and width of AoI
            :param cell_h, cell_w: height and width of cell
        """
        self.hmv = hmv
        self.hms = hms
        self.targets = targets
        # print(self.targets)
        self.radius = radius
        self.ue = []
        for r in radius:
            self.ue.append(r / 2)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.type_sensor = range(types)
        self.w = w
        self.h = h
    
        self.cell_h = cell_h
        self.cell_w = cell_w
  
        self.cell_r = math.sqrt((cell_h/2)**2 + (cell_w/2)**2)
        self.no_cell = (self.w // self.cell_w) * (self.h // self.cell_h)
        self.min_noS = self.w * self.h // ((max(self.radius)**2)*9)
        self.max_noS = self.w * self.h // ((min(self.radius)**2))
        self.max_diagonal = max([self._distance([self.w, self.h], [self.radius[i] + self.ue[i], self.radius[i] + self.ue[i]]) for i in range(len(self.radius))])
    


    def get_hms(self):
        return self.hms
    def get_radius(self):
        return self.radius

    def _senscost(self, node_list):
        x = (len(node_list) - self.min_noS) / (self.max_noS - self.min_noS)
        return 1 / (10*x + 1)


    def get_coverage_ratio(self, node_list, type_assignment):
        return self._coverage_ratio(node_list, type_assignment)[0]
    
    def _coverage_ratio(self, node_list, type_assignment):
        """
            Return coverage_ratio and list of covered target
        """
        target_corvered = []
        for target in self.targets:
            Pov = 1
            count = 0
            count_ = 0
            for index, sensor in enumerate(node_list):
                p = self._psm(sensor, target, type=type_assignment[index])
                if p == 0:
                    continue
                count_ += 1
                if self._distance(sensor, target) <= self.radius[type_assignment[index]]:
                  count += 1
                Pov *= p
            
            Pov = 1 - Pov
            if count == 1 and count_ == 1 and (1-Pov>=self.threshold):
                target_corvered.append(target)
            elif Pov >= self.threshold and count_ > 1:
                target_corvered.append(target)
        

        return len(target_corvered) / self.no_cell, target_corvered

    def _md(self, node_list, type_assignment):
        min_dist_sensor = float('+inf')
        for ia, a in enumerate(node_list):
            for ib, b in enumerate(node_list):
                if a != b:
                    min_dist_sensor = min(min_dist_sensor, self._distance(a, b) * ((self.radius[type_assignment[ia]]) * (self.radius[type_assignment[ib]])))
        if min_dist_sensor == float('+inf'):
            min_dist_sensor = 0.0
        return min_dist_sensor / (self.max_diagonal)

    # Keep overlap sensor
    def _regularization1(self, node_list, type_assignment):
        no_interception = 0
        for ia, a in enumerate(node_list):
            for ib, b in enumerate(node_list):
                if a!=b:
                    if(self._distance(a, b) < (self.radius[type_assignment[ia]] + self.radius[type_assignment[ib]])):
                            no_interception += 1
        no_interception = no_interception/2
        n = len(node_list)
        no_interception = no_interception/(n*(n-1)/2)
        
        return 1-no_interception
    
    ## Keep every cell has 1 sensor (HARD)
    def _regularization2(self, node_list, type_assignment):
        node_in_cells = []
        for target in self.targets:
            s = 0
            for node in node_list:
                if self._distance(target, node) <= self.cell_r:
                    s+=1
            if s==1:
                node_in_cells.append(s)
        
        return len(node_in_cells)
    
    ##  Keep target inside one Sensor range
    def _regularization4(self, node_list, type_assignment):
        node_per_cells = []
        for target in self.targets:
            s = 0
            for inode, node in enumerate(node_list):
                if self._distance(target, node) <= type_assignment[inode]:
                    s+=1
            if s==1:
                node_per_cells.append(s)
        
        return len(node_per_cells)

    ## Keep no sensor but count type assigment
    def _regularization3(self, node_list, type_assignment):
        # no_used = len(node_list)
        no_used_convert = sum(type_assignment) + (len(type_assignment)-sum(type_assignment))/2
        return 1- no_used_convert/25

    def get_fitness(self, harmony):

        used = []
        
        for id, sensor in enumerate(harmony):
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            else:
                used.append(sensor)
        
        if len(used) < self.min_noS:
                return (float('-inf'), 0), []

        type_traces = [[random.choice([0, 1]) for j in range(len(used))] for i in range(1)]
        
        best_fitness = float('-inf')
        best_coverage_ratio = 0
        best_trace = None
            
        for type_trace in type_traces:
        
            coverage_ratio, _ = self._coverage_ratio(used, type_trace)
            
            # fitness =  (coverage_ratio)  * self._senscost(used)* self._md(used, type_trace)
            fitness =  (coverage_ratio)  * self._senscost(used)
            #     self._regularization3(used, type_trace)
            # fitness =  (coverage_ratio)  * self._senscost(used) * self._regularization4(used, type_trace) * self._regularization3(used, type_trace)*\
            #      self._regularization1(used, type_trace) * self._regularization2(used, type_trace) 

                # *\ #* self._regularization1(used, type_trace) #*\

            if fitness > best_fitness:
                best_fitness = fitness
                best_coverage_ratio = coverage_ratio
                best_trace = type_trace

        return (best_fitness, best_coverage_ratio), best_trace

    def _distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def _psm(self,x, y, type):
        distance = self._distance(x, y)
        
        if distance < self.radius[type] - self.ue[type]:
            return 1
        elif distance > self.radius[type] + self.ue[type]:
            return 0
        else:
            lambda1 = self.ue[type] - self.radius[type] + distance
            lambda2 = self.ue[type] + self.radius[type] - distance
            lambda1 = math.pow(lambda1, self.beta1)
            lambda2 = math.pow(lambda2, self.beta2)
            return math.exp(-(self.alpha1*lambda1/lambda2 + self.alpha2))
