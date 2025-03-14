import numpy as np
import random
import heapq
from collections.abc import Iterable



class BaseReservoir:
    def __init__(self, size):
        self.size = size
        self.sample_num = 0
        self.data = []
    
    def get_replacement_probability(self):
        raise NotImplementedError("Subclasses must implement get_replacement_probability().")
    
    def add(self, vals):
        if not isinstance(vals, Iterable):
            vals = [vals]

        for val in vals:
            self.sample_num += 1

            if self.sample_num <= self.size:
                self.data.append(val)
            else:
                p_replace = self.get_replacement_probability()
                if np.random.random() < p_replace:
                    idx_to_replace = int(self.size * np.random.random())
                    self.data[idx_to_replace] = val
    
    def __len__(self):
        return self.sample_num

    def get_data(self):
        return random.sample(self.data, len(self.data))

    def get_samples(self, k, with_replacement=True):
        if not self.data:
            raise ValueError("Cannot sample from an empty dataset")

        if with_replacement:
            return random.choices(self.data, k=k)
        
        if k > len(self.data):
            raise ValueError(f"k={k} is larger than dataset size={len(self.data)} when sampling without replacement.")

        return random.sample(self.data, k=k)


class RegretReservoir(BaseReservoir):
    def get_replacement_probability(self):
        return self.size / self.sample_num


class PolicyReservoir(BaseReservoir):
    def get_replacement_probability(self):
        return 1  # np.sqrt(self.size / self.sample_num)


class LinearReservoir(BaseReservoir):
    def __init__(self, size):
        super().__init__(size)
        self.heap = []  # Min-heap storing (weight, index) pairs

    def add(self, vals):
        if not isinstance(vals, Iterable):
            vals = [vals]
        
        for val in vals:
            self.sample_num += 1
            weight = np.random.uniform() ** (100000 / self.sample_num) #I assume increasing this will make numerical stability better
            
            if self.sample_num <= self.size:
                idx = len(self.data)
                self.data.append(val)
                heapq.heappush(self.heap, (weight, idx))
            else:
                min_weight, min_idx = self.heap[0]
                if weight > min_weight:
                    heapq.heappop(self.heap)  # Remove smallest weight
                    heapq.heappush(self.heap, (weight, min_idx))
                    self.data[min_idx] = val  # Replace element in data
    
    def get_data(self):
        return [self.data[idx] for _, idx in sorted(self.heap)]
