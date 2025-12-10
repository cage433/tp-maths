import numpy as np

from scipy.special import ndtri


class BrownianBridge:
    def __init__(self, times):
        self.times = times
        n_times = len(times)
        self.n_times = n_times

        step_map = np.zeros((n_times,), int)
        self.left_index = np.zeros((n_times,), int)
        self.bridge_index = np.zeros((n_times,), int)
        self.right_index = np.zeros((n_times,), int)

        self.left_weight = np.zeros((n_times,), float)
        self.right_weight = np.zeros((n_times,), float)
        self.stddev = np.zeros((n_times,), float)

        step_map[n_times - 1] = 1
        self.bridge_index[0] = n_times - 1
        self.stddev[0] = np.sqrt(times[n_times - 1])
        self.left_weight[0] = 0.0
        self.right_weight[0] = 0.0

        j = 0
        k = 0
        l = 0

        def index_where(pred, l, i_start):
            for i, v in enumerate(l[i_start:]):
                if pred(v):
                    return i + i_start
            return -1

        def is_zero(n):
            return n == 0

        def is_non_zero(n):
            return n != 0

        for i in range(1, n_times):
            j = index_where(is_zero, step_map, j)
            k = j
            k = index_where(is_non_zero, step_map, k)
            l = j + ((k - 1 - j) >> 1)

            step_map[l] = i
            self.bridge_index[i] = l
            self.left_index[i] = j
            self.right_index[i] = k

            if j > 0:

                if times[k] == times[j - 1]:
                    self.left_weight[i] = 1.0
                else:
                    self.left_weight[i] = (times[k] - times[l]) / (times[k] - times[j - 1])
                self.stddev[i] = np.sqrt((times[l] - times[j - 1]) * self.left_weight[i])
            else:
                if times[k] == 0.0:
                    self.left_weight[i] = 1.0
                else:
                    self.left_weight[i] = (times[k] - times[l]) / times[k]
                self.stddev[i] = np.sqrt(times[l] * self.left_weight[i])

            self.right_weight[i] = 1.0 - self.left_weight[i]

            j = k + 1
            if j >= n_times:
                j = 0

    def generate(self, uniform_sample):

        n_paths, n_times = uniform_sample.shape

        if n_times != len(self.times):
            raise (Exception(f"uniform sample has invalid length"))

        normal_sample = ndtri(uniform_sample)
        path = np.zeros(uniform_sample.shape, float)
        path[:, self.n_times - 1] = self.stddev[0] * normal_sample[:, 0]

        i_time = 1

        while i_time < self.n_times:
            j = self.left_index[i_time]
            k = self.right_index[i_time]
            l = self.bridge_index[i_time]

            if j > 0:
                path[:, l] = self.left_weight[i_time] * path[:, j - 1] + \
                          self.right_weight[i_time] * path[:, k] + \
                          self.stddev[i_time] * normal_sample[:, i_time]
            else:
                path[:, l] = self.right_weight[i_time] * path[:, k] + self.stddev[i_time] * normal_sample[:, i_time]

            i_time += 1

        return path

    @staticmethod
    def from_uniforms(uniforms, n_variables, times):
        n_paths = uniforms.shape[0]
        n_times = len(times)

        if uniforms.shape != (n_paths, n_variables * n_times):
            raise Exception("Shape for brownians doesn't match")

        brownians = np.zeros((n_variables, n_paths, n_times))
        bridge = BrownianBridge(times)

        var_range = range(n_variables)
        for i_var in var_range:
            brownians[i_var, :, :] = bridge.generate(uniforms[:, i_var::n_variables])

        return brownians
