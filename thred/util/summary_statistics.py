import math


class SummaryStat:
    def __init__(self) -> None:
        super(SummaryStat, self).__init__()
        self.__count = 0
        self.__sum = 0
        self.__avg = 0
        self.__variance = 0
        self.__min = float('inf')
        self.__max = float('-inf')

    def accept(self, value):
        self.__count += 1
        self.__sum += value
        self.__min = min(self.__min, value)
        self.__max = max(self.__max, value)

        diff = value - self.__avg
        self.__avg += diff / self.__count
        self.__variance += diff * (value - self.__avg)

    def get_average(self):
        return self.__avg

    def get_sum(self):
        return self.__sum

    def get_variance(self):
        return (self.__variance / (self.__count - 1)) if self.__count > 1 else 0

    def get_stdev(self):
        return math.sqrt(self.get_variance())

    def get_min(self):
        return self.__min

    def get_max(self):
        return self.__max


class SampledSummaryStat(SummaryStat):

    def __init__(self) -> None:
        super(SampledSummaryStat, self).__init__()
        self.__samples = []

    def accept(self, value):
        super(SampledSummaryStat, self).accept(value)
        self.__samples.append(value)

    def get_median(self):
        if not self.__samples:
            return 0.0

        sorted_samples = sorted(self.__samples)
        mid_index = len(sorted_samples) // 2

        if len(sorted_samples) % 2 == 0:
            return (sorted_samples[mid_index - 1] + sorted_samples[mid_index]) / 2
        else:
            return sorted_samples[mid_index]
