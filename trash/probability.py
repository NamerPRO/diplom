import numpy as np


class LogProbability:

    @staticmethod
    def zero():
        return LogProbability(0)

    @staticmethod
    def one():
        return LogProbability(1)

    @staticmethod
    def half():
        return LogProbability(0.5)

    def __init__(self, probability, log_prob=False):
        self.__eps = 1e-5
        if log_prob:
            # if probability > self.__eps:
            #     raise ValueError("Logarithmic probability cannot be greater than 0.")
            self.__probability = np.float64(probability)
        else:
            # if probability < -self.__eps or probability - 1 > self.__eps:
            #     raise ValueError("Probability must be between 0 and 1.")
            self.__probability = self.__plog(probability)

    def __add__(self, other):
        if isinstance(other, LogProbability):
            return LogProbability(self.__probability + self.__plog(1 + np.exp(other.as_float() - self.__probability)), True)
        # self.__check(other)
        return LogProbability(self.__probability + self.__plog(1 + np.exp(other - self.__probability)), True)

    def __iadd__(self, other):
        if isinstance(other, LogProbability):
            self.__probability += self.__plog(1 + np.exp(other.as_float() - self.__probability))
        else:
            # self.__check(other)
            self.__probability += self.__plog(1 + np.exp(other - self.__probability))
        # if self.__probability > self.__eps:
        #     raise ValueError("Logarithmic probability cannot be greater than 0.")
        return self

    def __mul__(self, other):
        if isinstance(other, LogProbability):
            return LogProbability(self.__probability + other.as_float(), True)
        # self.__check(other)
        return LogProbability(self.__probability + other, True)

    def __imul__(self, other):
        if isinstance(other, LogProbability):
            self.__probability += other.as_float()
        else:
            # self.__check(other)
            self.__probability += other
        # if self.__probability > self.__eps:
        #     raise ValueError("Logarithmic probability cannot be greater than 0.")
        return self

    def __truediv__(self, other):
        if isinstance(other, LogProbability):
            return LogProbability(self.__probability - other.as_float(), True)
        # self.__check(other)
        return LogProbability(self.__probability - other, True)

    def __itruediv__(self, other):
        if isinstance(other, LogProbability):
            self.__probability -= other.as_float()
        else:
            # self.__check(other)
            self.__probability -= other
        # if self.__probability > self.__eps:
        #     raise ValueError("Logarithmic probability cannot be greater than 0.")
        return self

    def __sub__(self, other):
        if isinstance(other, LogProbability):
            return LogProbability(self.__probability + self.__plog(1 - np.exp(other.as_float() - self.__probability)), True)
        # self.__check(other)
        return LogProbability(self.__probability + self.__plog(1 - np.exp(other - self.__probability)), True)

    def __isub__(self, other):
        if isinstance(other, LogProbability):
            self.__probability -= self.__plog(1 - np.exp(other.as_float() - self.__probability))
        else:
            # self.__check(other)
            self.__probability -= self.__plog(1 - np.exp(other - self.__probability))
        # if self.__probability > self.__eps:
        #     raise ValueError("Logarithmic probability cannot be greater than 0.")
        return self

    def sqrt(self):
        self.__probability /= 2


    def __eq__(self, other):
        if isinstance(other, LogProbability):
            # == case to compare when self.__probability == other.as_float() == -np.inf
            return self.__probability == other.as_float() or np.abs(self.__probability - other.as_float()) < self.__eps
        # self.__check(other)
        return self.__probability == other or np.abs(self.__probability - other) < self.__eps

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, LogProbability):
            return other.as_float() - self.__probability > self.__eps
        else:
            # self.__check(other)
            return other - self.__probability > self.__eps

    def __gt__(self, other):
        if isinstance(other, LogProbability):
            return self.__probability - other.as_float() > self.__eps
        else:
            # self.__check(other)
            return self.__probability - other > self.__eps

    def __le__(self, other):
        return not self.__gt__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __str__(self):
        return str(self.__probability)

    def to_probability(self):
        return np.exp(self.__probability)

    @staticmethod
    def sum(log_probabilities):
        if not all(isinstance(i, LogProbability) for i in log_probabilities):
            raise ValueError("Expected vector of logarithmic probabilities with each element of type LogProbability.")
        total = LogProbability.zero()
        for log_probability in log_probabilities:
            if total == LogProbability.zero():
                total = LogProbability(log_probability.as_float(), True)
            else:
                total += log_probability
        return total

    def part(self, x):
        return LogProbability(self.__probability * x, True)

    def as_float(self):
        return self.__probability

    def is_close(self, x):
        return np.isclose(self.__probability, x.as_float())

    def __plog(self, x):
        if x < -self.__eps:
            raise ValueError("Logarithmic probability of the performed operation does not exist. This is because result is negative in classical probabilities term.")
        return np.log(x) if np.abs(x) >= self.__eps else -np.inf

    @staticmethod
    def max(x):
        mmax = x[0]
        for i in range(1, len(x)):
            if mmax < x[i]:
                mmax = x[i]
        return mmax

    @staticmethod
    def norm(x):
        mmax, normm = LogProbability.max(x), 0
        for i in range(len(x)):
            normm += np.exp(2 * (x[i].as_float() - mmax.as_float()))
        return LogProbability(np.exp(mmax.as_float()) * np.sqrt(normm))

if __name__ == "__main__":
    x = LogProbability(0.2)
    y = LogProbability(0.3)

    print(LogProbability.norm([LogProbability(0.0000000003), LogProbability(0.0000000033), LogProbability(0.0000000033)]))
    print(np.linalg.norm([0.000033, 0.000033, 0.000033]))
    # print(x - y)

    w = LogProbability(0.2)
    z = LogProbability(0.3)

    print(w + -z)