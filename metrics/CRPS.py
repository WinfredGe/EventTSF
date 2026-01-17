import numpy as np


class CRPS:

    def __init__(self, ensemble_members, observation, adjusted_ensemble_size=200):

        self.fc = np.sort(ensemble_members)
        self.ob = observation
        self.__m = len(self.fc)
        self.M = int(adjusted_ensemble_size)
        self.__cdf_fc = None
        self.__cdf_ob = None
        self.__delta_fc = None
        self.crps = None
        self.fcrps = None
        self.acrps = None

    def __str__(self):
        "Kindly refer to the __doc__ method for documentation. i.e. print(CRPS.__doc__)."

    def compute(self):

        if (self.ob is not np.nan) and (not np.isnan(self.fc).any()):
            if self.ob < self.fc[0]:
                self.__cdf_fc = np.linspace(0, (self.__m - 1) / self.__m, self.__m)
                self.__cdf_ob = np.ones(self.__m)
                all_mem = np.array([self.ob] + list(self.fc), dtype=object)
                self.__delta_fc = np.array([all_mem[n + 1] - all_mem[n] for n in range(len(all_mem) - 1)], dtype=object)

            elif self.ob > self.fc[-1]:
                self.__cdf_fc = np.linspace(1 / self.__m, 1, self.__m)
                self.__cdf_ob = np.zeros(self.__m)
                all_mem = np.array(list(self.fc) + [self.ob], dtype=object)
                self.__delta_fc = np.array([all_mem[n + 1] - all_mem[n] for n in range(len(all_mem) - 1)], dtype=object)

            elif self.ob in self.fc:
                self.__cdf_fc = np.linspace(1 / self.__m, 1, self.__m)
                self.__cdf_ob = (self.fc >= self.ob)
                all_mem = self.fc
                self.__delta_fc = np.array(
                    [all_mem[n + 1] - all_mem[n] for n in range(len(all_mem) - 1)] + list(np.zeros(1)), dtype=object)

            else:
                cdf_fc = []
                cdf_ob = []
                delta_fc = []
                for f in range(len(self.fc) - 1):
                    if (self.fc[f] < self.ob) and (self.fc[f + 1] < self.ob):
                        cdf_fc.append((f + 1) * 1 / self.__m)
                        cdf_ob.append(0)
                        delta_fc.append(self.fc[f + 1] - self.fc[f])
                    elif (self.fc[f] < self.ob) and (self.fc[f + 1] > self.ob):
                        cdf_fc.append((f + 1) * 1 / self.__m)
                        cdf_fc.append((f + 1) * 1 / self.__m)
                        cdf_ob.append(0)
                        cdf_ob.append(1)
                        delta_fc.append(self.ob - self.fc[f])
                        delta_fc.append(self.fc[f + 1] - self.ob)
                    else:
                        cdf_fc.append((f + 1) * 1 / self.__m)
                        cdf_ob.append(1)
                        delta_fc.append(self.fc[f + 1] - self.fc[f])
                self.__cdf_fc = np.array(cdf_fc)
                self.__cdf_ob = np.array(cdf_ob)
                self.__delta_fc = np.array(delta_fc)

            self.crps = np.sum(np.array((self.__cdf_fc - self.__cdf_ob) ** 2) * self.__delta_fc)
            if self.__m == 1:
                self.fcrps = self.acrps = 'Not defined'
            else:
                self.fcrps = self.crps - np.sum(
                    np.array(((self.__cdf_fc * (1 - self.__cdf_fc)) / (self.__m - 1)) * self.__delta_fc))
                self.acrps = self.crps - np.sum(np.array((((1 - (self.__m / self.M)) * self.__cdf_fc * (
                            1 - self.__cdf_fc)) / (self.__m - 1)) * self.__delta_fc))
            return self.crps, self.fcrps, self.acrps
        else:
            return np.nan, np.nan, np.nan

