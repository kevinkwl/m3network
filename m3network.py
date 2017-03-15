from svmutil import *
import numpy as np

class Config:
    trainpath = "./data/train.txt"
    testpath = "./data/test.txt"
    kernel_type = 2  # RBF: 2, Linear = 0
    gamma = 256
    rbf_cost = 128
    linear_cost = 1024

    minblock = 50

    predict_options = "-q"  # quiet

    @property
    def options(self):
        if self.kernel_type == 0:
            options = "-s 0 -t {} -c {} -q".format(self.kernel_type, self.linear_cost)
        else:
            options = "-s 0 -t {} -c {} -g {} -q".format(self.kernel_type, self.rbf_cost, self.gamma)
        return options

class M3Network:

    def __init__(self, config=Config()):
        self.config = config
        self.y, self.x = svm_read_problem(self.config.trainpath)
        self.label_set = list(set(self.y))
        self.n_classes = len(self.label_set)
        self.models = [[[] for i in range(self.n_classes)] for i in range(self.n_classes)]
        self.n_entries = [0] * self.n_classes
        self.n_blocks = []
        self.blocks = [0] * self.n_classes

        self.testy, self.testx = svm_read_problem(self.config.testpath)

    def decompose(self):
        xy = zip(self.x, self.y)
        # group x with same class
        # xgroup[i] has label self.label_set[i]
        xgroup = [[xi for (xi, yi) in xy if yi == label] for label in self.label_set]
        self.n_entries = [len(g) for g in xgroup]

        minblock = self.config.minblock
        self.n_blocks = [ self.n_entries[i] / minblock or 1 for i in range(self.n_classes)]

        # label doesn't matter after grouping
        for i in range(self.n_classes):
            self.blocks[i] = [[xgroup[i][idx % self.n_entries[i]] for idx in range(j*minblock, (j+1)*minblock)]
                              for j in range(self.n_blocks[i])]



    def train(self):
        options = self.config.options
        minblock = self.config.minblock

        # train Mij
        for i in range(self.n_classes):
            for j in range(i+1, self.n_classes):
                # train Mij_ni_nj
                for Ni in range(self.n_blocks[i]):
                    self.models[i][j].append([])
                    for Nj in range(self.n_blocks[j]):
                        train_x = self.blocks[i][Ni] + self.blocks[j][Nj]

                        # 1 vs 0
                        train_y = [1] * minblock + [0] * minblock

                        model = svm_train(train_y, train_x, options)
                        self.models[i][j][Ni].append(model)

    def predict(self, x=None, quiet=False):
        if not x:
            x = self.testx
        options = self.config.predict_options
        pseudo_y = [0] * len(x)
        # through Mi_
        outMi = []  # output from each Mi module
        for mi in range(self.n_classes):
            outMij = []  # output from each Mij module
            for j in range(self.n_classes):
                outMij_ni = []
                if mi < j:
                    for ni in range(self.n_blocks[mi]):
                        outMij_ni_nj = []
                        for nj in range(self.n_blocks[j]):
                            _, _, pred_values = svm_predict(pseudo_y, x, self.models[mi][j][ni][nj], options)

                            # pred_values: [[v], [v], [v], [v] ... ]
                            values = [v[0] for v in pred_values]
                            outMij_ni_nj.append(values)

                        # each column of outMij_ni_nj has same +1 elements
                        # min
                        value_ni = np.min(outMij_ni_nj, axis=0).tolist()
                        outMij_ni.append(value_ni)
                    value_Mij = np.max(outMij_ni, axis=0).tolist()
                    outMij.append(value_Mij)
                # Mij = - Mji inv
                elif mi > j:
                    for nj in range(self.n_blocks[j]):
                        outMij_nj_ni = []
                        for ni in range(self.n_blocks[mi]):
                            _, _, pred_values = svm_predict(pseudo_y, x, self.models[j][mi][nj][ni], options)
                            values = [v[0] for v in pred_values]
                            outMij_nj_ni.append(values)

                        # each column of outMij_ni_nj has same +1 elements
                        # min
                        value_nj = np.min(outMij_nj_ni, axis=0).tolist()
                        outMij_ni.append(value_nj)

                    # invert by projecting to the negative class
                    value_Mij = (- np.max(outMij_ni, axis=0)).tolist()
                    outMij.append(value_Mij)

            # select min from all Mi_
            value_Mi = np.min(outMij, axis=0).tolist()
            outMi.append(value_Mi)
            if not quiet:
                print "finish on class", mi
        # the final result is the largest among all
        return np.argmax(outMi, axis=0).tolist()

    def acc(self, predicted, truey=None):
        if not truey:
            truey = self.testy
        assert len(predicted) == len(truey)
        hit = 0
        for i, idx in enumerate(predicted):
            if self.label_set[idx] == truey[i]:
                hit += 1
        accuracy = hit * 100.0 / len(truey)
        print "Accuracy = {}% ({}/{})".format(accuracy, hit, len(truey))
        return accuracy


def main():
    config = Config()
    m3n = M3Network(config)
    m3n.decompose()
    m3n.train()
    acc = m3n.acc(m3n.predict(quiet=True))


if __name__ == "__main__":
    main()