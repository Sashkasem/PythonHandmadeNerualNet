#import Layer
import Neuron
import math
import copy
import functions
class net:
    def __init__(self,topology):
        self.topology = topology
        self.numLayers = len(topology)
        self.layer = []
        self._m_layers = []
        self._m_error = 0.0
        self._m_resentAverageError = 0.0
        self._m_resentAverageSmoothingFactor = 0.0

        for layerNum in range(self.numLayers):
            layer = copy.deepcopy(self.layer)
            self._m_layers.append(layer)
            numOutputs = functions.numOut(layerNum, topology)
            for neuronNum in range(self.topology[layerNum]+1):
                self._m_layers[layerNum].append(Neuron.Neuron(numOutputs,neuronNum))
                print("Made a new neuron", layerNum, neuronNum)
            # add a bias neuron
            self._m_layers[-1][-1].setOutputVal(1.0)

    def getRecentAverageError(self):
        return self._m_resentAverageError
    def feedForward(self, inputVals):
        # заполнение входного слоя
        for i in range(len(inputVals)):
            self._m_layers[0][i].setOutputVal(inputVals[i])
        # forward propagation
        for layerNum in range(1,len(self._m_layers)):
            prevLayer = self._m_layers[layerNum-1]
            for n in range(len(self._m_layers[layerNum])-1):
                self._m_layers[layerNum][n].feedForward(prevLayer)
    def backProp(self, targetVals):
        #метод наименьших квадратов для все  сети
        outputLayer = self._m_layers[-1]
        self._m_error = 0.0
        for n in range(len(outputLayer)-1):
            delta = targetVals[n] - outputLayer[n].getOutputVal()
            self._m_error += delta*delta
        self._m_error /= len(outputLayer)-1
        self._m_error = math.sqrt(self._m_error)

        self._m_resentAverageError = (self._m_resentAverageError*self._m_resentAverageSmoothingFactor +
                                      self._m_error) / (self._m_resentAverageSmoothingFactor + 1.0)#?

        #грфдиент выходного слоя
        for n in range(len(outputLayer)-1):
            outputLayer[n].calcOutputGradietns(targetVals[n])

        # градиент скрытых и входных слоев
        for layerNum in range(len(self._m_layers)-2,0,-1):
            hiddenLayer = self._m_layers[layerNum]
            nextLayer = self._m_layers[layerNum+1]
            for n in range(len(hiddenLayer)):
                hiddenLayer[n].calcHiddenGradients(nextLayer)

        #меняет веса для всех слоев
        for layerNum in range(len(self._m_layers)-1,0,-1):
            layer = self._m_layers[layerNum]
            prevLayer = self._m_layers[layerNum-1]
            for n in range(len(layer)-1):
               layer[n].updateInputWeights(prevLayer)

    def getResults(self, resultVals):
        resultVals.clear()
        for n in range(len(self._m_layers[-1])-1):
            resultVals.append(self._m_layers[-1][n].getOutputVal())