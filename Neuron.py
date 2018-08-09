from collections import namedtuple
import copy
import math
import random
import Connection
#Connection = namedtuple("Connection",['weight','deltaWeight'])

class Neuron:
    def __init__(self,numOutputs, myIndex):
        self._m_myIndex = myIndex
        self.numOutputs = numOutputs
        self._m_outputVal = 0.0
        self.m_outputWeights = []
        self.m_gradient = 0.0
        self.eta = 0.10
        self.alpha = 0.1
        for c in range(self.numOutputs):
            self.m_outputWeights.append(Connection.connection())
            self.m_outputWeights[c].weight = random.random()
            print("Weights[c] = ", self.m_outputWeights[0].weight)
            print("Weights[c] = ",self.m_outputWeights[c].weight, " c = ",c)
            self.m_outputWeights[c].deltaWeight = 0.0
    def updateInputWeights(self,prevLayer):
        for n in range(len(prevLayer)):
            neuron = prevLayer[n]
            #print("neuron.m_outputWeights[self._m_myIndex].weight = ",neuron.m_outputWeights[self._m_myIndex].weight)
            oldDeltaWeight = neuron.m_outputWeights[self._m_myIndex].deltaWeight
            newDeltaWeight = (self.eta*neuron.getOutputVal()*self.m_gradient +
                              self.alpha*oldDeltaWeight)
            #print("problem: ",neuron.getOutputVal())
            neuron.m_outputWeights[self._m_myIndex].deltaWeight = newDeltaWeight
            neuron.m_outputWeights[self._m_myIndex].weight += newDeltaWeight
    def _sumDOW(self, nextlayer):
        sum = 0.0
        for n in range(len(nextlayer)-1):
            sum += self.m_outputWeights[n].weight * nextlayer[n].m_gradient
        return sum

    def _transferFunction(self, x):
        #print("x = ",x)
        #print("tanh = ", math.tanh(x))
        return math.tanh(x)
    def _transferFunctionDerivetive(self, x):
        return 1.0 - x*x
    def calcOutputGradietns(self,targetVal):
        delta = targetVal - self._m_outputVal
        self.m_gradient = delta * self._transferFunctionDerivetive(self._m_outputVal)
    def calcHiddenGradients(self,nextLayer):
        dow = self._sumDOW(nextLayer)
        self.m_gradient = dow*self._transferFunctionDerivetive(self._m_outputVal)


    def feedForward(self,prevLayer):
        sum = 0.0
        # суммирует предыдущий слой, подает на вход, включая байес-нейроны
        for n in range(len(prevLayer)):
            sum += prevLayer[n].getOutputVal()*prevLayer[n].m_outputWeights[self._m_myIndex].weight

        self._m_outputVal = self._transferFunction(sum/20.0)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print("in feedForward = ", self._m_outputVal)


    def setOutputVal(self, val):
        self._m_outputVal = val
        #print("in setOutputVal = ", self._m_outputVal)
    def getOutputVal(self):
        #print("in getOutputVal = ",self._m_outputVal)
        return self._m_outputVal
