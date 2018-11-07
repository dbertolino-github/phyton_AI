import math
import random
import string

class NN:
    
  def __init__(self, NI, NH, NO):
       
    # number of nodes in layers
    self.nodesInputNumber = NI + 1 # +1 for bias
    self.nodesHiddenNumber = NH
    self.nodesOutputNumber = NO
    
    # input_nodesitialize node-activations
    self.activationsInput, self.activationsHidden, self.activationsOutput = [],[],[]
    self.activationsInput = [1.0]*self.nodesInputNumber
    self.activationsHidden = [1.0]*self.nodesHiddenNumber
    self.activationsOutput = [1.0]*self.nodesOutputNumber

    # create node weight matrices
    self.weightsInput = makeMatrix (self.nodesInputNumber, self.nodesHiddenNumber)
    self.weightsOutput = makeMatrix (self.nodesHiddenNumber, self.nodesOutputNumber)
    
    # initialize node weights to random vals
    randomizeMatrix ( self.weightsInput, -0.2, 0.2 )
    randomizeMatrix ( self.weightsOutput, -2.0, 2.0 )
    
    # create last change in weights matrices for momentum
    self.lastChangesInput = makeMatrix (self.nodesInputNumber, self.nodesHiddenNumber)
    self.lastChangesOutput = makeMatrix (self.nodesHiddenNumber, self.nodesOutputNumber)
    
    
  def runNN (self, inputs):
     
    #CHECKING INPUT
    if len(inputs) != self.nodesInputNumber-1:
      print('incorrect number of inputs')
    
    #ACTIVATING INPUT
    for i in range(self.nodesInputNumber-1):
      self.activationsInput[i] = inputs[i]
     
    #INPUT --> HIDDEN
    for j in range(self.nodesHiddenNumber):
      sum = 0.0
      for i in range(self.nodesInputNumber):
        sum +=( self.activationsInput[i] * self.weightsInput[i][j] )
      self.activationsHidden[j] = sigmoid (sum)
    
    #HIDDEN --> OUTPUT
    for k in range(self.nodesOutputNumber):
      sum = 0.0
      for j in range(self.nodesHiddenNumber):        
        sum +=( self.activationsHidden[j] * self.weightsOutput[j][k] )
      self.activationsOutput[k] = sigmoid (sum)
      
    return self.activationsOutput


  def backPropagate (self, targets, N, M):
    # http://www.youtube.com/watch?v=aVId8KMsdUU&feature=BFa&list=LLldMCkmXl4j9_v0HeKdNcRA
    # we want to find the instantaneous rate of change of ( error with respect to weight from node j to node k)
    # output_delta is defined as an attribute of each ouput node. It is not the final rate we need.
    # To get the final rate we must multiply the delta by the activation of the hidden layer node in question.
    # This multiplication is done according to the chain rule as we are taking the derivative of the activation function
    # of the ouput node.
    # dE/dw[j][k] = (t[k] - activationsOutput[k]) * s'( SUM( w[j][k]*activationsHidden[j] ) ) * activationsHidden[j]
    
    #CALCULATE OUTPUT DELTAS
    output_deltas = [0.0] * self.nodesOutputNumber
    for k in range(self.nodesOutputNumber):
      error = targets[k] - self.activationsOutput[k]
      output_deltas[k] =  error * dsigmoid(self.activationsOutput[k]) 
   
    #UPDATE OUTPUT WEIGHTS
    for j in range(self.nodesHiddenNumber):
      for k in range(self.nodesOutputNumber):
        # output_deltas[k] * self.activationsHidden[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.activationsHidden[j]
        self.weightsOutput[j][k] += N*change + M*self.lastChangesOutput[j][k]
        self.lastChangesOutput[j][k] = change

    #CALCULATE HIDDEN DELTAS
    hidden_deltas = [0.0] * self.nodesHiddenNumber
    for j in range(self.nodesHiddenNumber):
      error = 0.0
      for k in range(self.nodesOutputNumber):
        error += output_deltas[k] * self.weightsOutput[j][k]
      hidden_deltas[j] = error * dsigmoid(self.activationsHidden[j])
    
    #UPDATE INTPUT WEIGHTS
    for i in range (self.nodesInputNumber):
      for j in range (self.nodesHiddenNumber):
        change = hidden_deltas[j] * self.activationsInput[i]
        #print 'activation',self.activationsInput[i],'synapse',i,j,'change',change
        self.weightsInput[i][j] += N*change + M*self.lastChangesInput[i][j]
        self.lastChangesInput[i][j] = change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * (targets[k]-self.activationsOutput[k])**2
    return error
        
        
  def weights(self):
    print('Input weights:')
    for i in range(self.nodesInputNumber):
      print(self.weightsInput[i])
    print('')
    print('Output weights:')
    for j in range(self.nodesHiddenNumber):
      print(self.weightsOutput[j])
    print ('')
    
  
  def test(self, patterns):
    for p in patterns:
      inputs = p[0]
      print('Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1])
      
  #N and M are two coefficients which role learning steps dimension
  #in terms of changes' weigth during backpropagation
  def train (self, patterns, max_iterations = 1000, N=0.5, M=0.1):
    for i in range(max_iterations):
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.runNN(inputs)
        error = self.backPropagate(targets, N, M)
      if i % 50 == 0:
        print('Combined error', error)
    self.test(patterns)
    

def sigmoid (x):
  return math.tanh(x)

  
# the derivative of the sigmoid function in terms of output
# proof here: 
# http://www.math10.com/en/algebra/hyperbolic-functions/hyperbolic-functions.html
def dsigmoid (y):
  return 1 - y**2


def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m

  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)


def main ():
  pat = [
      [[0,0], [1]],
      [[0,1], [1]],
      [[1,0], [1]],
      [[1,1], [0]]
  ]
  myNN = NN ( 2, 2, 1)
  myNN.train(pat)
  

if __name__ == "__main__":
    main()
