import numpy
import time
import scipy.io
import matplotlib.pyplot as plt

class layer:
    def __init__(self, layer_size,output_size,input):
        self.layer_size=layer_size
        self.output_size=output_size
        self.sig_grad=numpy.array([[]])
        self.theta_grad=numpy.array([[]])
        self.delta=numpy.array([[]])
        k=numpy.shape(input)[0]
        b=numpy.ones((k,1))
        self.input=numpy.append(b,input,1)
        m=numpy.shape(self.input)[1]
        self.theta=numpy.random.rand(m,self.output_size)*4-2
        self.activate()
    def activate(self): 
        self.h_t=self.input.dot(self.theta)
        self.sig_grad=numpy.multiply(self.h_t,1-self.h_t)  
        self.activation=(1/(1+numpy.exp(-self.h_t)))
        
class neural_network:
    def __init__(self, input_size, output_size,x,y):
        self.input_size=input_size
        self.output_size=output_size
        self.input_layer=layer(input_size,output_size,x)
        self.layers=[self.input_layer]
        self.x=x
        self.y=y.transpose()
        self.grad=numpy.array([[]])
        self.m=numpy.shape(self.x)[0]
        self.learn_rate=0
        self.y_matrix=numpy.zeros((numpy.size(self.y),numpy.max(self.y)))
        self.y_matrix[numpy.arange(numpy.size(self.y)),self.y-1]=1
        
    def add_hidden_layer(self,inputs):   
        m=numpy.shape(self.layers[-1].theta)[0]
        self.layers[-1].theta=numpy.random.rand(m,inputs)*4-2
        self.layers[-1].activate()
        hidden_layer=layer(inputs,self.output_size,self.layers[-1].activation)
        self.layers.insert(len(self.layers),hidden_layer)
        
    def cost(self):
        a=self.layers[-1]   
        temp1=numpy.sum(numpy.multiply(-self.y_matrix,numpy.log(a.activation)))
        temp2=numpy.sum(numpy.multiply(1-self.y_matrix,numpy.log(1-a.activation)))
        rtemp1=(numpy.sum([numpy.sum(self.layers[0].theta.transpose()[:,1:]**2,0)]))
        rtemp2=(numpy.sum([numpy.sum(self.layers[1].theta.transpose()[:,1:]**2,0)]))
        reg=(self.learn_rate/(2*self.m))*(rtemp1+rtemp2)
        self.c=(1/numpy.shape(self.x)[0])*numpy.sum(temp1-temp2)
        print(self.c)
        return self.c
    
    def forward_propogation(self):
        for i in range(len(self.layers)):
            self.layers[i].activate()
            k=numpy.shape(self.layers[i].activation)[0]
            b=numpy.ones((k,1))
            if i <len(self.layers)-1:
                self.layers[i+1].input=numpy.append(b,self.layers[i].activation,1)

    def back_propogation(self):
        o=self.layers[-1]
        delta_l=o.activation-self.y_matrix
        for i in reversed(self.layers):
            i.theta_grad=(1/5000)*i.input.transpose().dot(delta_l)
            gp=numpy.multiply(i.input,1-i.input)
            delta_l=numpy.multiply(delta_l.dot(i.theta.transpose()[:,1:]),gp[:,1:])


    def learn(self):
        self.cost()
        for i in range(1000):
            self.forward_propogation()
            self.back_propogation()
            for i in self.layers:
                i.theta=i.theta-i.theta_grad
        self.cost()

 
        
reader= scipy.io.loadmat(r'C:\Users\Sami Elsaeyed\Desktop\Machine Learning\Ex-4 Octave\ex4data1.mat')
reader2= scipy.io.loadmat(r'C:\Users\Sami Elsaeyed\Desktop\Machine Learning\Ex-4 Octave\ex4weights.mat')
x=reader['X']
y=reader['y']

xnor_gate=neural_network(400,10,x,y)
xnor_gate.add_hidden_layer(25)

xnor_gate.learn()

a=numpy.argmax(xnor_gate.layers[-1].activation,1)+1
print(numpy.average((a==xnor_gate.y[0])+0))

