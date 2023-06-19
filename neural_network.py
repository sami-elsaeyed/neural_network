import numpy
import time
import scipy.io

class layer:
    def __init__(self, layer_size,output_size,input):
        self.layer_size=layer_size
        self.output_size=output_size
        self.theta_grad=numpy.array([[]])
        self.delta=numpy.array([[]])
        k=numpy.shape(input)[0]
        b=numpy.ones((k,1))
        self.input=numpy.append(b,input,1)
        m=numpy.shape(self.input)[1]
        self.theta=numpy.random.rand(m,self.output_size)
        h_t=self.input.dot(self.theta)
        self.activation=numpy.around(1/(1+numpy.exp(-h_t)),10)
        self.sig_grad=(numpy.multiply(self.activation,1-self.activation))

class neural_network:
    def __init__(self, input_size, output_size,x,y):
        # self.input_layer=layer(input_size,output_size,x)
        # self.output_layer=layer(output_size)
        # self.layers=[self.input_layer,self.output_layer]
        # self.x=x
        # self.y=y
        # self.learn_rate=3
        # self.m=numpy.shape(self.x)[0]        
        # self.y_matrix=numpy.zeros((self.y.size,numpy.max(self.y)))
        # for i in range(self.y.size):
        #     self.y_matrix[i][self.y[i][0]-1]=1
        self.input_size=input_size
        self.output_size=output_size
        self.input_layer=layer(input_size,output_size,x)
        self.layers=[self.input_layer]
        self.x=x
        self.y=y
        self.learn_rate=3
        self.m=numpy.shape(self.x)[0]        
        # self.y_matrix=numpy.zeros((self.y.size,numpy.max(self.y)))
        # for i in range(self.y.size):
        #     self.y_matrix[i][self.y[i][0]-1]=1

        
    def add_hidden_layer(self,inputs):
        b=len(self.layers)
        a=self.layers[b-1].activation
        hidden_layer=layer(inputs,self.output_size,a)       
        m=numpy.shape(a)[0]
        self.layers[b-1].theta=numpy.random.rand(m,inputs+1)
        self.layers.insert(b,hidden_layer)

    def cost(self):    
        temp1=numpy.sum(numpy.multiply(-self.y_matrix,numpy.log(self.output_layer.activation)))
        temp2=numpy.sum(numpy.multiply(1-self.y_matrix,numpy.log(1-self.output_layer.activation)))
        rtemp1=(numpy.sum([numpy.sum(self.layers[0].theta.transpose()[:,1:]**2,0)]))
        rtemp2=(numpy.sum([numpy.sum(self.layers[1].theta.transpose()[:,1:]**2,0)]))
        reg=(self.learn_rate/(2*self.m))*(rtemp1+rtemp2)
        self.c=(1/numpy.shape(self.x)[0])*numpy.sum(temp1-temp2)+reg
        print(self.c)
        return self.c
    
    def forward_propogation(self):
        for i in range(len(self.layers)-1):
            m=numpy.shape(self.layers[i].activation)[0]
            #self.theta=numpy.random.rand(m,self.layers[i+1].layer_size)
            self.layers[i].bias()
            self.layers[i].forward_propogate()
            self.layers[i+1].activation=self.layers[i].output           
        self.output_layer.activation=self.layers[-2].output

    def back_propogation(self):
        self.output_layer.delta=self.output_layer.activation-self.y_matrix
        b=len(self.layers)
        for i in range(b-2,0,-1):  
            self.layers[i-1].sigmoid_gradient()
            self.layers[i].delta=numpy.multiply(self.layers[i-1].sig_grad,self.layers[i+1].delta.dot(self.layers[i].theta.transpose()[:,1:]))
        self.grad=numpy.array([])
        for i in range(b-1):
            reg=(self.learn_rate*self.layers[i].theta)
            self.layers[i].theta_grad=(1/b)*(self.layers[i].activation.transpose().dot(self.layers[i+1].delta)+reg)
            self.grad=numpy.array([numpy.concatenate((self.grad.flatten(),self.layers[i].theta_grad.flatten()))])
        self.grad=self.grad.transpose()
    def learn(self):
        self.forward_propogation()
        self.back_propogation()
        x=self.cost()
        # for j in range(100):
        #     self.forward_propogation()
        #     self.back_propogation()
        #     for i in self.layers:
        #         i.theta=i.theta-0.001*i.theta_grad

        # y=self.cost()
        # print(x-y)
        
reader= scipy.io.loadmat(r'C:\Users\Sami Elsaeyed\Desktop\Machine Learning\Ex-4 Octave\ex4data1.mat')
reader2= scipy.io.loadmat(r'C:\Users\Sami Elsaeyed\Desktop\Machine Learning\Ex-4 Octave\ex4weights.mat')
x=reader['X']
y=reader['y']


xnor_gate=neural_network(400,10,x,y)
xnor_gate.add_hidden_layer(25)


#xnor_gate.layers[0].theta=reader2['Theta1'].transpose()
#xnor_gate.layers[1].theta=reader2['Theta2'].transpose()
#xnor_gate.learn()
#xnor_gate.forward_propogation()
#xnor_gate.cost()
#xnor_gate.back_propogation()
for i in xnor_gate.layers:
    print(numpy.shape(i.input))
    print(numpy.shape(i.theta))
    print(numpy.shape(i.activation))
