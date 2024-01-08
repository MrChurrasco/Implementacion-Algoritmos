import math

def sign (a):
    if a<0:
        return -1
    elif a>0:
        return 1
    else:
        return 0

def R (z, s):
    r = max(0, 1-z)
    r = min (1-s, r)
    return r

#Funciones costo
#Square Loss
def square (f, y):
    V=(y-f)**2
    return V

#Hinge Loss
def hinge (f,y):
    V=max(0,1-y*f)
    return V

#Smoothed Hinge Loss
def smoothinge (f,y):
    a=y*f
    if a>=1:
        V=0
    elif a>0 and a<=1:
        V=(1-a)**2
        V=V/2
    elif a<=0:
        V=0.5-a
    return V

#Modified Square Loss
def modsquare (f,y):
    V=max(1-y*f,0)
    V=V**2
    return V

#Exponential Loss
def exp (f,y):
    V=math.exp(-y*f)
    return V

#Log loss
def log (f,y):
    V=math.log(1+math.exp(-y*f))
    return V

#Based on Sigmoid Loss
def sigmoid (f,y,gamma,theta,l):
    a=y*f
    if a>=-1 and a<=0:
        V=(1.2-gamma)-gamma*a
    elif a>0 and a<=theta:
        b=(1.2-2*gamma)*a
        b=b/theta
        V=(1.2-l)-b
    elif a>theta and a<=1:
        b=gamma/(1-theta)
        c=l*a
        c=c/(1-theta)
        V=b-c
    return V

#Phi-Learning
def phi (f,y):
    a=y*f
    if a>=0 and a<=1:
        V=1-a
    else:
        V=1-sign(a)
    return V

#Ramp Loss
def ramp (f, y, s, c):
    V = R(f*y, s)+ R(-f*y, s) + c
    return V

#Smooth non-convex loss
def smoothnonconvex (f,y,l):
    V = 1 - math.tanh(l*y*f)
    return V

#2-layer Neural New-works
def layer (f,y):
    V = 1 + math.exp(-y*f)
    V=1/V
    V = (1-V)**2
    return V

#Logistic difference Loss
def logistic (f,y,mu):
    a = math.log(1 + math.exp(-y*f))
    b = math.log(1 + math.exp(-y*f-mu))
    V=a-b
    return V

#Smoothed 0-1 Loss
def smooth01 (t):
    if t>1:
        V=0
    elif t>=-1 and t<=1:
        V=1/4*t**3-3/4*t+1/2
    elif t<-1:
        V=1
    return V
    
