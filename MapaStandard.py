import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from configparser import ConfigParser


#LECTURA DE PARAMETROS DE ENTRADA

parser = ConfigParser()
parser.read('Entradas.ini') 

K = parser.getfloat('Parametros','K')
n = parser.getint('Parametros','n')

t0 = parser.getfloat('CondicionesIniciales','t0')
p0 = parser.getfloat('CondicionesIniciales','p0')




K=K/(2*pi)


tt = np.zeros(n) 
pp = np.zeros(n)

tt2 = np.zeros(n) 
pp2 = np.zeros(n)


t = t0
p = p0

d0=1.e-9

p2=np.absolute(p0)+d0
t2=np.absolute(t0)+d0

delta=np.zeros(n)


for i in range(n):		
	p = np.mod(p + K * np.sin(2.*pi*t),1.)
	t = np.mod(t + p,1.)
	p2 = np.mod(p2 + K * np.sin(2.*pi*t2),1.)	
	t2 = np.mod(t2 + p2,1.)
	
	pp[i] = p
	tt[i] = t
	
	pp2[i] = p2
	tt2[i] = t2
	
	delta[i]=np.sqrt((p-p2)**2+(t-t2)**2)

x=np.linspace(1,n,n)
		
		
nombre='t0='+str(t0)+'. p0='+str(p0)

plt.figure()
plt.plot(tt,pp,'.',markersize='5',color='k')
plt.plot(tt2,pp2,'.',markersize='5',color='r')
plt.plot(t0,p0,'.',markersize='10.',color='g',label='Condicion Inicial')
plt.title('MapaStandard.'+nombre)
plt.savefig('MapaStandard.'+nombre+'.png')		

			
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('t')
plt.ylabel('p')

plt.show()

plt.figure()

plt.plot(x,delta,color='k')			
plt.xlim(0,n)
plt.title('Delta.'+nombre)
plt.xlabel('n')
plt.ylabel(r'$\delta$(n)')
plt.savefig('Delta.'+nombre+'.png')		
