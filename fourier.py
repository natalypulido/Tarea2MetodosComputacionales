import numpy as np 
import matplotlib.pyplot as plt 
#from scipy.io.wavfile import read 
from scipy.io.wavfile import *
from scipy.fftpack import fft, fftfreq


#Leer los archivos de datos
sol = read('Sol.wav')
do = read('Do.wav')
#print sol
#print do 


##Implementacion de la transformada de Fourier
def hallar_elemento(n,arreglo):
    k=np.linspace(0,len(arreglo)-1,len(arreglo))
    expo=np.exp((-1j*2*np.pi*k*n)/len(arreglo))
    arreglofinal=arreglo*expo
    return(sum(arreglofinal))

transformada = np.zeros(len(do[1]), dtype=complex)
for n in range (len(do[1])-1):
    transformada[n]=hallar_elemento(n,do[1])


#Paquete de python para comprobar mi transformada de Fourier
#transformada=np.fft.fft(do[1])

##Frecuencias 
transformada_1=transformada.copy()
n=len(do[1])
#print n
sr=do[0]
dt=1.0/sr
#print dt
freq =np.fft.fftfreq(n, dt)
#print freq
#plot(freq,abs(transformada_1))


##Hacer una funcion que filtre y elimine la frecuencia con mayor amplitud
transformada_2=transformada.copy()
max_amplitud=np.argmax(abs(transformada_2))
#print transformada[max_amplitud]
frecuencia=freq[(max_amplitud)]
#print frecuencia
donde = np.where(freq==-frecuencia)[0][0]
transformada_2[max_amplitud-10:max_amplitud+10]=0
transformada_2[donde-10:donde+10]=0
#plot(freq, abs(transformada_2))

##Hacer una funcion que sea un filtro pasa bajos 
transformada_3=transformada.copy()
valor = 1000
transformada_3[abs(freq) > valor] = 0
#plt.plot(freq,(abs(transformada_3)))


##Grafica con tres sub-plots
fig, medidas= plt.subplots(3,1, figsize=(10,10))   
fig.text(0.5, 0.04, 'Frequency (common)', ha='center')
fig.text(0.04, 0.5, 'Amplitude (common)', va='center', rotation='vertical') 
medidas[0].plot(freq,transformada_1, label='Original', c='fuchsia')
medidas[0].legend(loc=0)
medidas[1].plot(freq,transformada_2, label='Sin frecuencia maxima', c='g')
medidas[1].legend(loc=0)
medidas[2].plot(freq,transformada_3, label='Filtro pasa bajos', c='gold')
medidas[2].legend(loc=0)
plt.legend()
plt.savefig("DoFiltros.pdf")
plt.close()

## Frecuencia  fundamental pase de ser 260Hz a 391Hz
transformada_4=transformada.copy()
SRdo= (do[0])
#print SRdo
new_samplerate=(SRdo*(391.0/260.0))
#print new_samplerate
n_1=len(do[1])
#print n
sr_new=new_samplerate
dt=1.0/sr_new
#print dt
frec_nueva=np.fft.fftfreq(n_1, dt)
#plot(frec_nueva, abs(transformada_4))

##Para sol trnasformada con implementacion propia
def halfontsize=8lar_elemento_sol(n,arreglo):
    k=np.linspace(0,len(arreglo)-1,len(arreglo))
    expo=np.exp((-1j*2*np.pi*k*n)/len(arreglo))
    arreglofinal=arreglo*expo
    return(sum(arreglofinal))

transformada_sol = np.zeros(len(sol[1]), dtype=complex)
for n in range (len(sol[1])-1):
    transformada_sol[n]=hallar_elemento_sol(n,sol[1])

##Pueba con paquete 
#transformada_sol=np.fft.fft(sol[1])
#plot(abs(transformada))

nsol=len(sol[1])
#print n
sr_sol=sol[0]
dt_sol=1.0/sr_sol
#print dt
freq_sol=np.fft.fftfreq(nsol, dt_sol)
#print freq
plt.plot(freq_sol, abs(transformada_sol), c='g')
plt.plot(frec_nueva, abs(transformada_4), c='r')
plt.tight_layout()
plt.xlabel('Frequency', fontsize=8)
plt.ylabel('Amplitude', fontsize=8)
plt.savefig("DoSol.pdf")
plt.close()

##Inversas
nueva_1=np.fft.ifft(transformada_2)
nueva_2=np.fft.ifft(transformada_3)
nueva_3=np.fft.ifft(transformada_sol)

##Parte real
real_nueva_1=np.real(nueva_1)
real_nueva_2=np.real(nueva_2)
real_nueva_3=np.real(nueva_3)

##Normalizo
real_nueva_1=(real_nueva_1/(real_nueva_1.max()-real_nueva_1.min()))*2
real_nueva_2=(real_nueva_2/(real_nueva_2.max()-real_nueva_2.min()))*2
real_nueva_3=(real_nueva_3/(real_nueva_3.max()-real_nueva_3.min()))*2

real_nueva_1=real_nueva_1.astype(np.float32)
real_nueva_2=real_nueva_2.astype(np.float32)
real_nueva_3=real_nueva_3.astype(np.float32)

lis1=write('Do_picos.wav', 16000, real_nueva_1)
lis2=write('Do_pasabajos.wav', 16000, real_nueva_2)
lis3=write('Dosol.wav', int(sr_new), real_nueva_3)




