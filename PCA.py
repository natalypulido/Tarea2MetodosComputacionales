import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

##Implementacion de pandas tomado de: Dropping Rows And Columns In pandas Dataframe (https://chrisalbon.com/python/pandas_dropping_column_and_rows.html) 
##Note: axis=1 denotes that we are referring to a column, not a row
datos = pd.read_csv('DatosBancoMundial5.csv')
datos = datos.drop('Time', axis=1)
datos = datos.drop('Time Code', axis=1)
datos = datos.drop('Series Name', axis=1)
datos = datos.drop('Series Code', axis=1)
datos = datos.as_matrix()

##Leer los archivos de datos y guardar las variables relevantes en arrays
total_impuestos= datos[0,:]
costo_negocio= datos[1,:]
desempleo_femenino= datos[2,:]
desempleo_masculino= datos[3,:]
ratio= datos[4,:]

#print total_impuestos
#print costo_negocio
#print desempleo_femenino
#print desempleo_masculino
#print ratio


##Centrar y normalizar los datos
##total_impuestos
mean_total_impuestos=np.mean(total_impuestos)
#print mean_total_impuestos
std_total_impuestos=np.std(total_impuestos)
#print std_total_impuestos

total_impuestos_norm=[]
for i in range (len(total_impuestos)):
    a=((total_impuestos[i]-mean_total_impuestos)/std_total_impuestos)
    total_impuestos_norm.append(a) 
#print total_impuestos_norm

##costo_negocio
mean_costo_negocio=np.mean(costo_negocio)
#print mean_costo_negocio
std_costo_negocio=np.std(costo_negocio)
#print std_costo_negocio

costo_negocio_norm=[]
for i in range (len(costo_negocio)):
    b=((costo_negocio[i]-mean_costo_negocio)/std_costo_negocio)
    costo_negocio_norm.append(b) 
#print costo_negocio_norm

##desempleo_femenino
mean_desempleo_femenino=np.mean(desempleo_femenino)
#print mean_desempleo_femenino
std_desempleo_femenino=np.std(desempleo_femenino)
#print std_desempleo_femenino

desempleo_femenino_norm=[]
for i in range (len(desempleo_femenino)):
    c=((desempleo_femenino[i]-mean_desempleo_femenino)/std_desempleo_femenino)
    desempleo_femenino_norm.append(c) 
#print desempleo_femenino_norm

##desempleo_masculino
mean_desempleo_masculino=np.mean(desempleo_masculino)
#print mean_desempleo_masculino
std_desempleo_masculino=np.std(desempleo_masculino)
#print std_desempleo_masculino

desempleo_masculino_norm=[]
for i in range (len(desempleo_masculino)):
    d=((desempleo_masculino[i]-mean_desempleo_masculino)/std_desempleo_masculino)
    desempleo_masculino_norm.append(d) 
#print desempleo_masculino_norm

##ratio
mean_ratio=np.mean(ratio)
#print mean_ratio
std_ratio=np.std(ratio)
#print std_ratio

ratio_norm=[]
for i in range (len(ratio)):
    e=((ratio[i]-mean_ratio)/std_ratio)
    ratio_norm.append(e) 
#print ratio_norm

##Matriz total
matriz_tot=[]
matriz_tot.append(total_impuestos_norm)
matriz_tot.append(costo_negocio_norm)
matriz_tot.append(desempleo_femenino_norm)
matriz_tot.append(desempleo_masculino_norm)
matriz_tot.append(ratio_norm)
#print matriz_tot

##Corvierto mi matriz en un array para implementarlo en el calculo de las matriz de covarianza 
matrix = np.asarray(matriz_tot)
#print matrix

##Saco el np.cov para comprobrar/comparar con mi calculo de la matriz de covarianza  
matriz_cov=np.cov(matriz_tot)
#print matriz_cov


##Grafica de datos
fig, medidas= plt.subplots(5, figsize=(10,10)) 
fig.text(0.5, 0.04, 'common X', ha='center')
fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
medidas[0].plot(total_impuestos, color='gold', label='Total impuestos')
medidas[0].legend()
medidas[0].set_title('Exploracion de Datos')
medidas[1].plot(costo_negocio_norm, color='blue', label='Costo negocio')
medidas[1].legend()
medidas[2].plot(desempleo_femenino_norm, color='red', label='Desempleo femenino')
medidas[2].legend()
medidas[3].plot(desempleo_masculino_norm, color='green', label='Desempleo masculino' )
medidas[3].legend()
medidas[4].plot(ratio_norm, color='fuchsia', label='Ratio')
medidas[4].legend()
plt.legend()
#plt.show()
plt.savefig("ExploracionDatos.pdf")
plt.close()

##Matriz de covarianza 
n = np.size(matrix[0])
#print n

matriz_covarianza = np.zeros([5,5])
#print matriz

for i in range(0,5):
    for j in range (i,5):
        ##Contador 
        new=0
        for k in range(n):
            ##Voy sumando 
            new +=(matrix[i,k]-np.mean(matrix[:,i]))*(matrix[j,k]-np.mean(matrix[j,:]))/(n-1)
        matriz_covarianza[i,j]=new
        matriz_covarianza[j,i]=new                                            
#print matriz_covarianza                           


##Dos componentes principales del problema 
valores, vectores=np.linalg.eig(matriz_covarianza)
#print "valores", valores
#print "vectores\n", vectores 

print "El primer componente principal es", vectores[:,0]
print "El segundo componente principal es", vectores[:,1]


##Graficar los datos nuevamente en el sistema de referencia de los dos componentes principales

#Transponer los datos en un nuevo sub espacio
trans_vectores=np.transpose(vectores)
#print trans_vectores

trans_datos=np.transpose(datos)
#print trans_datos

##Para obtener una matriz 5x223. 
nuevos_datos = np.dot(trans_vectores, matriz_tot)

#print nuevos_datos
#shape(nuevos_datos)

##Grafica de datos en el nuevo sub espacio 
fig = plt.figure(figsize=(13,4))
ax = plt.axes()
plt.scatter(nuevos_datos[0,:], nuevos_datos[1,:])
x_line = np.linspace(-3.0,3.0)
ax.set_aspect(1.0)
plt.title('Transformed samples')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
#plt.show()
plt.savefig('PCAdatos.pdf')
plt.close()

##Grafica agrupaciones de las variables originales en el sistema de referencia de los dos componentes principales

#Tengo en cuenta las componentes principales 
PC_1=vectores[:,0]
#print PCA_1
PC_2=vectores[:,1]
#print PCA_2

plt.scatter(PC_1, PC_2)

plt.scatter(PC_1[0], PC_2[0],color='yellow',marker='^',s=120, alpha=1, label='Total impuestos')
plt.scatter(PC_1[1], PC_2[1],color='blue',marker='^',s=120, alpha=1, label='Costo negocios')
plt.scatter(PC_1[2], PC_2[2],color='red',marker='^',s=120, alpha=1, label='Desempleo Femenino')
plt.scatter(PC_1[3], PC_2[3],color='green',marker='^',s=120, alpha=1, label='Desempleo Masculino')
plt.scatter(PC_1[4], PC_2[4],color='fuchsia',marker='^',s=120, alpha=1, label='Ratio')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.title('Agrupacion de Variables')
plt.legend(loc=1,prop={'size':10})
#plt.show()
plt.savefig('PCAvariables.pdf')
plt.close()


print ("VARIABLES CORRELACIONADAS: Teniendo en cuenta la grafica de agrupaciones principales se observa que: Las variables de total impuestos y costos de negocios (color amarillo y azul) se pueden agrupar y se le asignaria el nombre de VALORES, las variables de desempleo femenino y desempleo masculino (color rojo y verde) se pueden agrupar y se le asignaria el nombre de DESEMPLEO y finalmente el ultimo grupo seria RATIO (color rosado). Con esto concluyo que nuestro problema se reduce a un grupo de 3 variables en donde principalmente se tenian 5 variables.")
