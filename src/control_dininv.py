#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from markers import *
from projfunctions import *
from roslib import packages

import rbdl


rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
fqact = open("/tmp/qactual.dat", "w")
fqdes = open("/tmp/qdeseado.dat", "w")
fxact = open("/tmp/xactual.dat", "w")
fxdes = open("/tmp/xdeseado.dat", "w")

# Nombres de las articulaciones
jnames = ['pan_joint', 'tilt_joint', 'joint10', 'joint11', 'jc1', 'jc2',  'jc3']

# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q  = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 0.4, 0.5])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([1.0, 1.0, 0.0, 0.3, 0., 0, 0])
# Velocidad articular deseada
dqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Aceleracion articular deseada
ddqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('./../urdf/robot.urdf')
ndof   = modelo.q_size     # Grados de libertad
zeros = np.zeros(ndof)     # Vector de ceros

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = 20*np.diag(valores)
Kd = 2*np.sqrt(Kp)

while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine(q)[0:3,3]
    print(x.shape)
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+','+str(x[0])+','+str(x[1])+','+str(x[2])+'\n') #+','+str(x[3])+','+str(x[4])+','+str(x[5])+','+str(x[6])
    fxdes.write(str(t)+','+str(xdes[0])+','+str(xdes[1])+','+str(xdes[2])+'\n') #+','+str(xdes[3])+','+str(xdes[4])+','+str(xdes[5])+','+str(xdes[6])+
    fqact.write(str(t)+','+str(q[0])+','+str(q[1])+','+ str(q[2])+
                ','+ str(q[3])+','+str(q[4])+','+str(q[5])+','+str(q[6])+'\n ')
    fqdes.write(str(t)+','+str(qdes[0])+','+str(qdes[1])+','+ str(qdes[2])+
                ','+ str(qdes[3])+','+str(qdes[4])+','+str(qdes[5])+','+str(qdes[6])+'\n ')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
    M = np.zeros((ndof, ndof))
    b = np.zeros((ndof))

    rbdl.CompositeRigidBodyAlgorithm(modelo, q, M)
    #print(M.shape)
    rbdl.NonlinearEffects(modelo, q, dq, b)

    Kp = np.diag([15.5, 10., 5.0, 3.5, 2.9, 2.8, 2.4])
    Kd = np.diag([7., 6., 3., 2.4, 2.6, 2.3, 2.3])

    y = ddqdes - ddq + Kd.dot(dqdes - dq) + Kp.dot(qdes - q)
    u = M.dot(y) + b
    print(M)
    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
