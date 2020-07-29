import numpy as np
from copy import copy
import rbdl

pi = np.pi

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/robot.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.pinv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq

def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    
    """
    l=np.array([0.022,1.5,0.502,0.6,0.4,0.155,0.07,0.14])#0.4
    # Longitudes (en metros)
    T0 = dh(0,pi,0,pi/2)
    T1 = dh(l[1]-q[0],pi,0,pi/2)
    T2 = dh(l[0]+l[2],pi/2+q[1],0,pi/2)
    T3 = dh(0,q[2]-pi/2,-l[3],0)
    T4 = dh(0,-pi/2+q[3],-l[4],pi/2)
    T5 = dh(0,pi+q[4],l[5],pi/2)
    T6 = dh(0,pi/2+q[5],0,pi/2)
    T7 = dh(l[6]+l[7],q[6],0,0)
    
    # Efector final con respecto a la base
    T = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(T0,T1),T2),T3),T4),T5),T6),T7)
    return T



def ikine(l,xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    """

    #se utiliza metodo de newton
    epsilon  = 0.001
    max_iter = 5000
    delta    = 0.00001
    

    q  = copy(q0)
    for i in range(max_iter):#iteracion
        # Main loop
        
        J = jacobian_samo(l,q, 0.0001)#jacobiano de q
        T=fkine_samo(l,q)#cin directa de 1
        f = T[:3,3]#posicion actual segun cin directa
        e = xdes-f #error (pos deseada - actual)
        q = q + np.dot(np.linalg.pinv(J), e) #actualizacion de q
        print(e)
        # Condicion de termino 
        if (np.linalg.norm(e) < epsilon):#break si el error es menor al tolerable 
            break
        pass
    q1=q[0]
    q = q%(2*np.pi)#se normaliza q
    q[0]=q1
    if np.linalg.norm(e)>epsilon:#si el error es muy grande vuelve al q inicial
        q=q0
    return q


def fkine_samo(l,q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6,q7]
    """
    # Longitudes (en metros)

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    #T1 = dh(l[0],pi+q[0],0,pi/2)
    #T2 = dh(l[1]-q[1],-pi/2,-l[2],-pi/2)
    #T3 = dh(0,-q[2],-l[3],0)
    #T4 = dh(0,pi/2-q[3],-l[4],pi/2)
    #T5 = dh(0,-pi-q[4],0,pi/2)
    #T6 = dh(0,pi/2-q[5],0,pi/2)
    #T7 = dh(l[5],pi+q[6],0,0)
    #T1 = dh(l[0],pi+q[0],0,pi/2)
    T0 = dh(0,pi,0,pi/2)
    T1 = dh(l[1]-q[0],pi,0,pi/2)
    T2 = dh(l[0]+l[2],pi/2+q[1],0,pi/2)
    T3 = dh(0,q[2]-pi/2,-l[3],0)
    T4 = dh(0,-pi/2+q[3],-l[4],pi/2)
    T5 = dh(0,pi+q[4],l[5],pi/2)
    T6 = dh(0,pi/2+q[5],0,pi/2)
    T7 = dh(l[6]+l[7],q[6],0,0)

    # Efector final con respecto a la base con producto  punto
    
    T = T0.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7)
    
    return T


def jacobian_samo(l,q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x6
    J = np.zeros((3,7))#3 debido a x y z y 7 por las 6 articulaciones
    # Transformacion homogenea inicial (usando q)
    T=fkine_samo(l,q)
    
    # Iteracion para la derivada de cada columna
    for i in xrange(7):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i]=dq[i]+delta#incremento de delta en la artic. i 
        # Transformacion homogenea luego del incremento (q+delta)
        dT=fkine_samo(l,dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[:,[i]]=(dT[:3,[3]]-T[:3,[3]])/delta#se va ajustando artic. por artic.
    return J


def ikine_samo(l,xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    """

    #se utiliza metodo de newton
    epsilon  = 0.001
    max_iter = 5000
    delta    = 0.00001
    

    q  = copy(q0)
    for i in range(max_iter):#iteracion
        # Main loop
        
        J = jacobian_samo(l,q, 0.0001)#jacobiano de q
        T=fkine_samo(l,q)#cin directa de 1
        f = T[:3,3]#posicion actual segun cin directa
        e = xdes-f #error (pos deseada - actual)
        q = q + np.dot(np.linalg.pinv(J), e) #actualizacion de q
        print(e)
        # Condicion de termino 
        if (np.linalg.norm(e) < epsilon):#break si el error es menor al tolerable 
            break
        pass
    q1=q[0]
    q = q%(2*np.pi)#se normaliza q
    q[0]=q1
    if np.linalg.norm(e)>epsilon:#si el error es muy grande vuelve al q inicial
        q=q0
    return q

def jacobian_pose(q, delta=0.001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J_p = np.zeros((3,7))
    J_o = np.zeros((4,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta  
        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J_p[:,i] = (dT[0:3, 3] - T[0:3, 3])/delta
        J_o[:,i] = quatError(rot2quat(T[0:3, 0:3]), rot2quat(dT[0:3, 0:3]))/delta #(rot2quat(dT[0:3, 0:3]) - rot2quat(T[0:3, 0:3]))/delta
    
    J = np.vstack((J_p,J_o))
    return J




def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R



def quatError(Qdes, Q):
    """
    Compute difference between quaterions.
    Input:
    ------
    	- Qdes: 	Desired quaternion
    	- Q   :		Current quaternion

    Output:
    -------
    	- Qe  :		Error quaternion	
    """

    we = Qdes[0]*Q[0] + np.dot(Qdes[1:4].transpose(),Q[1:4]) - 1
    e  = -Qdes[0]*Q[1:4] + Q[0]*Qdes[1:4] - np.cross(np.transpose(Qdes[1:4]), np.transpose(Q[1:4]))
    Qe = np.array([ we, e[0], e[1], e[2] ])

    return Qe   


def incremental(key,desiredpos):
    if key in ['w']:
        desiredpos[0]=desiredpos[0]+0.1
 
    if key in ['s']:
        desiredpos[0]=desiredpos[0]-0.1
 
    if key in ['a']:
        desiredpos[1]=desiredpos[1]+0.1
 
    if key in ['d']:
        desiredpos[1]=desiredpos[1]-0.1
 
    if key in ['q']:
        desiredpos[2]=desiredpos[2]+0.1
 
    if key in ['e']:
        desiredpos[2]=desiredpos[2]-0.1
 
    return desiredpos
