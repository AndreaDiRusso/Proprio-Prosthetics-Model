# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:54:58 2017

@author: adirusso
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import pandas as pd
from helper_functions import *
import math
import matplotlib.pyplot as plt
from collections import deque
import os, argparse, pickle, copy
try:
    from mujoco_py.generated import const
    from mujoco_py import functions
except:
    pass

import pdb, copy
from mpl_toolkits.mplot3d import Axes3D
import math

try:
    import quaternion as quat
except:
    pass

def Force_feed_gain(simulation, act_id, Len_Opt):
#   IL_Left = 0, GMED_Left = 1, GMAX_Left = 2, BF_Left = 3, RF_Left = 4, VAS_Left = 5, GAS_Left = 6, SOL_Left = 7, TA_Left = 8, EDL_Left = 9, FHL_Left = 10
#   IL_Left = 11, GMED_Left = 12, GMAX_Left = 13, BF_Left = 14, RF_Left = 15, VAS_Left = 16, GAS_Left = 17, SOL_Left = 18, TA_Left = 19, EDL_Left = 20, FHL_Left = 21    
#   
    F_max = np.array([1000, 643.99, 1182.40, 401.17, 432.84, 717.89, 2512.61, 717.89, 823.46, 253.37, 137.24, 306.16, 
                            643.99, 1182.40, 401.17, 432.84, 717.89, 2512.61, 717.89, 823.46, 253.37, 137.24, 306.16])   
    c = 0.05
    w = 0.4
        
    f_l = math.exp(c*abs((simulation.data.actuator_length[act_id] - Len_Opt[act_id])/w*Len_Opt[act_id])  **3)
        
    v_max = -12*Len_Opt[act_id]
    N = 1.5
    K = 5
        
    if simulation.data.actuator_velocity[act_id] < 0:
            
        f_v = (v_max - simulation.data.actuator_velocity[act_id])/(v_max + K*simulation.data.actuator_velocity[act_id])
    
    else:
        
        f_v = N + (N-1)*(v_max + simulation.data.actuator_velocity[act_id])/(7.56*K*simulation.data.actuator_velocity[act_id] - v_max)
        
    simulation.model.actuator_biasprm[act_id, :] = np.array([0., 0., 0.])
    simulation.model.actuator_gainprm[act_id, :] = np.array([F_max[act_id], 0., 0.])*f_l*f_v
    simulation.model.actuator_dynprm[act_id, :] = np.array([0.01, 0., 0.])
    
    F = - simulation.data.act[act_id-1]*simulation.model.actuator_gainprm[act_id, 0] #simulation.data.act NoneType!!!
    
    return F, simulation.model.actuator_gainprm[act_id, 0], f_l, f_v

def contact(simulation, debugging = False, zeroPad = True):

    Contact = ['None']*6
    Force_contact = np.zeros((6, 1))
    i = 0
    for idx, contact in enumerate(simulation.data.contact):

        contactForce = np.zeros((6))
        functions.mj_contactForce(simulation.model, simulation.data, idx, contactForce)
        
        if np.sum(contactForce**2) > 0:
            C = simulation.model.geom_id2name(contact.geom1)
            F = np.sqrt(np.sum(contactForce**2))
            if C != None:
                Contact[i] = C
                Force_contact[i] = F
                i = i+1
            if debugging:
                print('Contact geom 1:')
                print(simulation.model.geom_id2name(contact.geom1))
                print('Contact geom 2:')
                print(simulation.model.geom_id2name(contact.geom2))
                print('Contact Force:')
                print(contactForce)
                print('------------------------------------------')                

    return Contact, Force_contact    


MjModel = load_model_from_path('C:/Users/adirusso/Documents/GitHub/Proprio-Prosthetics-Model/Version-3-0/murdoc_gen.xml')

simulation =  MjSim(MjModel, data=None, nsubsteps=2, udd_callback=None)

simulation.model.body_mass[1] = 10000
M = simulation.model.body_mass
monkey = np.sum(M) - M[1]
functions.mj_setTotalmass(MjModel, np.sum(M)*10/monkey)
simulation =  MjSim(MjModel, data=None, nsubsteps=2, udd_callback=None)

simulation.forward()

dt = 0.002  
t = 0.002

while (t<=0.40):
    
    simulation.step()
    
    t=t+dt
    
Len_Opt = simulation.data.actuator_length

sim =  MjSim(MjModel, data=None, nsubsteps=2, udd_callback=None)

sim.model.body_mass[1] = 10000
M = sim.model.body_mass
monkey = np.sum(M) - M[1]
functions.mj_setTotalmass(MjModel, np.sum(M)*10/monkey)
sim =  MjSim(MjModel, data=None, nsubsteps=2, udd_callback=None)

sim.forward()    

n_act = len(sim.data.ctrl)
dt = 0.002  
t = 0.002
F_m = np.asarray([0]*n_act)
fl = np.asarray([0]*n_act)
fv = np.asarray([0]*n_act)
G = np.asarray([0]*n_act)
L_ce = np.asarray([0]*n_act)
k_bw = -1.2/1000
Delta_S = -0.25
F_l = 0
F_r = 0
Teta_R = 0
Teta_L = 0

joints = {
                "Hip_Right:x" : {'value': -0.70},
                "Hip_Right:y" : {'value': -0.17},
                "Hip_Right:z" : {'value': 0.0},
                "Knee_Right:x" : {'value': 2.09},
                "Ankle_Right:x" : {'value': -0.87},
                "Ankle_Right:y" : {'value': 0.00},
                "Toes_Right:x" : {'value': -2.09},
                "Hip_Left:x" : {'value': -0.85},
                "Hip_Left:y" : {'value': 0.17},
                "Hip_Left:z" : {'value': 0.},
                "Knee_Left:x" : {'value': 0.},
                "Ankle_Left:x" : {'value': 0.79},
                "Ankle_Left:y" : {'value': 0.},
                "Toes_Left:x" : {'value': 0.}
                }
    
sim = pose_model(sim, joints)

S_stance = -0.01*np.ones(n_act)
S_stance[[6, 17]] = -0.05*np.ones(2)
S_stance[[3, 4, 5, 14, 15, 16]] = -0.09*np.ones(6)

S_swing = -0.01*np.ones(n_act)

F_max = np.array([10, 643.99, 1182.40, 401.17, 432.84, 717.89, 2512.61, 717.89, 823.46, 253.37, 137.24, 306.16, 
                      643.99, 1182.40, 401.17, 432.84, 717.89, 2512.61, 717.89, 823.46, 253.37, 137.24, 306.16])  

G_m = -np.array([0, 0, 0, 0.4, 0.65, 0.35, 1.15, 1.1, 1.2, 1.1, 1.1, 1.2, 
                    0, 0, 0.4, 0.65, 0.35, 1.15, 1.1, 1.2, 1.1, 1.1, 1.2,
                    0.3, 4, 0.3])
    
for i in range(len(F_max)):
    G_m[i] = G_m[i]/F_max[i]
    
G_m[23] = G_m[23]/F_max[8]
G_m[24] = G_m[24]/F_max[4]
G_m[25] = G_m[25]/F_max[11]
    
d_tl = -10
d_tm = -5
d_ts = -2

Loff_TA = np.array([0.71*Len_Opt[9], 0.71*Len_Opt[20]])
Loff_EDL = np.array([0.71*Len_Opt[10], 0.71*Len_Opt[21]])
Loff_RF = np.array([0.6*Len_Opt[5], 0.6*Len_Opt[16]])
Loff_BF = np.array([0.85*Len_Opt[4], 0.85*Len_Opt[15]])

Time = int(10/dt)

F_m = np.zeros((Time, len(sim.data.ctrl))) 
F_left = np.zeros((Time, 1))
F_right = np.zeros((Time, 1))
Teta_R = np.zeros((Time, 1)) 
Teta_L = np.zeros((Time, 1))  
P = np.zeros((Time, len(sim.data.ctrl)))
F = np.zeros((Time, len(sim.data.ctrl)))
L_ce = np.zeros((Time, len(sim.data.ctrl)))
Gain = np.zeros((Time, len(sim.data.ctrl)))
function_l = np.zeros((Time, len(sim.data.ctrl)))
function_v = np.zeros((Time, len(sim.data.ctrl)))
Fl = [0]*Time
Fr = [0]*Time
p = np.zeros((Time, 1))
Stance_R = np.zeros((Time, 1))
Stance_L = np.zeros((Time, 1))
    
while (t<=0.40):
    
    sim.step()
    
    t=t+dt

sim.data.ctrl[0] = 1.

viewer = MjViewer(sim)    

while (t<=10):
        
    if t>2.5 and t-dt<2.5:
        sim.data.ctrl[0] = 0
        v1 = sim.data.qvel[0]

    for act_id in range(1, n_act):
        F_m[int(t/dt-1), act_id], Gain[int(t/dt-1), act_id], function_l[int(t/dt-1), act_id], function_v[int(t/dt-1), act_id]  = Force_feed_gain(sim, act_id, Len_Opt)
    
    L_ce[int(t/dt-1), :] = sim.data.actuator_length
    
    stance_right = 0
    stance_left = 0
    Dsup = 0

    for j in range(len(sim.data.site_xpos)):
        if sim.model.site_id2name(j)=='GT_Right':
            GT_Right = sim.data.site_xpos[j]
        elif sim.model.site_id2name(j)=='GT_Left':
            GT_Left = sim.data.site_xpos[j]
    T_R = np.asarray(np.arctan(GT_Right[2]/GT_Right[1]))
    T_L = np.asarray(np.arctan(GT_Left[2]/GT_Left[1]))
    Teta_R[int(t/dt-1)] = T_R
    Teta_R_dot = np.gradient(Teta_R, dt, axis = 0)
    Teta_L[int(t/dt-1)] = T_L
    Teta_L_dot = np.gradient(Teta_L, dt, axis = 0)
    
    Contact, Force_contact = contact(sim, debugging = False, zeroPad = True)
    r = np.zeros((len(Contact), 1))
    l = np.zeros((len(Contact), 1))
    for i in range(len(Contact)):
        if 'Right' in Contact[i]:
            stance_right = 1
            r[i] = Force_contact[i]
            
        if 'Left' in Contact[i]:
            stance_left = 1
            l[i] = Force_contact[i]
            
    F_l = np.asarray(np.sum(l))
    F_r = np.asarray(np.sum(r))
    
    F_left[int(t/dt-1)] = F_l
    
    F_right[int(t/dt-1)] = F_r
    
    Tau_des_swing_R =  Teta_R[int(t/dt+d_ts)] + Teta_R_dot[int(t/dt+d_ts)]
    Tau_des_swing_L =  Teta_L[int(t/dt+d_ts)] + Teta_L_dot[int(t/dt+d_ts)]
    
    Tau_des_stance_R = F_right[int(t/dt+d_ts)]*( Teta_R[int(t/dt+d_ts)] + Teta_R_dot[int(t/dt+d_ts)] - Tau_des_swing_L)
    Tau_des_stance_L = F_left[int(t/dt+d_ts)]*( Teta_L[int(t/dt+d_ts)] + Teta_L_dot[int(t/dt+d_ts)] - Tau_des_swing_R) 
            
    Dsup = stance_right*stance_left
     
    if stance_right==1:
        
        sim.data.ctrl[12] = S_stance[12] - Tau_des_stance_R/(F_max[12]*sim.data.ten_moment[11, 8])
        sim.data.ctrl[13] = S_stance[13] + Tau_des_stance_R/(F_max[13]*sim.data.ten_moment[12, 8])
        sim.data.ctrl[14] = S_stance[14] + k_bw*F_right[int(t/dt+d_ts)] - Delta_S*Dsup #positive!!
        sim.data.ctrl[15] = S_stance[15] + k_bw*F_right[int(t/dt+d_ts)]
        sim.data.ctrl[16] = S_stance[16] + k_bw*F_right[int(t/dt+d_ts)] + Delta_S*Dsup
        sim.data.ctrl[17] = S_stance[17] + G_m[17]*F_m[int(t/dt+d_tm), 17] - k_bw*F_left[d_ts]*Dsup
        sim.data.ctrl[18] = S_stance[18] + G_m[18]*F_m[int(t/dt+d_tl), 18]
        sim.data.ctrl[19] = S_stance[19] + G_m[19]*F_m[int(t/dt+d_tl), 19] 
        sim.data.ctrl[20] = S_stance[20] + G_m[20]*(L_ce[int(t/dt+d_tl), 20] - Loff_TA[1]) - G_m[23]*F_m[int(t/dt+d_tl), 19]
        sim.data.ctrl[21] = S_stance[21] + G_m[21]*(L_ce[int(t/dt+d_tl), 21] - Loff_EDL[1]) - G_m[25]*F_m[int(t/dt+d_tl), 22]        
        sim.data.ctrl[22] = S_stance[22] + G_m[22]*F_m[int(t/dt+d_tl), 22]
        
    elif stance_right==0:
        
        sim.data.ctrl[12] = S_swing[12] - Tau_des_swing_R/(F_max[12]*sim.data.ten_moment[11, 8])
        sim.data.ctrl[13] = S_swing[13] + Tau_des_swing_R/(F_max[13]*sim.data.ten_moment[12, 8])
        sim.data.ctrl[14] = S_swing[14] + G_m[14]*F_m[int(t/dt+d_ts), 14]
        sim.data.ctrl[15] = S_swing[15] + G_m[15]*F_m[int(t/dt+d_ts), 15] 
        sim.data.ctrl[16] = S_swing[16] + G_m[16]*(L_ce[int(t/dt+d_ts), 16] - Loff_RF[1]) - G_m[24]*(L_ce[int(t/dt+d_ts), 15] - Loff_BF[1])
        sim.data.ctrl[17] = S_swing[17]
        sim.data.ctrl[18] = S_swing[18]        
        sim.data.ctrl[19] = S_swing[19]
        sim.data.ctrl[20] = S_swing[20] + G_m[20]*(L_ce[int(t/dt+d_tl), 20] - Loff_TA[1])
        sim.data.ctrl[21] = S_swing[21] + G_m[21]*(L_ce[int(t/dt+d_tl), 21] - Loff_EDL[1])
        sim.data.ctrl[22] = S_swing[22]  
        
    if stance_left==1:
        
        sim.data.ctrl[1] = S_stance[1] - Tau_des_stance_L/(F_max[1]*sim.data.ten_moment[0, 15])
        sim.data.ctrl[2] = S_stance[2] + Tau_des_stance_L/(F_max[2]*sim.data.ten_moment[1, 15])
        sim.data.ctrl[3] = S_stance[3] + k_bw*F_left[int(t/dt+d_ts)] - Delta_S*Dsup
        sim.data.ctrl[4] = S_stance[4] + k_bw*F_left[int(t/dt+d_ts)]
        sim.data.ctrl[5] = S_stance[5] + k_bw*F_left[int(t/dt+d_ts)] + Delta_S*Dsup
        sim.data.ctrl[6] = S_stance[6] + G_m[6]*F_m[int(t/dt+d_tm), 6] - k_bw*F_right[int(t/dt+d_ts)]*Dsup
        sim.data.ctrl[7] = S_stance[7] + G_m[7]*F_m[int(t/dt+d_tl), 7] 
        sim.data.ctrl[8] = S_stance[8] + G_m[8]*F_m[int(t/dt+d_tl), 8]
        sim.data.ctrl[9] = S_stance[9] + G_m[9]*(L_ce[int(t/dt+d_tl), 9] - Loff_TA[0]) - G_m[23]*F_m[int(t/dt+d_tl), 8] #!!!
        sim.data.ctrl[10] = S_stance[10] + G_m[10]*(L_ce[int(t/dt+d_tl), 10] - Loff_TA[0]) - G_m[25]*F_m[int(t/dt+d_tl), 11] #!!!
        sim.data.ctrl[11] = S_stance[11] + G_m[11]*F_m[int(t/dt+d_tl), 11]
        
    elif stance_left==0:
                  
        sim.data.ctrl[1] = S_swing[1] - Tau_des_swing_L/(F_max[1]*sim.data.ten_moment[0, 15])
        sim.data.ctrl[2] = S_swing[2] + Tau_des_swing_L/(F_max[2]*sim.data.ten_moment[1, 15])
        sim.data.ctrl[3] = S_swing[3] + G_m[3]*F_m[int(t/dt+d_ts), 3]
        sim.data.ctrl[4] = S_swing[4] + G_m[4]*F_m[int(t/dt+d_ts), 4]
        sim.data.ctrl[5] = S_swing[5] + G_m[5]*(L_ce[int(t/dt+d_ts), 5] - Loff_RF[0]) - G_m[24]*(L_ce[int(t/dt+d_ts), 4] - Loff_BF[0])
        sim.data.ctrl[6] = S_swing[6]      
        sim.data.ctrl[7] = S_swing[7]
        sim.data.ctrl[8] = S_swing[8]
        sim.data.ctrl[9] = S_swing[9] + G_m[9]*(L_ce[int(t/dt+d_tl), 9] - Loff_TA[0]) 
        sim.data.ctrl[10] = S_swing[10] + G_m[10]*(L_ce[int(t/dt+d_tl), 10] - Loff_EDL[0])       
        sim.data.ctrl[11] = S_swing[11]
        
    #viewer.cam.fixedcamid += 1
    #viewer.cam.type = const.CAMERA_FIXED
    
    #for i in range(1, len(sim.data.ctrl)):
     #       if sim.data.ctrl[i]>=0:
      #          sim.data.ctrl[i]=0
                
          #  elif sim.data.ctrl[i]<=-1:
           #     sim.data.ctrl[i]=-1
           
    Stance_R[int(t/dt-1)] = stance_right
    Stance_L[int(t/dt-1)] = stance_left
    
    P[int(t/dt-1), :] = sim.data.ctrl
    
    viewer.render()
    sim.step()
    
    t=t+dt
    
'''
plt.figure(0)
plt.plot(k_bw*F_right) 
plt.show()

plt.figure(1)
plt.plot(k_bw*F_left) 
plt.show()

B = P
for i in range(len(P[:, 0])):
    for j in range(len(P[0, :])):
        if P[i, j]<=-1:
            B[i, j] = -1
        elif P[i, j]>=0:
            B[i, j] = 0
            


for i in range(1, len(P[0, :])):
    plt.figure(i)
    plt.plot(P[:, i]) 
    plt.show()

tr = np.zeros((Time, 1))

for i in range(int(0.40*dt), len(Stance_L)):
    
    if Stance_L[i] == 1:
        tr[i] = S_stance[6] + G_m[6]*F_m[i+d_ts, 7]
        
    else:
        tr[i] = S_swing[6]


plt.figure(0)
plt.plot(Stance_R)
plt.show() 


plt.figure(1)
plt.plot(Stance_L)
plt.show() 

plt.figure(2)
plt.plot(p)
plt.show() 

plt.figure(3)
plt.plot(G_m[7]*F_m[:, 7] )
plt.show() 

plt.figure(4)
plt.plot(S_stance[6] + G_m[6]*F_m[:, 7] )
plt.show()
    
plt.figure(5)
plt.plot([S_swing[6]]*len(sim.data.ctrl))
plt.show()
v2 = sim.data.qvel[0] 

'''

#   Tread = 0     
#   IL_Left  =  1, GMED_Left  =  2, GMAX_Left  =  3, BF_Left  =  4, RF_Left  =  5, VAS_Left  =  6, GAS_Left  =  7, SOL_Left  =  8, TA_Left  =  9, EDL_Left  =  10, FHL_Left  = 11
#   IL_Right = 12, GMED_Right = 13, GMAX_Right = 14, BF_Right = 15, RF_Right = 16, VAS_Right = 17, GAS_Right = 18, SOL_Right = 19, TA_Right = 20, EDL_Right = 21, FHL_Right = 22    # 
    
    
