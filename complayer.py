import sys
import time
from zmq_comm import *
from evdev import InputDevice
from select import select

WIDTH = 600
HEIGHT = 400-20       
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80

def predict_ballpos(ballpos, ballspd, pad2pos):
    y = HEIGHT/2
    if(ballspd[0] > 0):
        dx = pad2pos[0][0] - ballpos[0] - 20
        t = dx / ballspd[0]
        y = ballpos[1] + t * ballspd[1]
        if(y > HEIGHT):
            y = 2*HEIGHT - y
        elif(y < 20):
            y = 40 - y
    else:
        pass

    if(y-10 < pad2pos[0][1]):
        return 'up'
    elif(y+10 > pad2pos[3][1]):
        return 'down'
    else:
        return 'hold'

ppp = zmq_comm_cli_c(name='pong',port=1201)
shutdown = False
    

param = {}
param['com'] = 'hold'
qp = ''

while(not shutdown):
    res = ppp.query(qp)
    param['com'] = predict_ballpos(res['ball_pos'], res['ball_vel'], res['paddle2_pos'])
    ppp.execute(param)
    time.sleep(0.01)
'''
'''
#ppp.stop()
print('end')