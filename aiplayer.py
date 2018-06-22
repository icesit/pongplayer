import tensorflow as tf
from zmq_comm import *
import random
import time
import numpy as np

traing = False
actionlist = ['hold', 'up', 'down']
actiondict = {'hold':0, 'up':1, 'down':2}
learning_rate = 0.1
if(traing):
    CHOOSE_POLILCY_RATE = 0.8
else:
    CHOOSE_POLILCY_RATE = 1
GAMA = 0.9
ONE_STEP_COST = -0.1
PAD_REWARD = 10
posneg_rate = 0
tf.set_random_seed(777)

class action_q_nn_c:
    def __init__(self, name='name'):
        self.X = tf.placeholder(tf.float32, [None, 6])
        self.Y = tf.placeholder(tf.float32, [None, 1]) #hold up down

        self.W1 = tf.get_variable(name+"W1", shape=[6, 8],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.Variable(tf.random_normal([8]))
        L1 = tf.nn.sigmoid(tf.matmul(self.X, self.W1) + self.b1)

        self.W2 = tf.get_variable(name+"W2", shape=[8, 8],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.Variable(tf.random_normal([8]))
        L2 = tf.nn.sigmoid(tf.matmul(L1, self.W2) + self.b2)

        self.W3 = tf.get_variable(name+"W3", shape=[8, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.Variable(tf.random_normal([1]))
        self.hypothesis = tf.matmul(L2, self.W3) + self.b3
        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.sess = tf.Session()
        
        #self.saver = tf.train.Saver({'W1':self.W1, 'b1':self.b1, 'W2':self.W2, 'b2':self.b2, 'W3':self.W3, 'b3':self.b3})
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())
        self.name = name
        self.load()

    def train(self, inx, iny):
        feed_dict = {self.X:inx, self.Y:iny}
        #print(inx,iny)#self.minicost
        cost_val,_ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
        print('['+self.name+'] train cost:%f'%cost_val)

    def eval(self, inx):
        feed_dict = {self.X:inx}
        out = self.sess.run(self.hypothesis, feed_dict=feed_dict)
        return out

    def save(self):
        #saver = tf.train.Saver({self.name+'W1':self.W1, self.name+'b1':self.b1, self.name+'W2':self.W2, self.name+'b2':self.b2, self.name+'W3':self.W3, self.name+'b3':self.b3})
        self.saver.save(self.sess, 'model/model-'+self.name)
        print('save model '+self.name)
        #print(self.sess.run(self.W1))

    def load(self):
        ww1 = self.name+'W1'
        ww2 = self.name+'W2'
        ww3 = self.name+'W3'
        bb1 = self.name+'b1'
        bb2 = self.name+'b2'
        bb3 = self.name+'b3'
        self.saver = tf.train.Saver({ww1:self.W1,bb1:self.b1, ww2:self.W2, bb2:self.b2, ww3:self.W3, bb3:self.b3})
        if(os.path.exists('model/model-'+self.name+'.meta')):
            self.saver.restore(self.sess, 'model/model-'+self.name)
            print('load model '+self.name)
            #print(self.sess.run(self.W1))


class pongaiplayer:
    def __init__(self):
        self.ppp = zmq_comm_cli_c(name='pong',port=1201)
        self.param = {}
        self.param['ai'] = 'hold'
        self.qp = ''
        self.shutdown = False
        self.onetrainstate = []
        self.predictout_pos = []
        self.predictout_neg = []
        self.last_ball_vel = [0,0]
        self.max_train_size = 100
        self.cnt = 0
        #self.initnet()
        self.sess = tf.Session()
        self.nets_pos = [action_q_nn_c('hold'),action_q_nn_c('up') ,action_q_nn_c('down') ]
        self.nets_neg = [action_q_nn_c('holdneg'),action_q_nn_c('upneg') ,action_q_nn_c('downneg') ]
        self.file = open('samples/samples_pos.txt', 'a+')
        self.file = open('samples/samples_neg.txt', 'a+')
        #self.upnet = 
        #self.downnet = 
        #self.load()
        self.ppp.reset()
        return

    def action(self):
        res = self.ppp.query(self.qp)
        if(traing):
            self.train(res)
        else:
            self.param['ai'] = aiplayer.pong(res)
            self.ppp.execute(self.param)

    def pong(self, states):
        inx = [ [states['ball_pos'][0], states['ball_pos'][1], states['paddle1_pos'][0][0], states['paddle1_pos'][0][1], states['paddle1_pos'][3][0], states['paddle1_pos'][3][1]] ]
        #states['ball_vel'][0], states['ball_vel'][1], 
        #out = self.sess.run(self.hypothesis, feed_dict={self.X:inx})
        out = np.array([self.nets_pos[0].eval(inx)[0][0], self.nets_pos[1].eval(inx)[0][0], self.nets_pos[2].eval(inx)[0][0]])
        outneg = np.array([self.nets_neg[0].eval(inx)[0][0], self.nets_neg[1].eval(inx)[0][0], self.nets_neg[2].eval(inx)[0][0]])
        #print(out,outneg)
        mix = out*posneg_rate + (1-posneg_rate)*outneg
        if(traing):
            self.predictout_pos.append(out)
            self.predictout_neg.append(outneg)
        output = self.sess.run(tf.argmax(mix))
        print(out[0],out[1],out[2],outneg[0],outneg[1],outneg[2],actionlist[output])
        return actionlist[output]

    def train(self, states):
        if(states['ball_vel'][0]>0 and self.last_ball_vel[0]<0):
            pad_success = True
        else:
            pad_success = False
        #print(states['ball_vel'][0], self.last_ball_vel[0])
        self.last_ball_vel = states['ball_vel']
        if(pad_success or states['scorelr'][1]>0):
            # do one train
            #print('do one train')
            self.one_train(states, pad_success, states['scorelr'][1])
            self.ppp.reset()
            self.onetrainstate = []
            self.predictout_pos = []
            self.predictout_neg = []
            self.last_ball_vel = [0,0]
        else:
            if(len(self.onetrainstate)>self.max_train_size):
                self.onetrainstate.remove(self.onetrainstate[0])
            self.param['ai'] = aiplayer.pong(states)
            if(random.random()>CHOOSE_POLILCY_RATE):
                self.param['ai'] = actionlist[int(random.random()*3)]
            self.ppp.execute(self.param)
            states['action'] = actiondict[self.param['ai']]
            self.onetrainstate.append(states)
        

    def one_train(self,states,pad_success,scorer):
        self.cnt += 1
        if(self.cnt % 50 == 0):
            self.save()
            print('save one training result')
            #if(self.cnt > 5000):
            #    self.shutdown
            #    print('trained 5000 times')

        samples = len(self.onetrainstate)-1
        if(samples<1):
            return

        if(scorer>0):
            #return
            reward = -PAD_REWARD
            inx = [[],[],[]]
            iny = [[],[],[]]
            for j in range(samples+1):
                i = samples - j
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1],  self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                #self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1],
                oney = [reward]
                #print(self.onetrainstate[i]['action'])
                if(self.predictout_neg[i][self.onetrainstate[i]['action']] > oney[0]):
                    inx[self.onetrainstate[i]['action']].append(onex)
                    iny[self.onetrainstate[i]['action']].append(oney)
                if(not self.onetrainstate[i]['action'] == 0):
                    reward = reward*GAMA# + ONE_STEP_COST
                else:
                    reward = reward*GAMA
                
            print('train fail sample:%d'%samples)
            for i in range(3):
                if(len(inx[i])>0):
                    self.nets_neg[i].train(inx[i], iny[i])
        elif(pad_success):
            reward = PAD_REWARD
            inx = [[],[],[]]
            iny = [[],[],[]]
            for j in range(samples+1):
                i = samples - j
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1], self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                #self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1], 
                oney = [reward]
                if(self.predictout_pos[i][self.onetrainstate[i]['action']] < oney[0]):
                    inx[self.onetrainstate[i]['action']].append(onex)
                    iny[self.onetrainstate[i]['action']].append(oney)
                if(not self.onetrainstate[i]['action'] == 0):
                    reward = reward*GAMA + ONE_STEP_COST
                else:
                    reward = reward*GAMA
                
            print('train success sample:%d'%samples)
            for i in range(3):
                if(len(inx[i])>0):
                    self.nets_pos[i].train(inx[i], iny[i])
            self.savesample(inx, iny)
        '''
        if(scorer>0):
            return
            reward = -100
            inx = [[],[],[]]
            iny = [[],[],[]]
            cntup = []
            cntdown = []
            cnthold = []
            for i in range(samples):
                if(inx[self.onetrainstate[i]['action']] == 1):
                    cntup.append(i)
                elif(inx[self.onetrainstate[i]['action']] == 2):
                    cntdown.append(i)
                else:
                    cnthold.append(i)
            if(len(cntup)>len(cntdown)):
                if(len(cntup)>len(cnthold)):
                    rj = cntup
                else:
                    rj = cnthold
            elif(len(cntdown)>len(cnthold)):
                rj = cntdown
            else:
                rj = cnthold
            for j in range(len(rj)):
                i = rj[len(rj)-1-j]
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1], self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1], self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                oney = [reward]
                reward = reward*GAMA
                inx[self.onetrainstate[i]['action']].append(onex)
                iny[self.onetrainstate[i]['action']].append(oney)
            print('train fail sample:%d'%samples)
            for i in range(3):
                if(len(inx[i])>0):
                    self.nets[i].train(inx[i], iny[i])
        elif(pad_success):
            reward = 10
            inx = [[],[],[]]
            iny = [[],[],[]]
            cntup = []
            cntdown = []
            cnthold = []
            for i in range(samples):
                if(inx[self.onetrainstate[i]['action']] == 1):
                    cntup.append(i)
                elif(inx[self.onetrainstate[i]['action']] == 2):
                    cntdown.append(i)
                else:
                    cnthold.append(i)
            if(len(cntup)>len(cntdown)):
                if(len(cntup)>len(cnthold)):
                    rj = cntup
                else:
                    rj = cnthold
            elif(len(cntdown)>len(cnthold)):
                rj = cntdown
            else:
                rj = cnthold
            for j in range(len(rj)):
                i = rj[len(rj)-1-j]
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1], self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1], self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                oney = [reward]
                reward = reward*GAMA
                inx[self.onetrainstate[i]['action']].append(onex)
                iny[self.onetrainstate[i]['action']].append(oney)
            print('train success sample:%d'%samples)
            for i in range(3):
                if(len(inx[i])>0):
                    self.nets[i].train(inx[i], iny[i])
        '''
        '''
        if(scorer>0):
            return
            #fail pad
            inx = []
            iny = []
            for i in range(samples):
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1], self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1], self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                oney = [0.7, 0.7, 0.7]
                oney[self.onetrainstate[i]['action']] = 0.1
                inx.append(onex)
                iny.append(oney)
            feed_dict = {self.X:inx, self.Y:iny}
            for i in range(5):
                c, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
            print('one failpad train cost %f' % c)
        elif(pad_success):
            #success pad
            inx = []
            iny = []
            for i in range(samples):
                onex = [self.onetrainstate[i]['ball_pos'][0], self.onetrainstate[i]['ball_pos'][1], self.onetrainstate[i]['ball_vel'][0], self.onetrainstate[i]['ball_vel'][1], self.onetrainstate[i]['paddle1_pos'][0][0], self.onetrainstate[i]['paddle1_pos'][0][1], self.onetrainstate[i]['paddle1_pos'][3][0], self.onetrainstate[i]['paddle1_pos'][3][1]]
                oney = [0.1, 0.1, 0.1]
                oney[self.onetrainstate[i]['action']] = 0.8
                inx.append(onex)
                iny.append(oney)
            feed_dict = {self.X:inx, self.Y:iny}
            print(iny)
            for i in range(5):
                c, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
            print('one successpad train cost %f' % c)
        '''

    def savesample(self, samplesx, samplesy):
        for i in range(len(samplesy)):#3
            for j in range(len(samplesy[i])):
                a = samplesx[i][j]
                a.append(samplesy[i][j][0])
                self.file.writelines(str(a)+'\n')
        print('save one sample')

    def save(self):
        #saver = tf.train.Saver({'W1':self.W1, 'b1':self.b1, 'W2':self.W2, 'b2':self.b2, 'W3':self.W3, 'b3':self.b3})
        #self.saver.save(self.sess, 'model/model-pong')
        self.nets_pos[0].save()
        self.nets_pos[1].save()
        self.nets_pos[2].save()
        self.nets_neg[0].save()
        self.nets_neg[1].save()
        self.nets_neg[2].save()
        self.file.close()
        self.file = open('samples/samples.txt', 'a+')
        print('save model')

    def load(self):
        #saver = tf.train.Saver({'W1':self.W1, 'b1':self.b1, 'W2':self.W2, 'b2':self.b2, 'W3':self.W3, 'b3':self.b3})
        self.saver.restore(self.sess, 'model/model-pong')
        print('load model')

    def initnet(self):
        # input ballposxy, ballvelxy, padupxy, paddownxy
        self.X = tf.placeholder(tf.float32, [None, 8])
        self.Y = tf.placeholder(tf.float32, [None, 3]) #hold up down

        W1 = tf.get_variable("W1", shape=[8, 16],
                     initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([16]))
        L1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)

        W2 = tf.get_variable("W2", shape=[16, 8],
                     initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([8]))
        L2 = tf.nn.relu(tf.matmul(L1, self.W2) + self.b2)

        W3 = tf.get_variable("W3", shape=[8, 3],
                     initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([3]))
        self.hypothesis = tf.matmul(L2, self.W3) + self.b3
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver({'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3})


aiplayer = pongaiplayer()

while(not aiplayer.shutdown):
    #start = time.time()
    aiplayer.action()
    #print('one policy time:%f'%(time.time()-start))

print('ai shutdown')