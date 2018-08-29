# -*- coding: utf-8 -*-
import os
import random
import gym
import time
import numpy as np
from zmq_comm import *
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 10000
TRAIN_MODEL = True

actionlist = ['hold', 'up', 'down']

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        if(TRAIN_MODEL):
            self.epsilon = 0.7  # exploration rate
        else:
            self.epsilon = 0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.1 / EPISODES
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        for i in range(30):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                if(done):
                    target = reward
                else:
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                #print('target')
                target_f = self.model.predict(state)
                #print(target_f,action)
                target_f[0][action] = target
                #print(target_f)
                history = self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

        return history

    def load(self, name):
        if(os.path.exists(name)):
            print('load')
            self.model.load_weights(name)
        else:
            print('no saved file')

    def save(self, name):
        print('save')
        self.model.save_weights(name)

def nextstate(cli):
    res = cli.query('')
    state = [[res['ball_vel'][0], res['ball_vel'][1], res['ball_pos'][0], res['ball_pos'][1], res['paddle1_pos'][0][0], res['paddle1_pos'][0][1], res['paddle1_pos'][3][0], res['paddle1_pos'][3][1]]]
    state = np.array(state)
    return state, res['scorelr']

def train():
    pongcli = zmq_comm_cli_c(name='pong',port=1201)
    state_size = 8
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/pong-dqn.h5")
    done = False
    batch_size = 32
    lastscore = [0,0]
    history = None
    avescore = 0
    for e in range(EPISODES):
        pongcli.reset()
        state,score = nextstate(pongcli)
        #lastscore = [score[0],score[1]]
        done = False
        #print(state,type(state),state.shape)
        for tt in range(500):
            action = agent.act(state)
            pongcli.execute({'ai':actionlist[action]})
            time.sleep(0.05)
            next_state,score = nextstate(pongcli)
            reward = float(tt)/10 + (score[0])*10 - (score[1])*10
            if(score[0] > 0 or score[1]>0):
                done = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            #lastscore = [score[0],score[1]]
            if done:
                print("episode: {}/{}, time: {}, score: {:2}, epsilon: {:.4}"
                      .format(e, EPISODES, tt, reward, agent.epsilon))
                avescore += reward/10
                if len(agent.memory) > batch_size:
                    history = agent.replay(batch_size)
                if(not history == None):
                    print('training loss=%f'%history.history['loss'][0])
                break
            

        if e % 10 == 0:
            print('past 10 episode average score is {:3}'.format(avescore))
            avescore = 0
            agent.save("./save/pong-dqn.h5")

def test():
    pongcli = zmq_comm_cli_c(name='pong',port=1201)
    state_size = 8
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/pong-dqn.h5")
    done = False
    batch_size = 32
    lastscore = [0,0]
    history = None
    avescore = 0
    tt = 0
    while True:
        state,score = nextstate(pongcli)
        action = agent.act(state)
        pongcli.execute({'ai':actionlist[action]})
        tt += 1
        if(score[0] > 0 or score[1]>0):
            print("this test work frames(20fps): {}, epsilon: {:.4}".format(tt,agent.epsilon))
            tt = 0
        time.sleep(0.05)

if __name__ == "__main__":    
    if(TRAIN_MODEL):
        train()
    else:
        test()