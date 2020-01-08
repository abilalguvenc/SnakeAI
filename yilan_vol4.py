import random
import numpy as np
import time
from keras.layers import Dense, Activation,Dropout,Softmax
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import pickle
import pygame
import os

_image_library = {}
def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
            canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
            image = pygame.image.load(canonicalized_path)
            _image_library[path] = image
    return image

class ReplayBuffer():
    def __init__(self,max_size,input_shape,n_actions,discrete):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype = dtype)
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def saveStateMemory(self):        
        f = open("memory_header.txt","w")
        f.write(str(self.mem_size)+"\n")
        f.write(str(self.mem_cntr)+"\n")
        f.write(str(self.input_shape)+"\n")
        f.write(str(self.discrete)+"\n")
        f.close()
        np.savetxt(fname="state.txt",X=self.state_memory,fmt="%i")
        np.savetxt(fname="state_new.txt",X=self.new_state_memory,fmt="%i")
        np.savetxt(fname="action.txt",X=self.action_memory,fmt="%i")
        np.savetxt(fname="reward.txt",X=self.reward_memory,fmt="%i")
        np.savetxt(fname="terminal.txt",X=self.terminal_memory)
                    

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states,actions,rewards,states_,terminal

def build_dqn(lr, n_actions, input_dims, fcl_dims, fc2_dims):
    model = Sequential([
    Dense(fcl_dims, input_shape=(input_dims,)),
    Activation('relu'),
    Dense(fc2_dims),
    Activation('relu'),
    Dense(n_actions)])
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model

class Agent(object):
    def __init__(self,alpha,gamma,n_actions,epsilon,batch_size,
                 input_dims,epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size,input_dims,n_actions,discrete = True)
        self.q_eval = build_dqn(alpha,n_actions,input_dims,128,128)

    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def choose_action(self,state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand< self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                               self.memory.sample_buffer(self.batch_size)
    
        action_values = np.array(self.action_space, dtype = np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        q_target[batch_index, action_indices] = reward + \
                                                self.gamma*np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state,q_target,verbose=0)
        
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min
    
    def save_model(self):
        self.q_eval.save(self.model_file)
        self.memory.saveStateMemory()
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)
        self.memory = loadStateMemory()

def loadStateMemory():
    f = open("memory_header.txt","r")
    satirlar = f.readlines()
    f.close()
    size = int(satirlar[0])
    cntr = int(satirlar[1])
    in_shape = int(satirlar[2])
    if "True" in satirlar[3]:
        dc = True
    else:
        dc = False
    memory = ReplayBuffer(max_size = size,input_shape = in_shape,n_actions=3,discrete = dc)
    memory.state_memory = np.loadtxt(fname="state.txt",dtype='int')
    memory.new_state_memory = np.loadtxt(fname="state_new.txt",dtype='int')
    memory.action_memory = np.loadtxt(fname="action.txt",dtype='int')
    memory.reward_memory = np.loadtxt(fname="reward.txt",dtype='int')
    memory.terminal_memory = np.loadtxt(fname="terminal.txt",dtype='float')
    memory.mem_cntr = cntr
    return memory

############################   

class Snake():

    def __init__(self,r,c):
        self.row = r
        self.col = c
        
        self.goingRight = True
        self.goingLeft = False
        self.goingUp = False
        self.goingDown = False
        self.isDead = False
        self.isDone = False

        self.head_x = r//2
        self.head_y = c//2

        self.gameScore = 0
        self.board = np.zeros((r,c))
        self.board[r//2][c//2] = 1
        self.putFood()

    def doStep(self,action):
        #düz gidiyorsa yönlerin değişmesine gerek yok
        if action == 1:  #mevcut yonden saga
            if self.goingRight:
                self.goingRight = False
                self.goingDown = True
            elif self.goingLeft:
                self.goingLeft = False
                self.goingUp = True
            elif self.goingUp:
                self.goingUp = False
                self.goingRight = True
            else:
                self.goingDown = False
                self.goingLeft = True

        elif action == 2:  #mevcut yonden sola
            if self.goingUp:
                self.goingUp = False
                self.goingLeft = True
            elif self.goingRight:
                self.goingRight = False
                self.goingUp = True
            elif self.goingLeft:
                self.goingLeft = False
                self.goingDown = True
            else:
                self.goingDown = False
                self.goingRight = True
        eaten = self.updatePosition()
        return eaten

    def updatePosition(self):
        if self.goingUp:
            if self.head_x == 0:
                self.isDead = True
                return False
            else:
                self.head_x -= 1
        elif self.goingDown:
            if self.head_x == self.row-1:
                self.isDead = True
                return False
            else:
                self.head_x += 1
        elif self.goingRight:
            if self.head_y == self.col-1:
                self.isDead = True
                return False
            else:
                self.head_y += 1
        else:
            if self.head_y == 0:
                self.isDead = True
                return False
            else:
                self.head_y -= 1
        if self.board[self.head_x][self.head_y] > 0:
            self.isDead = True
            return False

        tailMax = self.board[0][0]
        tail_x = 0
        tail_y = 0
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j] > tailMax:
                    tail_x = i
                    tail_y = j
                    tailMax = self.board[i][j]
                if self.board[i][j] > 0:
                    self.board[i][j] += 1
        eaten = False
        if self.board[self.head_x][self.head_y] == -1:
            eaten = True
            self.gameScore += 1
            if self.gameScore < self.row*self.col-1:
                self.putFood()
            else:
                self.isDone = True 
        if not eaten:
            self.board[tail_x][tail_y] = 0
        self.board[self.head_x][self.head_y] = 1
        
        return eaten

    def putFood(self):
        if not self.gameScore == self.row*self.col:            
            boundary = (self.row*self.col) - self.gameScore - 1
            foodDistance = random.randint(1,boundary)
            count = 0
            for i in range(self.row):
                for j in range(self.col):
                    if self.board[i][j] == 0:
                        count += 1
                    if count == foodDistance:
                        self.board[i][j] = -1
                        return
                
        
        
    def getState(self):
        vector = []
        
        vector.append(int(self.goingRight))
        vector.append(int(self.goingLeft))
        vector.append(int(self.goingUp))
        vector.append(int(self.goingDown))

        
        bodyNotFound = True
        food_row = -1
        body_row = -1
        
        
        i = self.head_x - 1
        j = self.head_y

        while i!=-1:
            if self.board[i][j] == -1:
                food_row = i
            if self.board[i][j] > 0 and bodyNotFound:
                body_row = i
                bodyNotFound = False
            i-=1
        if bodyNotFound:
            vector.append(1)
        else:
            vector.append((self.head_x - body_row) / self.row)
        if food_row == -1:
            vector.append(1)
        else:
            vector.append((self.head_x-food_row) / self.row) 
        vector.append(self.head_x/self.row) #tahtanın en üstüne olan uzaklık

        i = self.head_x + 1
        bodyNotFound = True
        food_row = -1
        body_row = -1
        
        while i != self.row:
            if self.board[i][j] == -1:
                food_row = i
            if self.board[i][j] > 0 and bodyNotFound:
                body_row = i
                bodyNotFound = False
            i += 1
        
        if bodyNotFound:
            vector.append(1)
        else:
            vector.append((body_row - self.head_x) / self.row)
        if food_row == -1:
            vector.append(1)
        else:
            vector.append((food_row - self.head_x) / self.row)
        vector.append((self.row - self.head_x)/self.row)

        i = self.head_x
        j = self.head_y-1
        bodyNotFound = True
        body_col = -1
        food_col = -1
        while j!=-1:
            if self.board[i][j] == -1:
                food_col = j
            if self.board[i][j] > 0 and bodyNotFound:
                body_col = j
                bodyNotFound = False
            j -= 1
        if bodyNotFound:
            vector.append(1)
        else:
            vector.append((self.head_y - body_col) / self.col)
        if food_col == -1:
            vector.append(1)
        else:
            vector.append((self.head_y-food_col) / self.col) 
        vector.append(self.head_y/self.col)

        i = self.head_x
        j = self.head_y+1
        bodyNotFound = True
        body_col = -1
        food_col = -1
        while j!=self.col:
            if self.board[i][j] == -1:
                food_col = j
            if self.board[i][j] > 0 and bodyNotFound:
                body_col = j
                bodyNotFound = False
            j += 1
            
        if bodyNotFound:
            vector.append(1)
        else:
            vector.append((body_col - self.head_y) / self.col)
        if food_col == -1:
            vector.append(1)
        else:
            vector.append((food_col - self.head_y) / self.col) 
        vector.append((self.col - self.head_y)/self.col)
        return np.asarray(vector)

    def printBoard(self):
        for i in range(self.row):
            print(self.board[i])


def main():
    n_games = 51 
    game_records = [] 
    agent = Agent(gamma = 0.9, epsilon = 1.0, alpha = 0.0005, input_dims = 16,
                  n_actions = 3, mem_size = 100000, batch_size = 512, epsilon_end = 0.01)
    #agent.load_model()
    #f = open("ajan.dat","rb")
    #agent = pickle.load(f)
    #f.close()
    scores = []
    t_ilk = time.time() 
    i=0
    #print(t_end)
    for j in range(n_games):
        snake = Snake(10,10)
        done = False
        score = 0
        curr_state = snake.getState()
##        f = open(str(j)+".txt","w")
##        for satir in snake.board:
##            f.write(str(satir))
##            f.write("\n")
        game_record = []
        while not done:
            game_record.append(snake.board.copy())
            action = agent.choose_action(curr_state)
            eaten = snake.doStep(action)
##            f.write("\n")
##            for satir in snake.board:
##                f.write(str(satir))
##                f.write("\n")
            reward = 0
            if eaten:               
                if snake.isDone:
                    reward = 1000
                else:
                    reward = 10
            if snake.isDead:
                reward = -10
            new_state = snake.getState()
            if snake.isDead or snake.isDone:
                done = True
            else:
                done = False
            score += reward
            agent.remember(curr_state, action, reward, new_state, done)
            curr_state = new_state
            agent.learn()
        #f.close()

        game_record.append(snake.board.copy())
        game_records.append(game_record.copy()) 
        scores.append(score)

        avg_score = np.mean(scores[0:i+1])
        #if i%100 == 0:
        print('epsiode ', i, 'score %.2f' % score,'average score %.2f' %avg_score)
        i+=1
        #print(time.time())
        

    t_son = time.time()
    f = open("ajan.obj","wb")
    pickle.dump(agent,f)
    f.close()
    
    pygame.init()
    dis = pygame.display.set_mode((1050, 640))
    pygame.display.set_caption('Snake AI')
    dis.blit(get_image('background.png'), (0, 0))
    clock = pygame.time.Clock()

    unitSize, xBeg, yBeg = 30, 59, 244
    gen, genMax, step, speed = 0, len(game_records), 0, 10
    game_over, game_pause = False, False

    white = (255,255,255)
    aqua = (77,113,138)
    colorHead = (254,212,1)
    colorTail = (176,145,0)
    colorFly = (255,0,48)

    font = pygame.font.Font('freesansbold.ttf', 32) 

    textGen, textStep = font.render(" ", True, colorHead, aqua), font.render(" ", True, colorHead, aqua) 
    rectGen, rectStep = textGen.get_rect(), textStep.get_rect()  
    rectGen.center, rectStep.center = (180, 149), (180, 187) 

    textGen = font.render(str(gen)+"/"+str(genMax-1), True, colorHead, aqua)

    dis.blit(textStep, rectStep)
    dis.blit(textGen, rectGen)
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gen, step = gen-1, 0
                    if (gen == -1): gen = genMax-1
                    textGen = font.render(str(gen)+"/"+str(genMax-1)+"   ", True, colorHead, aqua)
                    dis.blit(textGen, rectGen)
                    game_pause = False
                elif event.key == pygame.K_RIGHT:
                    gen, step = gen+1, 0
                    if (gen == genMax): gen = 0
                    textGen = font.render(str(gen)+"/"+str(genMax-1)+"   ", True, colorHead, aqua) 
                    dis.blit(textGen, rectGen)
                    game_pause = False
                elif event.key == pygame.K_m:
                    gen, step = genMax-1, 0
                    textGen = font.render(str(gen)+"/"+str(genMax-1)+"   ", True, colorHead, aqua) 
                    dis.blit(textGen, rectGen)
                    game_pause = False
                elif event.key == pygame.K_UP:
                    speed *= 2
                elif event.key == pygame.K_DOWN:
                    speed /= 2
                elif event.key == pygame.K_ESCAPE:
                    game_over = True
                elif event.key == pygame.K_SPACE:
                    if game_pause:
                        game_pause = False
                    else:
                        game_pause = True 
        if not game_pause:
            score = 0
            pygame.draw.rect(dis, white, [xBeg-1, yBeg-1, 302, 302])
            for i in range(10):
                for j in range(10):
                    val = game_records[gen][step][i][j]
                    if(val > 1): 
                        score += 1
                        if((i<9) and (game_records[gen][step][i+1][j] == val-1)):
                           pygame.draw.rect(dis, colorTail, [j*unitSize+xBeg+1, i*unitSize+yBeg+1, unitSize-2, unitSize])
                        elif((i>0) and (game_records[gen][step][i-1][j] == val-1)):
                           pygame.draw.rect(dis, colorTail, [j*unitSize+xBeg+1, i*unitSize+yBeg-1, unitSize-2, unitSize])
                        elif((j<9) and (game_records[gen][step][i][j+1] == val-1)):
                           pygame.draw.rect(dis, colorTail, [j*unitSize+xBeg+1, i*unitSize+yBeg+1, unitSize, unitSize-2])
                        elif((j>0) and (game_records[gen][step][i][j-1] == val-1)):
                           pygame.draw.rect(dis, colorTail, [j*unitSize+xBeg-1, i*unitSize+yBeg+1, unitSize, unitSize-2])
                    elif(val == 1):
                        xHead, yHead = j, i
                    elif(val == -1):
                        pygame.draw.rect(dis, colorFly, [j*unitSize+xBeg+1, i*unitSize+yBeg+1, unitSize-2, unitSize-2])
                        
            pygame.draw.rect(dis, colorHead, [xHead*unitSize+xBeg-1, yHead*unitSize+yBeg-1, unitSize+2, unitSize+2])
            textStep = font.render(str(step)+"     ", True, colorHead, aqua) 
            dis.blit(textStep, rectStep)

            step += 1
            if(step == len(game_records[gen])):
                game_pause = True
                step = 0
                textScore = font.render("Score "+str(score), True, colorFly, (255,220,220))
                rectScore = textScore.get_rect()
                rectScore.center = (xBeg+150, yBeg+30)
                dis.blit(textScore, rectScore)
        
        pygame.display.update()
        
        clock.tick(speed)
    
    pygame.quit()
    quit()

main()



        
