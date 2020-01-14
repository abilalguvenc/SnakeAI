import numpy as np
import pygame
import os
from Snake import Snake
from Agent import Agent, loadAgent, saveAgent
import matplotlib.pyplot as plt 
import statistics
  
_image_library = {}
def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
            canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
            image = pygame.image.load(canonicalized_path)
            _image_library[path] = image
    return image

def trainAgent(n_games, game_records):
    agent = Agent(gamma = 0.9, epsilon = 1.0, alpha = 0.0005, input_dims = 16,
                  n_actions = 3, mem_size = 100000, batch_size = 512, epsilon_end = 0.01)
    scores = []
    i = 0
    best_score = 15
    best_score_avg = 0
    best_records = []

    for j in range(n_games):
        snake = Snake(10,10)
        done = False
        score = 0
        curr_state = snake.getState()
        game_record = []
        while not done:
            game_record.append(snake.board.copy())
            action = agent.choose_action(curr_state)
            eaten = snake.doStep(action)
            reward = 0
            if eaten:               
                if snake.isDone:
                    reward = 1000
                else:
                    score += 1
                    reward = 10
            if snake.isDead:
                reward = -10
                if(snake.isCaged()): 
                    reward = -1
            new_state = snake.getState()
            if snake.isDead or snake.isDone:
                done = True
            else:
                done = False
            agent.store(curr_state, action, reward, new_state, done)
            curr_state = new_state
            agent.learn()

        game_record.append(snake.board.copy())
        game_records.append(game_record.copy()) 
        scores.append(score)

        avg_score = np.mean(scores[0:i+1])
        print('Game %3d Score %3d\tAverage Score %.2f' %(i, score, avg_score))
        
        if(score>(best_score-6)):
            new_records = []
            new_score_avg = agentPlay(agent, new_records, 10, True)
            print("\tNew Record Avg:", new_score_avg)
            if(new_score_avg>best_score_avg):
                best_records = new_records.copy()
                best_score = score
                best_score_avg = new_score_avg
                saveAgent(agent, "agents/agent_"+str(n_games)+".obj", True)
        i+=1
    if(best_score != 149.5):
        for i in range(10):
            game_records.append(best_records[i].copy())
    else:
        saveAgent(agent, "agents/agent_"+str(n_games)+".obj", False)
    
def printAscendingScores(game_records):
    print("\n")
    hi_score, score_sum = -1, 0
    game_id, game_score = [], []
    for i in range(len(game_records)):
        new_score = np.max(game_records[i][len(game_records[i])-1]) - 1
        game_id.append(i)
        game_score.append(new_score)
        score_sum += new_score
        if (hi_score < new_score):
            hi_score = new_score
            print("Game:%3d - Score: %d" %(i, hi_score))
    print("Standard Deviation:", statistics.stdev(game_score))
    plt.plot(game_id, game_score) 
    plt.xlabel('Game') 
    plt.ylabel('Score') 
    plt.title('Average Score: %.2f' %(score_sum/len(game_records)))
    plt.show() 
    

def agentPlay(agent, game_records, n_games, useTab):
    scores = []
    i = 0

    for i in range(n_games):
        snake = Snake(10,10)
        done = False
        score = 0
        curr_state = snake.getState()
        game_record = []
        while not done:
            game_record.append(snake.board.copy())
            action = agent.choose_action(curr_state)
            eaten = snake.doStep(action)
            if eaten:               
                if snake.isDone:
                    score = 1000
                else:
                    score += 1
                
            new_state = snake.getState()
            if snake.isDead or snake.isDone:
                done = True
            else:
                done = False
            curr_state = new_state

        game_record.append(snake.board.copy())
        game_records.append(game_record.copy()) 
        scores.append(score)

        avg_score = np.mean(scores[0:i+1])
        if useTab:
            print('\tGame %3d Score %3d\tAverage Score %.2f' %(i, score, avg_score))
        else:
            print('Game %3d Score %3d\tAverage Score %.2f' %(i, score, avg_score))
        i+=1
    return avg_score

def main():
    game_records = [] 

    print("Select your choice\n1. Load Agent\n2. Train Agent")
    if(input("Choice: ") == "1"):
        agent_no = input("Select Agent ID: agent_")
        agent = loadAgent("agents/agent_"+agent_no+".obj")
        agentPlay(agent, game_records, 51, False)
    else:
        n_games = int(input("Game Limit: "))
        trainAgent(n_games, game_records)
        
    printAscendingScores(game_records)
    
    ###########################################################################################################
    
    pygame.init()
    dis = pygame.display.set_mode((1050, 640))
    pygame.display.set_caption('Snake AI')
    dis.blit(get_image('images/background.png'), (0, 0))
    clock = pygame.time.Clock()

    unitSize, xBeg, yBeg = 30, 59, 244
    gen, genMax, step, speed = 0, len(game_records), 0, 10
    game_over, game_pause = False, False

    white       = (255, 255, 255)
    aqua        = ( 77, 113, 138)
    colorHead   = (254, 212,   1)
    colorTail   = (176, 145,   0)
    colorFly    = (255,   0,  48)

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
                        else:
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