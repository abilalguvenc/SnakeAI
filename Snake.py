import numpy as np
import random

class Snake():
    def __init__(self,r,c):
        self.row, self.col = r, c
        self.goingRight, self.goingLeft, self.goingUp, self.goingDown = True, False, False, False
        self.isDead, self.isDone = False, False
        self.head_x, self.head_y = r//2, c//2
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
        tail_x, tail_y, eaten = 0, 0, False
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j] > tailMax:
                    tail_x = i
                    tail_y = j
                    tailMax = self.board[i][j]
                if self.board[i][j] > 0:
                    self.board[i][j] += 1
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
        food_row, body_row = -1, -1
        i, j = self.head_x - 1, self.head_y

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
        food_row, body_row = -1, -1
        
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

    def isCaged(self):
        if(self.head_x>0):
            up = self.board[self.head_x -1][self.head_y]
            if (up == 0 or up == -1): return False
        if(self.head_x<self.row-1):
            down = self.board[self.head_x +1][self.head_y]
            if (down == 0 or down == -1): return False
        if(self.head_y>0):
            left = self.board[self.head_x][self.head_y-1]
            if (left == 0 or left == -1): return False
        if(self.head_y<self.col-1):
            right = self.board[self.head_x][self.head_y+1]
            if (right == 0 or right == -1): return False
        return True

    def printBoard(self):
        for i in range(self.row):
            print(self.board[i])
