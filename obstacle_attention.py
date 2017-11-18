import pygame
from avoid_util import *
import random
import time


sleep_time = 0.1
width, height = 400, 800
Kinds = ["car", "people", "plastic"]
Kinds_Picture = ['images/cycle_blue.png', 'images/star_blue.png', 'images/triangle_blue.png']
Kinds_Attention = ['images/cycle_green.png', 'images/star_green.png', 'images/triangle_green.png']
NUM = 20


class Obstacle:
    def __init__(self):
        self.picture = pygame.image.load('images/cycle_blue.png')
        self.position_x = 0
        self.position_y = 0
        self.picture_kind_index = 0
        self.kind = ''
        self.init_position()
                
    def init_position(self):
        if random.random() < 0.5:
            self.position_x = 0
        else:
            self.position_x = width-20
        self.position_y = random.randint(0, height-20)
        self.picture_kind_index = random.randint(0, 2)
        self.kind = Kinds[self.picture_kind_index]
        self.picture = pygame.image.load(Kinds_Picture[self.picture_kind_index])


class Player:
    def __init__(self):
        self.picture = pygame.image.load('images/cycle_red-10.png')
        self.position_x = 0
        self.position_y = 0
        self.init_position()

    def init_position(self):
        self.position_x = width / 2 - 10
        self.position_y = height - 20


class AvoidGame:
    def __init__(self, obstacle_num):
        self.player = Player()
        self.obstacle_num = obstacle_num
        self.obstacles = []
        for _ in range(self.obstacle_num):
            temp_obstacle = Obstacle()
            self.obstacles.append(temp_obstacle)
        self.screen = pygame.display.set_mode((width, height))
    
    def show_game(self):
        self.screen.fill(0)
        self.screen.blit(self.player.picture, (self.player.position_x, self.player.position_y))
        for index in range(self.obstacle_num):
            self.screen.blit(self.obstacles[index].picture, (self.obstacles[index].position_x,
                                                             self.obstacles[index].position_y))
        pygame.display.flip()
        time.sleep(sleep_time)

    def obstacle_go(self):
        for index in range(self.obstacle_num):
            temp_x, temp_y = obstacle_rand_go()
            speed = random.randint(1, 3)
            self.obstacles[index].position_x += temp_x*speed
            self.obstacles[index].position_y += temp_y*speed
            if self.obstacles[index].position_x < 0 or self.obstacles[index].position_x > width-20 \
                    or self.obstacles[index].position_y < 0 or self.obstacles[index].position_y > height-20:
                self.obstacles[index].init_position()

    def player_go(self):
        self.player.position_y -= 20
        if self.player.position_y < 0:
            self.player.init_position()

    def detect_attention(self):
        for index in range(self.obstacle_num):
            dist = math.sqrt(math.pow(self.obstacles[index].position_x - self.player.position_x, 2)
                             + math.pow(self.obstacles[index].position_y - self.player.position_y, 2))
            if self.obstacles[index].kind == 'people' and dist < 40:
                self.obstacles[index].picture = pygame.image.load(
                    Kinds_Attention[self.obstacles[index].picture_kind_index])
            elif self.obstacles[index].kind == 'car' and dist < 80:
                self.obstacles[index].picture = pygame.image.load(
                    Kinds_Attention[self.obstacles[index].picture_kind_index])
            else:
                self.obstacles[index].picture = pygame.image.load(
                    Kinds_Picture[self.obstacles[index].picture_kind_index])


if __name__ == '__main__':
    pygame.init()
    avoidGame = AvoidGame(NUM)
    avoidGame.show_game()
    while 1:
        avoidGame.player_go()
        avoidGame.obstacle_go()
        avoidGame.detect_attention()
        avoidGame.show_game()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
