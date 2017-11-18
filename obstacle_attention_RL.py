import pygame
from avoid_util import *
import random
import time
import numpy as np

sleep_time = 0.1
width, height = 200, 400
Kinds = ["car", "people", "plastic"]
Kinds_Picture = ['images/cycle_blue_10.png', 'images/cycle_blue_10.png', 'images/cycle_blue_10.png']
Kinds_Attention = ['images/cycle_green_10.png', 'images/cycle_green_10.png', 'images/cycle_green_10.png']
NUM = 10
MAX_SPEED = 10
standard_acc = 10
Length = 4


class Obstacle:
    def __init__(self):
        self.picture = pygame.image.load('images/cycle_blue.png')
        self.position_x = 0
        self.position_y = 0
        self.picture_kind_index = 0
        self.kind = ''
        self.drt = 0
        self.vel = 0
        self.turn_drt = 0
        self.acc = 0
        self.init_position()
        self.attention_flag = 0

    def init_position(self):
        if random.random() < 0.5:
            self.position_x = 0
            self.acc = 10
            self.drt = 0
        else:
            self.position_x = width - 20
            self.acc = -10
            self.drt = 180
        self.position_y = random.randint(0, height - 20)
        self.picture_kind_index = random.randint(0, 2)
        self.kind = Kinds[self.picture_kind_index]
        self.picture = pygame.image.load(Kinds_Picture[self.picture_kind_index])
        self.vel = 0
        self.turn_drt = 0

    def go(self):
        if self.position_x != -1:
            if self.vel < 0:
                self.vel = 0
            elif self.vel > MAX_SPEED:
                self.vel = MAX_SPEED
            random_turn = random.random()
            if random_turn < 0.1:
                self.turn_drt = 30
            elif random_turn > 0.9:
                self.turn_drt = -30
            else:
                self.turn_drt = 0
            self.drt += self.turn_drt
            self.drt = self.drt % 360
            self.position_x += self.vel * round(math.cos(self.drt / 180 * math.pi))
            self.position_y += self.vel * round(math.sin(self.drt / 180 * math.pi))
            if self.position_x < 0 or self.position_x > width or self.position_y < 0 or self.position_y > height:
                self.position_x = -1
                self.position_y = -1
            random_acc = random.random()
            if self.vel > 0:
                if random_acc > 0.2:
                    if self.acc == standard_acc:
                        self.acc = 0
                    elif self.acc == 0:
                        if random.random() > 0.5:
                            self.acc = standard_acc
                        else:
                            self.acc = standard_acc
                    elif self.acc == -standard_acc:
                        self.acc = 0
                else:
                    self.acc = self.acc
            else:
                if random.random() > 0.3:
                    self.acc = standard_acc
            self.vel += self.acc
            del random_turn
            del random_acc


class Player:
    def __init__(self):
        self.picture = pygame.image.load('images/cycle_red-10.png')
        self.position_x = 0
        self.position_y = 0
        self.init_position()

    def init_position(self):
        self.position_x = width / 2 - 10
        self.position_y = height - 20

    def go(self):
        self.position_y -= 10
        if self.position_y < 0:
            self.position_y = -1


class AvoidGame:
    def __init__(self, obstacle_num):
        self.player = Player()
        self.obstacle_num = obstacle_num
        self.obstacles = []
        for _ in range(self.obstacle_num):
            temp_obstacle = Obstacle()
            self.obstacles.append(temp_obstacle)
        self.screen = pygame.display.set_mode((width, height))
        self.cnt = 0
        self.attention_flags = np.zeros(obstacle_num)
        self.predict_flags = np.zeros(obstacle_num)
        self.observations = np.zeros([obstacle_num*Length, 2])

    def show_game(self):
        self.screen.fill(0)
        self.cnt += 1
        if 2 <= self.cnt <= 5:
            for i in range(self.obstacle_num):
                self.observations[i * Length + self.cnt - 2, 0] = self.obstacles[i].position_y - self.player.position_y
                self.observations[i * Length + self.cnt - 2, 1] = self.obstacles[i].position_y - self.player.position_y
        if self.cnt == 6:
            for i in range(self.obstacle_num):
                # predict attention
                self.predict_flags[i] = 0
        if self.player.position_y == -1:

            # clear
            for i in range(self.obstacle_num):
                self.predict_flags[i] = 0
                self.attention_flags[i] = 0
            self.cnt = 0
            self.player.init_position()
            for index in range(self.obstacle_num):
                self.obstacles[index].init_position()
        if self.player.position_y != -1:
            self.screen.blit(self.player.picture, (self.player.position_x, self.player.position_y))
        for index in range(self.obstacle_num):
            if self.obstacles[index].position_y != -1:
                self.screen.blit(self.obstacles[index].picture, (self.obstacles[index].position_x,
                                                                 self.obstacles[index].position_y))
        pygame.display.flip()
        time.sleep(sleep_time)

    def play(self):
        for index in range(self.obstacle_num):
            self.obstacles[index].go()
        self.player.go()

    def detect_attention(self):
        for index in range(self.obstacle_num):
            dist = math.sqrt(math.pow(self.obstacles[index].position_x - self.player.position_x, 2)
                             + math.pow(self.obstacles[index].position_y - self.player.position_y, 2))
            if dist < 40:
                self.obstacles[index].picture = pygame.image.load(
                    Kinds_Attention[self.obstacles[index].picture_kind_index])
                self.attention_flags[index] = 1


if __name__ == '__main__':
    pygame.init()
    avoidGame = AvoidGame(NUM)
    # avoidGame.show_game()
    while 1:
        avoidGame.play()
        avoidGame.detect_attention()
        avoidGame.show_game()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
