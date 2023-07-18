import pygame
import neat
import time
import os
import random
pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png"))) #load in images
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMGS = BIRD_IMGS 
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5 #time to loop through all images
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0 #frames per second
        self.vel = 0 
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5 #0,0 is top left in pygame so moving in -ve direction is up
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count +=1

        d = self.vel * self.tick_count + 1.5*self.tick_count**2 #bird arc

        if d >= 16: #terminal vel
            d = 16

        if d < 0:
            d -= 2 #falling

        self.y = d + self.y

        if d < 0 or self.y < self.height + 50: #used to calculate rotation of the image
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]

        elif self.img_count < self.ANIMATION_TIME*2: #loop through the image animation every 5 seconds
            self.img = self.IMGS[1]

        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]

        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]

        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rectangle = rotated_image.get_rect(center=self.img.get_rect(topleft= (self.x,self.y)).center) #rotates image around the center

        win.blit(rotated_image, new_rectangle.topleft)     

    def get_mask(self):
        return pygame.mask.from_surface(self.img) #used for collisions


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self,x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,450) # spawn the pipes at different heights
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_pipe_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_pipe_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_pipe_mask, bottom_offset)
        t_point = bird_mask.overlap(top_pipe_mask, top_offset)

        if t_point or b_point:
            return True
        else:
            return False

class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH #Cycle the two images to look like a continuous image


        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG,(0,0))
    for bird in birds:
        bird.draw(win)
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))
    base.draw(win)

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    birds = []
    global GEN
    GEN += 1

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config) #for every genome attach a network and bird and corresponding fitness to it
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)


    base = Base(730)
    pipes = [Pipe(random.randrange(500,800))] #randomise pipe distances
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    run = True
    clock = pygame.time.Clock()
    score = 0
    while(run):
        clock.tick(1000) #60 times per second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        pipe_ind = 0
        if len(birds) > 0: #if the bird has passed the first pipe look at the next pipe in the list
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.015 #add fitness to the bird for every x it moves

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):

                if pipe.collide(bird): #if the bird collides with a pipe then remove it from the list and network
                    ge[x].fitness -= 0.1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width()< 0: #if the pipe has been passed then remove it from the list of pipes
                rem.append(pipe)
            pipe.move()

        if add_pipe:
            score += 1 #every time a pipe is passed increment the score and fitness
            for g in ge:
                g.fitness += 0.5

            pipes.append(Pipe(random.randrange(500,800))) #add a new pipe

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0: #if a bird hits the ground remove it from the list
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win,birds, pipes, base, score, GEN) #draw the game


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
