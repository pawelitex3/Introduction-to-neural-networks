import numpy as np
import matplotlib.pyplot as plt
import pygame

pygame.init()

class Kohonen(object):
  def __init__(self, h, w, dim):
    self.shape = (h, w, dim)
    self.som = np.random.random(self.shape) # Zadanie: np.random.random
    #self.som = np.zeros(self.shape)
    self.data = []
    self.alpha0 = 1.5
    #self.sigma = 2.5
    self.sigma1 = 2.495
    self.sigma2 = 2.5
    self.beta = 2500
    self.steps = 3000

  def train(self, data):

    self.data = np.copy(data)
    index = np.random.choice(range(len(self.data)))
    best_neuron = self.find_best_neuron(self.data[index])
    self.update_som(best_neuron, self.data[index], t)

  def find_best_neuron(self, input_vector):
    list_neurons = []
    for x in range(self.shape[0]):
      for y in range(self.shape[1]):
        dist = np.linalg.norm(input_vector - self.som[x, y])
        list_neurons.append(((x, y), dist))

    list_neurons.sort(key=lambda x: x[1])
    return list_neurons[0][0]

  def update_som(self, bn, dp, t):
    for x in range(self.shape[0]):
      for y in range(self.shape[1]):
        dist_to_bn = np.linalg.norm(np.array(bn) - np.array([x, y])) 
        self.update_cell((x, y), dp, t, dist_to_bn)

  def update_cell(self, cell, dp, t, dist_to_bn):
    self.som[cell] += self.alpha(t) * self.G(dist_to_bn) * (dp - self.som[cell])

  def G(self, rho):
    #return np.exp(-rho**2/(2*self.sigma**2))
    return 2*np.exp(-rho**2 / (2*self.sigma1**2)) - np.exp(-rho**2 / (2*self.sigma2**2))

  def alpha(self, t):
    return self.alpha0 * np.exp(-t/self.beta)

  def draw(self, screen):
      lista = []
      for x in range(self.shape[0]):
          for y in range(self.shape[1]):
              lista.append(self.som[x, y])
              if x < self.shape[0]-1:
                  pygame.draw.line(window, (255,255,255), self.som[x, y] , self.som[x+1, y])
              if y < self.shape[1]-1:
                  pygame.draw.line(window, (255,255,255), self.som[x, y] , self.som[x, y+1])
              
              
              pygame.draw.circle(screen, (0, 204, 204), self.som[x, y], 5)

class Button(object):
    def __init__(self, x, y, width=30, height=30, text='', version=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def was_clicked(self, x, y):
        return (self.x < x < self.x+self.width) and (self.y < y < self.y+self.height)


def create_buttons(buttons):
    row = []
    row.append(Button(10, 370, 80, 30, 'run'))
    row.append(Button(100, 370, 80, 30, 'clear'))
    row.append(Button(190, 370, 80, 30, 'circle'))
    row.append(Button(280, 370, 80, 30, 'square'))
    buttons.append(row)

def draw_buttons(buttons, screen):
    for row in buttons:
        for button in row:
            pygame.draw.rect(screen, (180, 180, 180), (button.x, button.y, button.width, button.height), 0)
            font = pygame.font.SysFont('verdana', 20)
            text = font.render(button.text, 1, (0, 0, 0))
            x = button.x + button.width/2 - text.get_width()/2
            y = button.y + button.height/2 - text.get_height()/2
            screen.blit(text, (x, y))


def rotate(x, y, r):
    rx = (x*np.cos(r)) - (y*np.sin(r))
    ry = (y*np.cos(r)) + (x*np.sin(r))
    return (rx, ry)


def draw_circle(screen, points):
    center = (185, 185)
    radius = 150
    num_points = 300
    arc = (2 * np.pi) / num_points
    for p in range(num_points):
        (px,py) = rotate(0, radius, arc * p) 
        px += center[0]
        py += center[1]
        points.append((px,py))

    for point in points:
        pygame.draw.circle(screen, (160, 50, 50), (point[0], point[1]), 5)


def draw_square(screen, points):
    for i in range(96):
        if i==0 or i == 95:
            for j in range(96):
                points.append((50+j*3, 50+3*i))
        else:
            points.append((50, 50+3*i))
            points.append((335, 50+3*i))
    
    for point in points:
        pygame.draw.circle(screen, (160, 50, 50), (point[0], point[1]), 5)


def draw_points(screen, points):
    for point in points:
        pygame.draw.circle(screen, (160, 50, 50), (point[0], point[1]), 5)


window_size = width, height = 370, 410
window = pygame.display.set_mode(window_size)
window.fill((0, 0, 0))
buttons = []

create_buttons(buttons)
draw_buttons(buttons, window)
pygame.display.flip()
running = True

points = []

while running:
    for event in pygame.event.get():
        mouse = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if mouse[1] < 370:
                window.fill((0, 0, 0))
                points.append((mouse[0], mouse[1]))
                draw_buttons(buttons, window)
                draw_points(window, points)
                pygame.display.flip()
            else:
                for row in buttons:
                    for button in row:
                        if button.was_clicked(mouse[0], mouse[1]):
                            if button.text == 'run':
                                k = Kohonen(20, 20, 2)
                                #k.som += [width/2, height/2]
                                k.som *= [width, width]
                                for t in range(5000):
                                    window.fill((0, 0, 0))
                                    k.train(points)
                                    k.draw(window)
                                    draw_buttons(buttons, window)
                                    draw_points(window, points)
                                    pygame.display.flip()
                            elif button.text == 'clear':
                                points = []
                                window.fill((0, 0, 0))
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'circle':
                                points = []
                                window.fill((0, 0, 0))
                                draw_circle(window, points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'square':
                                points = []
                                window.fill((0, 0, 0))
                                draw_square(window, points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
            
            

