import numpy as np
import cv2
import pygame
import random

pygame.init()
numbers = [str(i+1) for i in range(20)]

class HopfieldNewtork(object):
  def __init__(self, N):
    self.N = N
    self.weights = np.zeros((self.N, self.N))
    self.iterations = 1000

  def little_dynamics(self, state):
    #Zadanie domowe!
    return state

  def glauber_dynamics(self, state):
    i = np.random.randint(0, self.N**2)
    a = np.matmul(self.weights[i, :], state)
    if a < 0:
      state[i] = -1
    else:
      state[i] = 1
    return state

  def run_dynamics(self, state, dtype='glauber'):
    for i in range(self.iterations):
      if dtype == 'glauber':
        state = self.glauber_dynamics(state)
      else:
        state = self.little_dynamics(state)

  def training(self, training_data):
    P = len(training_data)
    for i in range(self.N):
      for j in range(self.N):
        if i != j:
          for p in range(P):
            self.weights[i][j] += (training_data[p][i] * training_data[p][j])
    self.weights /= self.N

  def hamming(self, state, training_data):
    P = len(training_data)
    hamming_distance = np.zeros((self.iterations, P))
    for iteration in range(self.iterations):
      for i in range(self.N**2):
        a[i] = 0
        for j in range(self.N**2):
          a[i] += self.weights[i,j] * state[p, j]
      state = np.where(a < 0, -1, 1)
      for p in range(self.P):
        hamming_distance[iteration, p] = (state - training_data[p] != 0).sum()


class Button(object):
    def __init__(self, x, y, width=30, height=30, text='', version=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def was_clicked(self, x, y):
        return (self.x < x < self.x+self.width) and (self.y < y < self.y+self.height)


class Point(object):
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


def create_buttons(buttons):
    row = []
    row.append(Button(10, 620, 80, 30, 'run'))
    row.append(Button(100, 620, 80, 30, 'clear'))
    row.append(Button(190, 620, 80, 30, 'noise'))
    row.append(Button(280, 620, 80, 30, 'square'))
    for i in range(10):
        row.append(Button(10+i*60, 660, 50, 30, str(i+1)))

    for i in range(10):
        row.append(Button(10+i*60, 700, 50, 30, str(i+11)))

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


def create_points(points, image):
    image_copy = np.copy(image)
    for i in range(10000):
        points.append(Point(10+(i%100)*6, 10+(i//100)*6, (image_copy[i], image_copy[i], image_copy[i])))


def draw_image(screen, points):
    for i in range(10000):
        if points[i].color == (0, 0, 0):
            pygame.draw.rect(screen, np.array(points[i].color)+70, (points[i].x, points[i].y, 4, 4), 0)
        else:
            pygame.draw.rect(screen, np.array(points[i].color), (points[i].x, points[i].y, 4, 4), 0)


window_size = width, height = 620, 740
window = pygame.display.set_mode(window_size)
window.fill((0, 0, 0))

white = []
for i in range(10000):
    white.append(255)


images = []
for i in range(2):
    filename = str(i+1) + ".png"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #print(np.ravel(np.array(binary_img))/255)
    images.append(np.ravel(np.array(binary_img)))

buttons = []
current_points = white
displaying_points = []

create_points(displaying_points, current_points)
draw_image(window, displaying_points)

create_buttons(buttons)
draw_buttons(buttons, window)
pygame.display.flip()
running = True

current_points = []
displaying_points = []
for image in images:
    for i in range(len(image)):
        if image[i] == 255:
            image[i] == 1
        elif image[i] == 0:
            image[i] = -1
h = HopfieldNewtork(100)
h.training(images)

while running:
    for event in pygame.event.get():
        mouse = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
                for row in buttons:
                    for button in row:
                        if button.was_clicked(mouse[0], mouse[1]):
                            if button.text == 'run':
                                window.fill((0, 0, 0))
                                h.run_dynamics(current_points)
                                for i in range(len(current_points)):
                                    if current_points[i] == -1:
                                        current_points[i] = 255
                                    elif current_points[i] == 1:
                                        current_points[i] = 0
                                    print(current_points[i])
                                
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'clear':
                                window.fill((0, 0, 0))
                                current_points = np.copy(white)
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'noise':
                                number_of_changes = random.randint(50, 1000)
                                cells = random.sample(range(10000), number_of_changes)
                                for cell in cells:
                                    current_points[cell] = current_points[cell]*(-1)
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'square':
                                points = []
                                window.fill((0, 0, 0))
                                draw_square(window, points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text in numbers:
                                number = int(button.text)-1
                                window.fill((0, 0, 0))
                                current_points = np.copy(images[number])
                                for i in range(len(current_points)):
                                    if current_points[i] == 1:
                                        current_points[i] = 255
                                    elif current_points[i] == -1:
                                        current_points[i] = 0
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
            
            

