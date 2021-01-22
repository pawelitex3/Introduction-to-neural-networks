import numpy as np
import pygame
import random
import cv2
import matplotlib.pyplot as plt

pygame.init()
numbers = [str(i+1) for i in range(10)]

class HopfieldNewtork(object):
  def __init__(self, N):
    self.N = N
    self.weights = np.zeros((self.N, self.N))
    self.iterations = 5000

  def little_dynamics(self, state):
    state = np.dot(self.weights, np.transpose(state))
    for i in range(len(state)):
      if state[i] < 0:
        state[i] = -1
      else:
        state[i] = 1
    return state

  def glauber_dynamics(self, state):
    i = np.random.randint(0, self.N)
    a = np.matmul(self.weights[i, :], state)
    if a < 0:
      state[i] = -1
    else:
      state[i] = 1
    return state

  def run_dynamics(self, state, dtype='glauber'):
    energy = []
    for i in range(self.iterations):
      if dtype == 'glauber':
        state = self.glauber_dynamics(state)
        e = -0.5*np.dot(np.dot(np.transpose(state), self.weights), state)
        energy.append(e)
      else:
        state = self.little_dynamics(state)
    plt.plot(range(len(energy)), energy)
    plt.savefig('energy.pdf')
    plt.close()

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
      a = np.zeros(self.N)
      for i in range(self.N):
        a[i] = 0
        for j in range(self.N):
          a[i] += self.weights[i,j] * state[j]
      state = np.where(a < 0, -1, 1)
      for p in range(P):
        hamming_distance[iteration, p] = (state - training_data[p] != 0).sum()
      print(hamming_distance)


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
    row.append(Button(280, 620, 80, 30, 'negative'))
    for i in range(4):
        row.append(Button(10+i*60, 660, 50, 30, str(i+1)))

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


def create_points(points, image):
    image_copy = np.copy(image)
    for i in range(len(image)):
        points.append(Point(10+(i%50)*12, 10+(i//50)*12, (image_copy[i], image_copy[i], image_copy[i])))


def draw_image(screen, points):
    for i in range(2500):
        #print(points[i].color)
        if points[i].color == (0, 0, 0):
            pygame.draw.rect(screen, np.array(points[i].color)+70, (points[i].x, points[i].y, 10, 10), 0)
        else:
            pygame.draw.rect(screen, np.array(points[i].color), (points[i].x, points[i].y, 10, 10), 0)


window_size = width, height = 620, 740
window = pygame.display.set_mode(window_size)
window.fill((0, 0, 0))

white = []
for i in range(2500):
    white.append(255)


images = []
for i in range(4):
    filename = str(i+1) + ".png"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #print(np.ravel(np.array(binary_img))/255)
    images.append(np.ravel(np.array(binary_img)))

images = np.int16(images)

buttons = []
current_points = np.copy(white)
displaying_points = []

create_points(displaying_points, current_points)
draw_image(window, displaying_points)

white[white == 255] = -1

create_buttons(buttons)
draw_buttons(buttons, window)
pygame.display.flip()
running = True

current_points = []
displaying_points = []
for image in images:
    image[image == 0] = 1
    image[image == 255] = -1

h = HopfieldNewtork(2500)

h.training(images)

while running:
    for event in pygame.event.get():
        mouse = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
          if mouse[1] < 620:
            for i in range(len(displaying_points)):
              if mouse[0] >= displaying_points[i].x and mouse[0] < displaying_points[i].x + 10 and mouse[1] >= displaying_points[i].y and mouse[1] < displaying_points[i].y + 10:
                current_points[i] *= (-1)
                current_points[current_points == 1] = 0
                current_points[current_points == -1] = 255
                displaying_points = []
                create_points(displaying_points, current_points)
                draw_image(window, displaying_points)
                draw_buttons(buttons, window)
                current_points[current_points == 255] = -1
                current_points[current_points == 0] = 1
                pygame.display.flip()
          else:
                for row in buttons:
                    for button in row:
                        if button.was_clicked(mouse[0], mouse[1]):
                            if button.text == 'run':
                                window.fill((0, 0, 0))
                                h.run_dynamics(current_points)
                                differences = []
                                for image in images:
                                  diff = (image - current_points != 0).sum()
                                  differences.append(diff)
                                font = pygame.font.SysFont('verdana', 20)
                                t = f'number of difference: {min(differences)}'
                                text = font.render(t, 1, (255, 255, 0))
                                window.blit(text, (350, 680))
                                current_points[current_points == 1] = 0
                                current_points[current_points == -1] = 255
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                current_points[current_points == 255] = -1
                                current_points[current_points == 0] = 1
                                pygame.display.flip()
                            elif button.text == 'clear':
                                window.fill((0, 0, 0))
                                current_points = np.copy(white)
                                current_points[current_points == -1] = 255
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                current_points[current_points == 255] = -1
                                draw_buttons(buttons, window)
                                pygame.display.flip()
                            elif button.text == 'noise':
                                number_of_changes = random.randint(50, 100)
                                cells = random.sample(range(2500), number_of_changes)
                                for cell in cells:
                                    current_points[cell] = current_points[cell]*(-1)
                                current_points[current_points == 1] = 0
                                current_points[current_points == -1] = 255
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                current_points[current_points == 0] = 1
                                current_points[current_points == 255] = -1
                                pygame.display.flip()
                            elif button.text == 'negative':
                                current_points *= (-1)
                                current_points[current_points == 1] = 0
                                current_points[current_points == -1] = 255
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                current_points[current_points == 0] = 1
                                current_points[current_points == 255] = -1
                                pygame.display.flip()
                            elif button.text in numbers:
                                number = int(button.text)-1
                                window.fill((0, 0, 0))
                                current_points = np.copy(images[number])
                                current_points[current_points == 1] = 0
                                current_points[current_points == -1] = 255
                                displaying_points = []
                                create_points(displaying_points, current_points)
                                current_points[current_points == 255] = -1
                                current_points[current_points == 0] = 1
                                draw_image(window, displaying_points)
                                draw_buttons(buttons, window)
                                pygame.display.flip()
            
            

