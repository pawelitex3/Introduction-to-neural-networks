import numpy as np
import matplotlib.pyplot as plt
import pygame

pygame.init()
image = pygame.image.load('robot.jpg')
image_width = image.get_width()
image_height = image.get_height()
arm_length = 100.0


def find_arms(angles):
    alpha = np.pi - angles[0]
    beta = angles[1]*(-1.0)
    first_joint = translate(Point(image_width, image_height/2), alpha)
    second_joint = translate(first_joint, np.pi - beta + alpha)
    return first_joint, second_joint


def unstandarize(angles):
    return np.array(angles)*np.pi


def translate(center, angle):
    return Point(center.x + arm_length * np.sin(angle), center.y - arm_length * np.cos(angle))


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Example(object):
    def __init__(self, arm_length=10.0):
        self.center = Point(0, 0)
        self.arm_length = arm_length
        self.input = []
        self.output = []

    def generate(self, no_of_examples):
        for i in range(no_of_examples):
            point = self.generate_point(self.center)
            self.input.append([point.x, point.y])
        return self.input, self.output

    def generate_point(self, center):
        alpha = np.random.random()*np.pi
        beta = np.random.random()*np.pi
        self.output.append([alpha, beta])
        temppoint = self.translate(center, alpha)
        finalpoint = self.translate(temppoint, np.pi - beta + alpha)
        return finalpoint

    def translate(self, center, angle):
        return Point(center.x + self.arm_length * np.sin(angle), center.y - self.arm_length * np.cos(angle))


class Neural_Newtork(object):
    def __init__(self, eta = 0.005):
        self.input_size = 2
        self.output_size = 2
        self.hidden_size = 15
        self.eta = eta
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size) - 0.5
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size) - 0.5
        self.error = []

    def forward(self, x):
        self.a1 = np.dot(x, self.weights_1)
        self.y1 = self.sigmoid(self.a1)
        self.a2 = np.dot(self.y1, self.weights_2)
        self.y2 = self.sigmoid(self.a2)
        return self.y2

    def backward(self, x, y):
        self.epsilon_2 = y - self.y2
        self.delta_2 = self.epsilon_2 * self.sigmoid_derivative(self.y2)

        self.epsilon_1 = self.delta_2.dot(self.weights_2.T)
        self.delta_1 = self.epsilon_1 * self.sigmoid_derivative(self.y1)

        self.weights_1 += self.eta * x.T.dot(self.delta_1)
        self.weights_2 += self.eta * self.y1.T.dot(self.delta_2)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, s):
        return s*(1-s)

    def train(self, x, y):
        self.forward(x)
        self.backward(x, y)
        self.error.append(np.mean(np.square(y - self.y2)))


ex = Example(arm_length)
examples = ex.generate(2000)

x_min = np.min(examples[0])
x_max = np.max(examples[0])

x = (np.array(examples[0]) - x_min)/(x_max - x_min)*0.8 + 0.1
y = np.array(examples[1])/np.pi*0.8 + 0.1


NN = Neural_Newtork()
for i in range(10000):
    NN.train(x, y)
plt.plot(range(len(NN.error)), NN.error)
plt.savefig('errors.pdf')

fig, ax = plt.subplots()
ax.axis('equal')
for(x, y) in examples[0]:
    plt.scatter(x, y, marker='o')

plt.savefig('examples_0.png')

fig, ax = plt.subplots()
ax.axis('equal')
for(x, y) in examples[1]:
    plt.scatter(x, y, marker='o')

plt.savefig('examples_1.png')

window_size = width, height = 600, 500
window = pygame.display.set_mode(window_size)

window.fill((255, 255, 255))
window.blit(image, (0, 0))
pygame.display.flip()
running = True

while running:
    for event in pygame.event.get():
        mouse = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            
            if mouse[0] > image_width:
                x = mouse[0] - image_width
                y = mouse[1] - image_height/2
                window.fill((255, 255, 255))
                window.blit(image, (0, 0))
                pygame.draw.circle(window, (160, 50, 50), (mouse[0], mouse[1]), 5)

                prediction = unstandarize(NN.forward(((x + arm_length*2)/(arm_length*4)*0.8 + 0.1, (-y + arm_length*2)/(arm_length*4)*0.8 + 0.1)))
                arms = find_arms(prediction)

                pygame.draw.line(window, (200,60,40), (image_width, image_height/2), (arms[0].x, arms[0].y), width=5)
                pygame.draw.line(window, (40,60,255), (arms[0].x, arms[0].y), (arms[1].x, arms[1].y), width=5)

                pygame.display.flip()
