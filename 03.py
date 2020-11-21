import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
from termgraph import termgraph as tg
from os import system

pygame.init()


numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


numbers_matrix = [
    [
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
]


def fourier_transform(x):
    a = np.abs(np.fft.fft(x))
    return a/np.max(a)


def standarise_features(x):
    return (x - np.mean(x))/np.std(x)


class Button(object):
    def __init__(self, x, y, width=30, height=30, text='', version=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def was_clicked(self, x, y):
        return (self.x < x < self.x+self.width) and (self.y < y < self.y+self.height)


class Adaline(object):
    def __init__(self, no_of_input, number, learning_rate=0.01, iterations=10000, biased=False):
        self.no_of_input = no_of_input
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.random.random(2*self.no_of_input + 1) # Zadanie: dodanie biasu jest opcjonalne
        self.errors = []
        self.biased = biased
        self.number = number

    def _add_bias(self, x):
        if self.biased:
            #x = np.hstack!!!
            return 0
        else:
            return x
    
    def train(self, training_data_x, training_data_y):
        #preprocessed_training_data_x = standarise_features(training_data_x) # Zadanie: Standariza
        for _ in range(self.iterations):
            e = 0

            randomize_list = list(zip(training_data_x, training_data_y))
            random.shuffle(randomize_list)
            training_data_x, training_data_y = zip(*randomize_list)

            for x, y in zip(training_data_x, training_data_y):
                x_extended = np.concatenate([x, fourier_transform(x)])
                out = self.output(x)
                self.weights[1:] += self.learning_rate * (y - out) * x_extended * self.derivative(x_extended) # Zadanie: Co gdy mamy funkcje aktywacji sigmoidalną?
                self.weights[0] += self.learning_rate * (y - out)
                e += 0.5 * (y - out)**2
            self.errors.append(e)
        plt.plot(range(len(self.errors)), self.errors, label=str(self.number))
        plt.ylim(-0.5, 2.0)
        plt.legend()
        plt.savefig('errors.pdf')

    def activation(self, x): # Zadanie: dodanie funkcji aktywacji -> zmiana pochodnej
        return 1/(1 + np.exp(-x))
        #return x

    def output(self, input):
        inp = np.concatenate([input, fourier_transform(input)])
        summation = self.activation(np.dot(self.weights[1:], inp) + self._add_bias(self.weights[0]))
        return summation

    def derivative(self, input):
        return self.activation(input)*(1-self.activation(input))


def create_squares(squares):
    for i in range(10):
        row = []
        x = 10
        y = 10 + i*25
        for j in range(10):
            row.append(Button(x, y, width=20, height=20))
            x += 25
        squares.append(row)


def create_buttons(buttons):
    number = 1
    for i in range(3):
        row = []
        x = 10
        y = 310 + i*40
        for j in range(3):
            row.append(Button(x, y, 90, 30, str(number)))
            x += 100
            number += 1
        buttons.append(row)
    row = []
    row.append(Button(10, 430, 90, 30, '0'))
    row.append(Button(110, 430, 90, 30, 'clear'))
    row.append(Button(210, 430, 90, 30, 'random'))
    buttons.append(row)
    row = []
    row.append(Button(10, 470, 90, 30, 'negation'))
    row.append(Button(110, 470, 90, 30, 'up'))
    row.append(Button(210, 470, 90, 30, ''))
    buttons.append(row)
    row = []
    row.append(Button(10, 510, 90, 30, 'left'))
    row.append(Button(110, 510, 90, 30, 'down'))
    row.append(Button(210, 510, 90, 30, 'right'))
    buttons.append(row)
    row = []
    row.append(Button(10, 550, 290, 30, 'result: '))
    buttons.append(row)


def draw_squares(squares, screen, values):
    screen.fill((0, 0, 0))
    for row in range(len(squares)):
        for col in range(len(squares[row])):
            if values[row][col] == 0:
                pygame.draw.rect(screen, (180, 180, 180), (squares[row][col].x, squares[row][col].y, squares[row][col].width, squares[row][col].height), 0)
            else:
                pygame.draw.rect(screen, (0, 80, 80), (squares[row][col].x, squares[row][col].y, squares[row][col].width, squares[row][col].height), 0)


def draw_buttons(buttons, screen):
    for row in buttons:
        for button in row:
            pygame.draw.rect(screen, (180, 180, 180), (button.x, button.y, button.width, button.height), 0)
            font = pygame.font.SysFont('verdana', 20)
            text = font.render(button.text, 1, (0, 0, 0))
            x = button.x + button.width/2 - text.get_width()/2
            y = button.y + button.height/2 - text.get_height()/2
            screen.blit(text, (x, y))

    pygame.display.flip()


def main():
    main_size = width, height = 310, 590
    training_inputs = [np.ravel(n) for n in numbers_matrix]
    perceptrons = []

    for i in range(10):
        perceptrons.append(Adaline(10 * 10, number=i))
        labels = np.zeros(10)
        labels[i] = 1
        perceptrons[i].train(training_inputs, labels)

    running = True

    buttons = []
    squares = []
    values = np.zeros((10, 10))
    if running:
        main_screen = pygame.display.set_mode(main_size)
        create_squares(squares)
        draw_squares(squares, main_screen, values)
        create_buttons(buttons)
        draw_buttons(buttons, main_screen)

    while running:
        for event in pygame.event.get():
            mouse = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                row_int = 0
                for row in squares:
                    col_int = 0
                    for square in row:
                        if square.was_clicked(mouse[0], mouse[1]):
                            values[row_int][col_int] = values[row_int][col_int]*(-1) + 1
                            draw_squares(squares, main_screen, values)
                            draw_buttons(buttons, main_screen)
                        col_int += 1
                    row_int += 1
                for row in buttons:
                    for button in row:
                        if button.was_clicked(mouse[0], mouse[1]):
                            if button.text in numbers:
                                values = np.copy(numbers_matrix[int(button.text)])
                                #mpl.imshow(np.reshape(perceptrons[int(button.text)].weights[1:], (5, 5)))
                                #mpl.show()
                            elif button.text == 'negation':
                                values = np.copy(values) * (-1) + 1
                            elif button.text == 'clear':
                                values = np.zeros((10, 10))
                            elif button.text == 'random':
                                for i in range(len(values)):
                                    for j in range(len(values[i])):
                                        r = random.randrange(20)
                                        if r < 1:
                                            values[i][j] = values[i][j]*(-1) + 1
                            elif button.text == 'up':
                                for i in range(len(values)-1):
                                    first_row = np.copy(values[i])
                                    values[i], values[i+1] = values[i+1], first_row
                            elif button.text == 'down':
                                for i in range(len(values)-1, 0, -1):
                                    last_row = np.copy(values[i])
                                    values[i], values[i-1] = values[i-1], last_row
                            elif button.text == 'left':
                                for i in range(len(values)):
                                    for j in range(len(values[i])-1):
                                        values[i][j], values[i][j+1] = values[i][j+1], values[i][j]
                            elif button.text == 'right':
                                for i in range(len(values)):
                                    for j in range(len(values[i])-1, 0, -1):
                                        values[i][j], values[i][j-1] = values[i][j-1], values[i][j]

                data_now = np.ravel(values)
                output = ''
                system('cls')
                print("Confidence:")
                confidence = []
                inp = standarise_features(data_now)
                

                for i in range(10):
                    confidence.append(perceptrons[i].output(data_now))

                norm = np.linalg.norm(confidence)
                confidence = confidence/norm
                
                output += str(np.argmax(confidence))
                labels = []
                data = []
                normal_data = []
                for i in range(10):
                    data.append([ confidence[i] ])
                    labels.append(str(i))
                    normal_data.append([ confidence[i]*100 ])

                buttons[6][0].text = 'result: ' + output

                draw_squares(squares, main_screen, values)
                draw_buttons(buttons, main_screen)

                len_categories = 1
                args = {'title': None, 'width': 50,
                        'format': '{:<1.7f}', 'suffix': '', 'no_labels': False,
                        'color': None, 'vertical': False, 'stacked': True,
                        'different_scale': False, 'calendar': False,
                        'start_dt': None, 'custom_tick': '', 'delim': '',
                        'verbose': False, 'version': False}
                colors = [91, 94]
                tg.stacked_graph(labels, data, normal_data, len_categories, args, colors)


if __name__ == "__main__":
    main()
