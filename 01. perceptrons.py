from random import randrange
import pygame
import numpy as np

pygame.init()

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

numbers_matrix = [
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ]
]

class Perceptron(object):
    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=100):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.zeros(self.no_of_inputs + 1) # ZADANIE DOMOWE A

    def train(self, training_data, labels):
        for _ in range(self.iterations):
            for input, label in zip(training_data, labels): # LOSOWOŚĆ - ZADANIE DOMOWE 2
                # ZADANIE DOMOWE 3 - PLA
                # ZADANIE DOMOWE 4 - RATCHET RPLA
                # ZADANIE DOMOTE 5 - GUI: czyszczenie, 0-9, negacja, randomowe zmienianie bitów, przesuwanie do góry, dół, prawo, lewo
                prediction = self.output(input)
                self.weights[1:] += self.learning_rate * (label - prediction) * input
                self.weights[0] += self.learning_rate * (label - prediction)

    def output(self, input):
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation


class Button(object):
    def __init__(self, x, y, width=50, height=50, text=''):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def was_clicked(self, x, y):
        return (self.x < x < self.x+self.width) and (self.y < y < self.y+self.height)


def create_squares(squares):
    for i in range(5):
        row = []
        x = 10
        y = 10 + i*60
        for j in range(5):
            row.append(Button(x, y))
            x += 60
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
    row.append(Button(10, 550, 290, 30, ''))
    buttons.append(row)


def draw_squares(squares, screen, values):
    screen.fill((0, 0, 0))
    for row in range(len(squares)):
        for col in range(len(squares[row])):
            if values[row][col] == 0:
                pygame.draw.rect(screen, (180, 180, 180), (squares[row][col].x, squares[row][col].y, squares[row][col].width, squares[row][col].height), 0)
            else:
                pygame.draw.rect(screen, (0, 80, 80), (squares[row][col].x, squares[row][col].y, squares[row][col].width, squares[row][col].height), 0)
    #pygame.display.flip()


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
    size = width, height = 310, 590
    screen = pygame.display.set_mode(size)
    squares = []
    values = np.zeros((5, 5))
    buttons = []
    create_squares(squares)
    draw_squares(squares, screen, values)
    create_buttons(buttons)
    draw_buttons(buttons, screen)

    training_inputs = [np.ravel(n) for n in numbers_matrix]
    perceptrons = []

    for _ in range(10):
        perceptrons.append(Perceptron(5*5))

    for i in range(10):
        labels = np.zeros(10)
        labels[i] = 1
        perceptrons[i].train(training_inputs, labels)

    running = True

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
                            draw_squares(squares, screen, values)
                            draw_buttons(buttons, screen)
                        col_int +=1
                    row_int += 1
                for row in buttons:
                    for button in row:
                        if button.was_clicked(mouse[0], mouse[1]):
                            if button.text in numbers:
                                values = np.asarray(numbers_matrix[int(button.text)])
                            elif button.text == 'negation':
                                values = np.asarray(values) * (-1) + 1
                            elif button.text == 'clear':
                                values = np.zeros((5, 5))
                            elif button.text == 'random':
                                values = np.zeros((5, 5))
                                for i in range(len(values)):
                                    for j in range(len(values[i])):
                                        values[i][j] = randrange(2)
                            elif button.text == 'up':
                                for i in range(len(values)-1):
                                    first_row = np.asarray(values[i]).copy()
                                    values[i], values[i+1] = values[i+1], first_row
                            elif button.text == 'down':
                                for i in range(len(values)-1, 0, -1):
                                    last_row = np.asarray(values[i]).copy()
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
                for i in range(10):
                    if perceptrons[i].output(data_now) == 1:
                        if len(output) > 0:
                            output += ', '
                        output += str(i)

                buttons[6][0].text = 'result: ' + output

                draw_squares(squares, screen, values)
                draw_buttons(buttons, screen)


if __name__ == "__main__":
    main()
