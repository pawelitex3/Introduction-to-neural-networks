import random
import pygame
import numpy as np
import matplotlib.pyplot as mpl

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
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0]
    ],
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ],
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0]
    ]
]


class Perceptron(object):
    def __init__(self, no_of_inputs, learning_rate=0.001, iterations=1500):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.rand(self.no_of_inputs + 1)

    def train_SPLA(self, training_data, labels):
        for i in range(self.iterations):
            random_list = list(zip(training_data, labels))
            random.shuffle(random_list)
            training_data, labels = zip(*random_list)
            for input, label in zip(training_data, labels):
                input_copy = self.noisy(input)
                prediction = self.output(input_copy)
                error = label - prediction
                if error != 0.0:
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)

    def train_PLA(self, training_data, labels):
        life = 0
        leader = np.copy(self.weights)
        leader_life = 0
        for i in range(self.iterations):
            random_list = list(zip(training_data, labels))
            random.shuffle(random_list)
            training_data, labels = zip(*random_list)
            for input, label in zip(training_data, labels):
                input_copy = self.noisy(input)
                prediction = self.output(input_copy)
                error = label - prediction
                if error != 0.0:
                    if life > leader_life:
                        leader = np.copy(self.weights)
                        leader_life = life
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)
                    life = 0
                else:
                    life += 1

        if life > leader_life:
            leader = np.copy(self.weights)

        self.weights = np.copy(leader)

    def train_RPLA(self, training_data, labels):
        life = 0
        leader = self.weights
        leader_life = 0
        for i in range(self.iterations):
            random_list = list(zip(training_data, labels))
            random.shuffle(random_list)
            training_data, labels = zip(*random_list)
            for input, label in zip(training_data, labels):
                input_copy = self.noisy(input)
                prediction = self.output(input_copy)
                error = label - prediction
                if error != 0.0:
                    if life > leader_life:
                        old_correct = 0
                        new_correct = 0
                        for input_check, label_check in zip(training_data, labels):
                            old_prediction = self.output(input_check)
                            new_prediction = self.output(input_check)
                            if label-old_prediction == 0.0:
                                old_correct += 1
                            if label-new_prediction == 0.0:
                                new_correct += 1
                        if new_correct > old_correct:
                            leader = np.copy(self.weights)
                            leader_life = life
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)
                    life = 0
                else:
                    life += 1

        if life > leader_life:
            leader = np.copy(self.weights)

        self.weights = np.copy(leader)

    def train(self, training_data, labels):
        life = 0
        leader = self.weights
        leader_life = 0
        for i in range(500):
            random_list = list(zip(training_data, labels))
            random.shuffle(random_list)
            training_data, labels = zip(*random_list)
            for input, label in zip(training_data, labels):
                input_copy = self.noisy(input)
                prediction = self.output(input_copy)
                error = label - prediction
                if error != 0.0:
                    if life > leader_life:
                        old_correct = 0
                        new_correct = 0
                        for input_check, label_check in zip(training_data, labels):
                            old_prediction = self.output(input_check)
                            new_prediction = self.output(input_check)
                            if label-old_prediction == 0.0:
                                old_correct += 1
                            if label-new_prediction == 0.0:
                                new_correct += 1
                        if new_correct > old_correct:
                            leader = np.copy(self.weights)
                            leader_life = life
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)
                    life = 0
                else:
                    life += 1

        if life > leader_life:
            leader = np.copy(self.weights)

        self.weights = np.copy(leader)

    def output(self, input):
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def noisy(self, input):
        copy = np.copy(input)
        number_of_changes = random.randint(0, 2)
        cells = random.sample(range(25), number_of_changes)
        for i in cells:
            copy[i] = input[i]*(-1) + 1

        return copy


class Button(object):
    def __init__(self, x, y, width=50, height=50, text='', version=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.version = version

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
            row.append(Button(x, y, 40, 30, str(number)))
            row.append(Button(x+50, y, 40, 30, str(number), 2))
            x += 100
            number += 1
        buttons.append(row)
    row = []
    row.append(Button(10, 430, 40, 30, '0'))
    row.append(Button(60, 430, 40, 30, '0', 2))
    row.append(Button(110, 430, 90, 30, 'clear'))
    row.append(Button(210, 430, 90, 30, 'random'))
    buttons.append(row)
    row = []
    row.append(Button(10, 470, 90, 30, 'negation'))
    row.append(Button(110, 470, 90, 30, 'up'))
    row.append(Button(210, 470, 90, 30, 'plot'))
    buttons.append(row)
    row = []
    row.append(Button(10, 510, 90, 30, 'left'))
    row.append(Button(110, 510, 90, 30, 'down'))
    row.append(Button(210, 510, 90, 30, 'right'))
    buttons.append(row)
    row = []
    number = 0
    for i in range(2):
        row = []
        x = 10
        y = 550 + i*40
        for j in range(5):
            row.append(Button(x, y, 50, 30, str(number), version=3))
            x += 60
            number += 1
        buttons.append(row)
    row = []
    row.append(Button(10, 630, 290, 30, 'train', version=3))
    buttons.append(row)
    row = []
    row.append(Button(10, 670, 290, 30, 'result: '))
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
            if button.version == 3:
                pygame.draw.rect(screen, (30, 160, 60), (button.x, button.y, button.width, button.height), 0)
            else:
                pygame.draw.rect(screen, (180, 180, 180), (button.x, button.y, button.width, button.height), 0)
            font = pygame.font.SysFont('verdana', 20)
            text = font.render(button.text, 1, (0, 0, 0))
            x = button.x + button.width/2 - text.get_width()/2
            y = button.y + button.height/2 - text.get_height()/2
            screen.blit(text, (x, y))

    pygame.display.flip()


def main():
    main_size = width, height = 310, 710
    choose_size = 310, 70
    training_inputs = [np.ravel(n) for n in numbers_matrix]
    perceptrons = []

    for _ in range(10):
        perceptrons.append(Perceptron(5 * 5))


    choose_algorithm_screen = pygame.display.set_mode(choose_size)
    buttons = []
    row = []
    labels = []
    row.append(Button(10, 10, 90, text="SPLA"))
    row.append(Button(110, 10, 90, text="PLA"))
    row.append(Button(210, 10, 90, text="RPLA"))
    buttons.append(row)
    draw_buttons(buttons, choose_algorithm_screen)
    choosing = True
    running = True
    while choosing:
        for event in pygame.event.get():
            mouse = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                choosing = False
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons[0]:
                    if button.was_clicked(mouse[0], mouse[1]):
                        if button.text == "SPLA":
                            for i in range(20):
                                lab = np.zeros(20)
                                lab[i % 10] = 1
                                lab[i % 10 + 10] = 1
                                labels.append(lab)
                                perceptrons[i % 10].train_SPLA(training_inputs, lab)
                        elif button.text == "PLA":
                            for i in range(20):
                                lab = np.zeros(20)
                                lab[i % 10] = 1
                                lab[i % 10 + 10] = 1
                                labels.append(lab)
                                perceptrons[i % 10].train_PLA(training_inputs, lab)
                        elif button.text == "RPLA":
                            for i in range(20):
                                lab = np.zeros(20)
                                lab[i % 10] = 1
                                lab[i % 10 + 10] = 1
                                labels.append(lab)
                                perceptrons[i % 10].train_RPLA(training_inputs, lab)
                        choosing = False

    buttons = []
    squares = []
    values = np.zeros((5, 5))
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
                                if button.version == 1:
                                    values = np.copy(numbers_matrix[int(button.text)])
                                elif button.version == 2:
                                    values = np.copy(numbers_matrix[int(button.text) + 10])
                                elif button.version == 3:
                                    for i in range(len(perceptrons)):
                                        data_now = np.ravel(values)
                                        training_inputs.append(data_now)

                                        if int(button.text) == i:
                                            labels[i] = np.append(labels[i], np.array([1]))
                                        else:
                                            labels[i] = np.append(labels[i], np.array([0]))

                                        perceptrons[i].train(training_inputs, labels[i])

                            elif button.text == 'negation':
                                values = np.asarray(values) * (-1) + 1
                            elif button.text == 'clear':
                                values = np.zeros((5, 5))
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
                            elif button.text == 'plot':
                                for i in range(len(perceptrons)):
                                    name = str(i) + '.png'
                                    mpl.imshow(np.reshape(perceptrons[i].weights[1:], (5, 5)))
                                    mpl.savefig(name)
                            elif button.text == 'plot':
                                for i in range(len(perceptrons)):
                                    perceptrons[i].train(training_inputs, labels[i])

                data_now = np.ravel(values)
                output = ''
                for i in range(10):
                    if perceptrons[i].output(data_now) == 1:
                        if len(output) > 0:
                            output += ', '
                        output += str(i)

                buttons[9][0].text = 'result: ' + output

                draw_squares(squares, main_screen, values)
                draw_buttons(buttons, main_screen)


if __name__ == "__main__":
    main()
