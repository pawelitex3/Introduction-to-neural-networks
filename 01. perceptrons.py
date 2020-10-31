import random
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
    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=1000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.rand(self.no_of_inputs + 1)/10

    def train_SPLA(self, training_data, labels):
        for _ in range(self.iterations):
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
        leader = np.asarray(self.weights).copy()
        leader_life = 0
        for _ in range(self.iterations):
            random_list = list(zip(training_data, labels))
            random.shuffle(random_list)
            training_data, labels = zip(*random_list)
            for input, label in zip(training_data, labels):
                input_copy = self.noisy(input)
                prediction = self.output(input_copy)
                error = label - prediction
                if error != 0.0:
                    if life > leader_life:
                        leader = np.asarray(self.weights).copy()
                        leader_life = life
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)
                    life = 0
                else:
                    life += 1

        if life > leader_life:
            leader = np.asarray(self.weights).copy()
            leader_life = life

        self.weights = np.asarray(leader).copy()

    def train_RPLA(self, training_data, labels):
        life = 0
        leader = np.asarray(self.weights).copy()
        leader_life = 0
        for _ in range(self.iterations):
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
                            leader = np.asarray(self.weights).copy()
                            leader_life = life
                    self.weights[1:] += self.learning_rate * (label - prediction) * input_copy
                    self.weights[0] += self.learning_rate * (label - prediction)
                    life = 0
                else:
                    life += 1

        if life > leader_life:
            leader = np.asarray(self.weights).copy()
            leader_life = life

        self.weights = np.asarray(leader).copy()

    def output(self, input):
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def noisy(self, input):
        copy = np.asarray(input).copy()
        number_of_changes = np.random.randint(0, 3)
        changed_cells = []
        for _ in range(number_of_changes):
            cell = np.random.randint(0, 24)
            while cell in changed_cells:
                cell = np.random.randint(0, 24)
            copy[cell] = input[cell]*(-1) + 1

        return copy


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
    main_size = width, height = 310, 590
    choose_size = 310, 70
    training_inputs = [np.ravel(n) for n in numbers_matrix]
    perceptrons = []

    for _ in range(10):
        perceptrons.append(Perceptron(5 * 5))


    choose_algorithm_screen = pygame.display.set_mode(choose_size)
    buttons = []
    row = []
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
                            for i in range(10):
                                labels = np.zeros(10)
                                labels[i] = 1
                                perceptrons[i].train_SPLA(training_inputs, labels)
                        elif button.text == "PLA":
                            for i in range(10):
                                labels = np.zeros(10)
                                labels[i] = 1
                                perceptrons[i].train_PLA(training_inputs, labels)
                        elif button.text == "RPLA":
                            for i in range(10):
                                labels = np.zeros(10)
                                labels[i] = 1
                                perceptrons[i].train_RPLA(training_inputs, labels)
                        choosing = False




    buttons = []
    squares = []
    values = np.zeros((5, 5))
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
                                values = np.asarray(numbers_matrix[int(button.text)])
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

                draw_squares(squares, main_screen, values)
                draw_buttons(buttons, main_screen)


if __name__ == "__main__":
    main()
