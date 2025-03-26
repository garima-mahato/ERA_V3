# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.animation import Animation

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1020')
Config.set('graphics', 'height', '808')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
GAMMA = 0.9
TEMP = 100
brain = Dqn(5,3,GAMMA,TEMP)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.jpg")
GOALS = [(700,281),(945,280),(854,576)]
WINDOW_SIZE = (1020,808)
current_goal_index = 0  # Start with the first goal
goal_reach_count = 0  # Track how many times all goals are reached
max_goal_reaches = 2  # Stop after reaching all goals 10 times

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.jpg").convert('L')
    sand = np.asarray(img)/255
    goal_x = GOALS[0][0]
    goal_y = GOALS[0][1]
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Update position and angle
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # # Boundary checks for x and y positions
        # if self.x < 0:
        #     self.x = 0
        #     self.velocity_x = abs(self.velocity_x)
        #     print(f"Car hit left boundary. New velocity: {self.velocity_x}, {self.velocity_y}")
        # if self.x > longueur - self.width:
        #     self.x = longueur - self.width
        #     self.velocity_x = -abs(self.velocity_x)
        #     print(f"Car hit right boundary. New velocity: {self.velocity_x}, {self.velocity_y}")
        # if self.y < 0:
        #     self.y = 0
        #     self.velocity_y = abs(self.velocity_y)
        #     print(f"Car hit bottom boundary. New velocity: {self.velocity_x}, {self.velocity_y}")
        # if self.y > largeur - self.height:
        #     self.y = largeur - self.height
        #     self.velocity_y = -abs(self.velocity_y)
        #     print(f"Car hit top boundary. New velocity: {self.velocity_x}, {self.velocity_y}")

        # Update sensor positions
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # Update sensor signals
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10,
                                       int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10,
                                       int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10,
                                       int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.

        # Handle out-of-bound sensor signals
        if self.sensor1_x > longueur - 10 or self.sensor1_x < 10 or self.sensor1_y > largeur - 10 or self.sensor1_y < 10:
            self.signal1 = 10.
        if self.sensor2_x > longueur - 10 or self.sensor2_x < 10 or self.sensor2_y > largeur - 10 or self.sensor2_y < 10:
            self.signal2 = 10.
        if self.sensor3_x > longueur - 10 or self.sensor3_x < 10 or self.sensor3_y > largeur - 10 or self.sensor3_y < 10:
            self.signal3 = 10.
        

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class GoalAnimation(Widget):
    def animate(self, pos):
        # Set the initial position and size of the animation
        self.center = pos
        self.size = (10, 10)
        self.opacity = 1  # Ensure the widget is visible

        # Create an animation: grow the circle and fade it out
        anim = Animation(size=(100, 100), opacity=self.opacity, duration=1.0)
        anim.bind(on_complete=self.remove_from_parent)  # Remove the widget after animation
        anim.start(self)

    def remove_from_parent(self, *args):
        # Remove the widget from its parent
        if self.parent:
            self.parent.remove_widget(self)

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global current_goal_index, goal_reach_count
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global current_goal_index  # Track the current goal

        longueur = WINDOW_SIZE[0]  # Width of the map
        largeur = WINDOW_SIZE[1]  # Height of the map

        if first_update:
            init()

        # Set the current goal
        goal_x, goal_y = GOALS[current_goal_index]

        # Calculate orientation and distance to the goal
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        # Update the AI and get the action
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]

        # Move the car
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        # Update sensor ball positions
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Check if the car is on sand
        print(self.car.x , self.width, self.car.y, self.height, int(self.car.x), int(self.car.y))
        if self.car.x>=5 and self.car.x <= WINDOW_SIZE[0]-5 and self.car.y>=5 and self.car.y <= WINDOW_SIZE[1]-5 and sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)  # Slow down
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -1  # Negative reward for being on sand
        else:
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)  # Normal speed
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -0.2  # Small negative reward for moving
            if distance < last_distance:
                last_reward = 0.1  # Positive reward for getting closer to the goal

        # Boundary checks
        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > WINDOW_SIZE[0] - 5:
            self.car.x = WINDOW_SIZE[0] - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > WINDOW_SIZE[1] - 5:
            self.car.y = WINDOW_SIZE[1] - 5
            last_reward = -1

        # Check if the car has reached the current goal
        if distance < 25:
            # Trigger the goal animation
            self.show_goal_animation((goal_x, goal_y))
            current_goal_index += 1  # Move to the next goal
            if current_goal_index >= len(GOALS):  # All goals reached
                current_goal_index = 0
                goal_reach_count += 1
                print(f"All goals reached {goal_reach_count} times.")
                if goal_reach_count >= max_goal_reaches:
                    print("Training complete! All goals reached multiple times.")
                    brain.save(f"last_brain_gamma_{GAMMA}_temp_{TEMP}.pth")
                    brain.save_logs(f"signal_reward_log_gamma_{GAMMA}_temp_{TEMP}.csv")  # Save logs
                    plt.plot(scores)
                    plt.xlabel("Episodes")  # Add x-axis title
                    plt.ylabel("Scores")    # Add y-axis title
                    plt.title(f"AI Performance Over Time (Gamma: {GAMMA}, Temperature: {TEMP})")  # Optional: Add a graph title
                    plt.savefig(f"scores_gamma_{GAMMA}_temp_{TEMP}.png")
                    App.get_running_app().stop()  # Stop the Kivy app
                    return
                    

        # Update the last distance
        last_distance = distance

    def show_goal_animation(self, goal_pos):
        print(f"Triggering animation at position: {goal_pos}")  # Debug print
        animation = GoalAnimation()
        animation.center = goal_pos
        self.add_widget(animation)
        animation.animate(goal_pos)
    
    def draw_goal_markers(self):
        """Draw markers at the goal positions."""
        with self.canvas:
            for goal in GOALS:
                Color(1, 0, 0, 0.8)  # Red color with some transparency
                d = 20  # Diameter of the marker
                Ellipse(pos=(goal[0] - d / 2, goal[1] - d / 2), size=(d, d))

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        self.title = "Triwizard Tournament 2025"  # Set custom window title
        parent = Game()
        parent.size = (1020,808)
        parent.serve_car()
        parent.draw_goal_markers()  # Draw the goal markers
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (clearbtn.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * clearbtn.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        print(longueur,largeur,sand.shape)

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
