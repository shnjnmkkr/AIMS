import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

class DroneSimulator:
    def __init__(self, width=1000, height=600):
        self.width = width
        self.height = height
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Drone Simulator")
        
        # Enable 3D rendering
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Set up the camera
        self._setup_camera()
        
        # Initialize drones
        self.drones = [
            {
                'pos': np.array([-2.0, 0.0, -10.0], dtype=float),
                'rot': np.array([0., 0., 0.]),
                'color': (1.0, 0.0, 0.0)  # Red for drone 1
            },
            {
                'pos': np.array([2.0, 0.0, -10.0], dtype=float),
                'rot': np.array([0., 0., 0.]),
                'color': (0.0, 0.0, 1.0)  # Blue for drone 2
            }
        ]
        
        self.speed = 0.1
        self.gesture_speed = 0.15
        self.circle_radius = 1.0
        self.circle_speed = 0.05
        self.circle_angles = [0.0, 0.0]

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width/self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def render(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Draw grid for reference
        glPushMatrix()
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glVertex3f(i, -3, -15)
            glVertex3f(i, -3, -5)
            glVertex3f(-5, -3, i-10)
            glVertex3f(5, -3, i-10)
        glEnd()
        glPopMatrix()
        
        # Draw drones
        for drone in self.drones:
            self.draw_drone(drone)
        
        pygame.display.flip()

    def draw_drone(self, drone):
        glPushMatrix()
        glTranslatef(*drone['pos'])
        glRotatef(drone['rot'][0], 1, 0, 0)  # Pitch
        glRotatef(drone['rot'][1], 0, 1, 0)  # Yaw
        glRotatef(drone['rot'][2], 0, 0, 1)  # Roll
        
        # Set drone color
        glColor3f(*drone['color'])
        
        # Draw drone body (center cube)
        glPushMatrix()
        glScalef(0.2, 0.2, 0.2)
        self._draw_cube()
        glPopMatrix()
        
        # Draw drone arms
        for angle in range(0, 360, 90):
            glPushMatrix()
            glRotatef(angle, 0, 1, 0)
            
            # Draw arm
            glPushMatrix()
            glTranslatef(0.4, 0, 0)
            glScalef(0.8, 0.05, 0.05)  # Make it long and thin
            self._draw_cube()
            glPopMatrix()
            
            # Draw propeller hub
            glPushMatrix()
            glTranslatef(0.8, 0, 0)
            glScalef(0.1, 0.1, 0.1)
            self._draw_cube()
            glPopMatrix()
            
            # Draw propeller blades
            for blade in range(2):
                glPushMatrix()
                glTranslatef(0.8, 0, 0)
                glRotatef(blade * 180, 0, 1, 0)
                glTranslatef(0.2, 0, 0)
                glScalef(0.4, 0.02, 0.1)
                self._draw_cube()
                glPopMatrix()
            
            glPopMatrix()
        
        glPopMatrix()

    def _draw_cube(self):
        """Draw a simple cube using GL_QUADS"""
        vertices = [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],  # Front
            [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]       # Back
        ]
        
        surfaces = [
            [0, 1, 2, 3],  # Front
            [3, 2, 7, 6],  # Left
            [6, 7, 5, 4],  # Back
            [4, 5, 1, 0],  # Right
            [1, 5, 7, 2],  # Top
            [4, 0, 3, 6]   # Bottom
        ]
        
        glBegin(GL_QUADS)
        for surface in surfaces:
            for vertex in surface:
                glVertex3fv(vertices[vertex])
        glEnd()

    def update(self, commands):
        for i, command in enumerate(commands):
            if command is None:
                continue
                
            if command == "CIRCLE":
                # Update circle angle
                self.circle_angles[i] += self.circle_speed
                
                # Calculate new position in XZ plane
                base_x = -2.0 if i == 0 else 2.0
                self.drones[i]['pos'][0] = base_x + self.circle_radius * np.cos(self.circle_angles[i])
                self.drones[i]['pos'][2] = -10.0 + self.circle_radius * np.sin(self.circle_angles[i])
                
                # Add rotation for visual effect
                self.drones[i]['rot'][1] = np.degrees(self.circle_angles[i])  # Rotate around Y axis
            else:
                if command == "UP":
                    self.drones[i]['pos'][1] += self.speed
                elif command == "DOWN":
                    self.drones[i]['pos'][1] -= self.speed
                elif command == "LEFT":
                    self.drones[i]['pos'][0] -= self.speed
                elif command == "RIGHT":
                    self.drones[i]['pos'][0] += self.speed
                elif command == "FORWARD":
                    self.drones[i]['pos'][2] += self.speed
                elif command == "BACKWARD":
                    self.drones[i]['pos'][2] -= self.speed
                
                # Reset rotation when not circling
                self.drones[i]['rot'] = np.array([0., 0., 0.])
            
            # Keep drones within bounds
            self.drones[i]['pos'] = np.clip(
                self.drones[i]['pos'],
                [-5.0, -3.0, -15.0],
                [5.0, 3.0, -5.0]
            ) 