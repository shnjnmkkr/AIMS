import pygame
import numpy as np

class DroneSimulator:
    def __init__(self, width=1000, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Drone Simulator")
        
        # Initialize two drones with position, velocity, and acceleration
        self.drones = [
            {
                'pos': np.array([width/3, height/2, 0], dtype=float),
                'vel': np.array([0., 0., 0.]),
                'acc': np.array([0., 0., 0.]),
                'color': (255, 0, 0)  # Red for drone 1
            },
            {
                'pos': np.array([2*width/3, height/2, 0], dtype=float),
                'vel': np.array([0., 0., 0.]),
                'acc': np.array([0., 0., 0.]),
                'color': (0, 0, 255)  # Blue for drone 2
            }
        ]
        
        # Increase base speed for more pronounced movements
        self.speed = 20.0  # Increased from 5.0
        
        # Add movement amplification for yes/no gestures
        self.gesture_speed = 30.0  # Even faster for yes/no gestures
        
        # Add circle animation parameters
        self.circle_radius = 50.0
        self.circle_speed = 0.1
        self.circle_angles = [0.0, 0.0]  # One for each drone
    
    def update(self, commands):
        """
        Update drone positions based on commands
        commands: list of two commands, one for each drone
        """
        speed = self.speed
        for i, command in enumerate(commands):
            if command is None:
                continue
                
            if command == "CIRCLE":
                # Update circle angle
                self.circle_angles[i] += self.circle_speed
                
                # Calculate new position on circle
                center_x = self.drones[i]['pos'][0]
                center_y = self.drones[i]['pos'][1]
                
                self.drones[i]['pos'][0] = center_x + self.circle_radius * np.cos(self.circle_angles[i])
                self.drones[i]['pos'][1] = center_y + self.circle_radius * np.sin(self.circle_angles[i])
            else:
                if command == "UP":
                    self.drones[i]['pos'][1] -= speed
                elif command == "DOWN":
                    self.drones[i]['pos'][1] += speed
                elif command == "LEFT":
                    self.drones[i]['pos'][0] -= speed
                elif command == "RIGHT":
                    self.drones[i]['pos'][0] += speed
                elif command == "FORWARD":
                    self.drones[i]['pos'][2] += speed
                elif command == "BACKWARD":
                    self.drones[i]['pos'][2] -= speed
                
            # Keep drones within bounds
            self.drones[i]['pos'] = np.clip(
                self.drones[i]['pos'],
                [self.circle_radius, self.circle_radius, -100],
                [self.width - self.circle_radius, self.height - self.circle_radius, 100]
            )
    
    def render(self):
        self.screen.fill((255, 255, 255))  # White background
        
        for drone in self.drones:
            # Calculate size based on Z position (perspective effect)
            size = max(5, min(30, 20 + drone['pos'][2] / 10))
            
            # Draw drone
            pygame.draw.circle(
                self.screen,
                drone['color'],
                (int(drone['pos'][0]), int(drone['pos'][1])),
                int(size)
            )
        
        pygame.display.flip() 