import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_gestures(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        left_hand_gesture = None
        right_hand_gesture = None
        
        # Define colors for each hand (BGR format)
        red_color = (0, 0, 255)  # Red for left drone (BGR)
        blue_color = (255, 0, 0)  # Blue for right drone (BGR)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (left or right)
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Set color based on handedness (camera shows mirror image)
                # "Right" in camera is actually left hand (controls red drone)
                # "Left" in camera is actually right hand (controls blue drone)
                landmark_color = red_color if handedness == "Right" else blue_color
                
                # Draw landmarks with custom color
                drawing_spec = self.mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2)
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
                
                gesture = self._classify_gesture(hand_landmarks)
                
                # Fix the mapping: "Right" in camera is left hand and vice versa
                if handedness == "Right":  # Camera shows mirror image
                    left_hand_gesture = gesture
                else:
                    right_hand_gesture = gesture
                    
        return frame, left_hand_gesture, right_hand_gesture
    
    def _classify_gesture(self, landmarks):
        # Extract landmark positions
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        points = np.array(points)
        
        # Define key points
        thumb_tip = points[4]    # Thumb tip
        index_tip = points[8]    # Index finger tip
        middle_tip = points[12]  # Middle finger tip
        pinky_tip = points[20]   # Little finger tip
        
        # Get bases for reference
        thumb_base = points[2]   # Thumb base
        index_base = points[5]   # Index base
        middle_base = points[9]  # Middle base
        palm_center = points[9]  # Palm center
        
        # Get handedness (true for right hand, false for left hand)
        is_right_hand = points[4][0] < points[17][0]  # Thumb tip is left of pinky base for right hand
        
        # Check if hand is closed (fist) - STOP
        if self._is_fist(points):
            return "STOP"
        
        # Check for FORWARD - both thumb and index extended, others closed
        if (self._is_thumb_extended(thumb_tip, thumb_base) and
            self._is_finger_raised(index_tip, index_base) and
            not self._is_finger_raised(middle_tip, middle_base) and
            not self._is_finger_raised(pinky_tip, points[17])):
            return "FORWARD"
        
        # Check for CIRCLE - both thumb and pinky extended, others closed
        if (self._is_thumb_extended(thumb_tip, thumb_base) and
            (self._is_finger_raised(pinky_tip, points[17]) or self._is_finger_lowered(pinky_tip, points[17])) and
            not self._is_finger_raised(index_tip, index_base) and
            not self._is_finger_raised(middle_tip, middle_base)):
            return "CIRCLE"
        
        # Check for LEFT/RIGHT with thumb only
        if (self._is_thumb_extended(thumb_tip, thumb_base) and
            not self._is_finger_raised(index_tip, index_base) and
            not self._is_finger_raised(middle_tip, middle_base) and
            not self._is_finger_raised(pinky_tip, points[17])):
            # For right hand in camera (red drone): thumb pointing left means RIGHT
            # For left hand in camera (blue drone): thumb pointing left means LEFT
            thumb_points_left = thumb_tip[0] < thumb_base[0]
            if is_right_hand:
                return "RIGHT" if thumb_points_left else "LEFT"  # Red drone: left thumb = RIGHT
            else:
                return "RIGHT" if thumb_points_left else "LEFT"  # Blue drone: left thumb = LEFT
        
        # Check if only index finger is up - UP
        if (self._is_finger_raised(index_tip, index_base) and 
            not self._is_finger_raised(middle_tip, middle_base) and
            not self._is_finger_raised(pinky_tip, points[17]) and
            not self._is_thumb_extended(thumb_tip, thumb_base)):
            return "UP"
        
        # Check if both index and middle fingers are up - DOWN
        if (self._is_finger_raised(index_tip, index_base) and
            self._is_finger_raised(middle_tip, middle_base) and
            not self._is_finger_raised(pinky_tip, points[17]) and
            not self._is_thumb_extended(thumb_tip, thumb_base)):
            return "DOWN"
        
        # Check if only pinky is raised
        if (self._is_finger_raised(pinky_tip, points[17]) and
            not self._is_finger_raised(index_tip, index_base) and
            not self._is_finger_raised(middle_tip, middle_base) and
            not self._is_thumb_extended(thumb_tip, thumb_base)):
            return "RIGHT" if not is_right_hand else "LEFT"  # Right hand pinky = LEFT, Left hand pinky = RIGHT
        
        # Check for BACKWARD - thumb and index finger pinch
        if self._are_fingers_touching(thumb_tip, index_tip):
            # Make sure other fingers are down
            if (not self._is_finger_raised(middle_tip, middle_base) and
                not self._is_finger_raised(pinky_tip, points[17])):
                return "BACKWARD"
        
        return None

    def _is_fist(self, points, threshold=0.1):
        """Check if hand is in a fist position"""
        finger_tips = points[[8, 12, 16, 20]]  # All finger tips
        palm_center = points[9]
        distances = np.linalg.norm(finger_tips - palm_center, axis=1)
        return np.mean(distances) < threshold
    
    def _is_finger_raised(self, tip, base, threshold=0.1):
        """Check if a finger is raised by comparing y coordinates"""
        return tip[1] < base[1] - threshold
    
    def _is_finger_lowered(self, tip, base, threshold=0.1):
        """Check if a finger is lowered (for palm backward gestures)"""
        return tip[1] > base[1] + threshold
    
    def _is_thumb_extended(self, thumb_tip, thumb_base, threshold=0.1):
        """Simple check if thumb is extended laterally"""
        return abs(thumb_tip[0] - thumb_base[0]) > threshold
    
    def _are_fingers_touching(self, tip1, tip2, threshold=0.08):
        """Check if two finger tips are touching"""
        distance = np.linalg.norm(np.array([tip1[0], tip1[1], tip1[2]]) - 
                                np.array([tip2[0], tip2[1], tip2[2]]))
        return distance < threshold 