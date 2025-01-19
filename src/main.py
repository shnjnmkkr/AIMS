import cv2
from hand_tracking import HandTracker
from simulator import DroneSimulator
from voice_processor import VoiceProcessor
from llm_interface import LLMInterface
import time

def init_camera(index):
    """Initialize camera with specific index"""
    print(f"Attempting to initialize camera {index}...")
    
    # Try MacOS specific backend first
    cap = cv2.VideoCapture(index + cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print(f"Failed with AVFOUNDATION, trying default for camera {index}...")
        cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Successfully opened camera {index}")
        return cap
    
    print(f"Could not initialize camera {index}")
    return None

def nod_yes(simulator):
    """Make drones nod up and down multiple times with large movements"""
    original_speed = simulator.speed
    simulator.speed = simulator.gesture_speed  # Use faster speed for gestures
    
    center_y = simulator.height / 2
    for _ in range(3):  # 3 nods
        # Move both drones way up
        for _ in range(3):  # Multiple updates for larger movement
            simulator.update(["UP", "UP"])
            simulator.render()
        time.sleep(0.4)  # Longer pause at top
        
        # Move both drones way down
        for _ in range(6):  # Double the movement range down
            simulator.update(["DOWN", "DOWN"])
            simulator.render()
        time.sleep(0.4)
        
        # Return to neutral position with smooth motion
        for _ in range(3):
            simulator.update(["UP", "UP"])
            simulator.render()
        time.sleep(0.3)
    
    simulator.speed = original_speed  # Restore original speed

def shake_no(simulator):
    """Make drones shake left and right multiple times with large movements"""
    original_speed = simulator.speed
    simulator.speed = simulator.gesture_speed  # Use faster speed for gestures
    
    for _ in range(3):  # 3 shakes
        # Move both drones far left
        for _ in range(3):  # Multiple updates for larger movement
            simulator.update(["LEFT", "LEFT"])
            simulator.render()
        time.sleep(0.4)  # Longer pause at extremes
        
        # Move both drones far right
        for _ in range(6):  # Double the movement range
            simulator.update(["RIGHT", "RIGHT"])
            simulator.render()
        time.sleep(0.4)
        
        # Return to neutral position with smooth motion
        for _ in range(3):
            simulator.update(["LEFT", "LEFT"])
            simulator.render()
        time.sleep(0.3)
    
    simulator.speed = original_speed  # Restore original speed

def main():
    # Initialize components
    hand_tracker = HandTracker()
    simulator = DroneSimulator()
    voice_processor = VoiceProcessor()
    llm = LLMInterface("REDACTED")
    
    # Start with camera 0
    current_camera_index = 0
    cap = init_camera(current_camera_index)
    
    print("\nCamera Controls:")
    print("'1' - Switch to camera 0")
    print("'2' - Switch to camera 1")
    print("'3' - Switch to camera 2")
    print("'v' - Voice command")
    print("'q' - Quit")
    
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                # Process hand gestures
                frame, left_gesture, right_gesture = hand_tracker.detect_gestures(frame)
                
                # Update drone positions
                simulator.update([left_gesture, right_gesture])
                
                # Display the processed frame
                cv2.imshow('Hand Tracking', frame)
        else:
            # If no camera, just update simulator with no gestures
            simulator.update([None, None])
        
        # Render simulation regardless of camera status
        simulator.render()
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Camera switching controls
        if key in [ord('1'), ord('2'), ord('3')]:
            new_index = int(chr(key)) - 1  # Convert '1','2','3' to 0,1,2
            if new_index != current_camera_index:
                print(f"\nSwitching to camera {new_index}")
                if cap is not None:
                    cap.release()
                cap = init_camera(new_index)
                current_camera_index = new_index
                if cap is None:
                    print(f"Failed to switch to camera {new_index}, trying to revert...")
                    cap = init_camera(current_camera_index)
        
        # Voice command
        elif key == ord('v'):
            print("\nListening for your question...")
            question = voice_processor.listen()
            if question:
                print(f"You asked: {question}")
                print("Processing with LLM...")
                response = llm.process_question(question)
                print(f"Response: {response}")
                
                if response == 'yes':
                    print("Nodding yes...")
                    nod_yes(simulator)
                elif response == 'no':
                    print("Shaking no...")
                    shake_no(simulator)
        
        # Quit
        elif key == ord('q'):
            break
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 