import cv2
import os
from card import Card
from card_detector import detect_cards
from image_utils import load_images_from_directory, preprocess_frame
from template_matcher import identify_card

def main():
    """
    Main application entry point
    """
    print('OpenCV has opened')
    
    # Load template images
    directory = '/home/asj/mlblackjack/mlblackjack/training data'  # Update this path to your actual path
    templates = load_images_from_directory(directory)
    
    print(f"Loaded {len(templates)} template images:")
    for template_name, _ in templates:
        print(template_name)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Initialize video capture from the default camera
    count = 0  # used for editing the name of the saved frame
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:  # Main loop for processing frames
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Preprocess the frame
            binary = preprocess_frame(frame)
            
            # Detect cards in the frame
            cards = detect_cards(binary, frame)
            
            # Identify each detected card
            for card in cards:
                identify_card(card, templates)
                card.draw_on_frame(frame)
            
            # Flip the frame horizontally for a more intuitive display
            processed_frame = cv2.flip(frame, 1)
            
            # Display the processed frame
            cv2.imshow('Frame', processed_frame)
            
            # Check for keystroke
            key = cv2.waitKey(1)
            
            if key == ord('p'):  # Save the current frame when 'p' is pressed
                cv2.imwrite(f'photocv{count}.png', processed_frame)
                count += 1
                print(f"Frame saved as photocv{count-1}.png")
            
            if key == ord('q'):  # Exit the loop when 'q' is pressed
                print("You have exited the program.")
                break
    
    finally:
        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Destroy all windows
        print("OpenCV basics demonstration finished.")

if __name__ == "__main__":
    main()
