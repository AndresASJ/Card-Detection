import cv2
from card import Card

def is_card_shaped(contour):
    """
    Determines if a contour has the shape of a playing card
    """
    area = cv2.contourArea(contour)  # area of the detected shape
    if area < 500:  # Start with a lower area threshold
        return False

    perimeter = cv2.arcLength(contour, True)  # measurement made in pixels which means 100px == 1 inch
    epsilon = 0.05 * perimeter  # Loosen the approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)  # creates the shape we're looking for

    if len(approx) != 4:  # Check if the approximated contour has 4 points
        return False

    x, y, w, h = cv2.boundingRect(approx)  # Get the bounding rectangle of the contour
    aspect_ratio = float(w) / h  # Calculate the aspect ratio
    if not (0.5 < aspect_ratio < 2.0):  # Broaden aspect ratio range
        return False

    hull = cv2.convexHull(contour)  # smallest convex shape that can enclose the contour
    hull_area = cv2.contourArea(hull)  # area of hull
    solidity = float(area) / hull_area  # compares the area of a contour to the area of its convex hull
    return solidity > 0.8  # Lower solidity threshold

def create_card(contour, frame):
    """
    Creates a Card object from a contour and frame
    """
    x, y, w, h = cv2.boundingRect(contour)  # Get the bounding rectangle of the contour
    card_image = frame[y:y + h, x:x + w]  # Extract the card image from the frame
    return Card(contour, x, y, w, h, card_image)  # Create and return a new Card object

def detect_cards(binary, frame):
    """
    Detects cards in a binary image and returns a list of Card objects
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cards = []  # List to store detected cards
    for contour in contours:  # For every point in the total shape
        if is_card_shaped(contour):  # Check if the contour is card-shaped
            card = create_card(contour, frame)  # Create a Card object
            cards.append(card)  # Add the card to the list
    
    return cards
