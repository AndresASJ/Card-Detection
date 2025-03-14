import cv2

class Card:
    def __init__(self, contour, x, y, w, h, image):
        self.contour = contour  # The outline of the card
        self.x = x  # x-coordinate of the top-left corner of the card
        self.y = y  # y-coordinate of the top-left corner of the card
        self.w = w  # Width of the card
        self.h = h  # Height of the card
        self.image = image  # Image of the card
        self.identity = None  # it should describe what the card is, ie: Ace of Spades, 4 of hearts, etc
        self.confidence = 0.0  # Confidence score of the card identification

    def update_identity(self, identity, confidence):
        self.identity = identity  # Update the identity of the card
        self.confidence = confidence  # Update the confidence score

    def draw_on_frame(self, frame):
        cv2.drawContours(frame, [self.contour], -1, (0, 255, 0), 2)  # Draw the card's outline
        cv2.putText(frame, f"{self.identity} ({self.confidence:.2f})",
                    (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)  # Write the card's identity above it
