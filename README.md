# Card Detection

A computer vision system that detects and recognizes playing cards in real-time using OpenCV and Python. This project uses contour detection and feature matching to identify playing cards from a webcam feed, with potential applications in card games like blackjack.

## 📋 Features

- Real-time video capture from webcam
- Card detection using contour analysis
- Card recognition using ORB feature matching
- Support for multiple cards in a single frame
- Visual feedback with bounding boxes and card identification

## 🖼️ Demo

(coming soon)



## 🔧 Technologies Used

- Python 3.x
- OpenCV 4.x
- ORB (Oriented FAST and Rotated BRIEF) feature detection

## 📁 Project Structure

```
Card-Detection/
├── main.py                # Main application entry point
├── card.py                # Card class definition
├── card_detector.py       # Functions for detecting cards in images
├── image_utils.py         # Utility functions for image processing
├── template_matcher.py    # Functions for matching cards to templates
└── training data/         # Directory containing template card images
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- OpenCV 4.x

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/AndresASJ/Card-Detection.git
   cd Card-Detection
   ```

2. Install the required dependencies:
   ```
   pip install opencv-python
   ```

3. Update the template directory path in `main.py` to match your local setup:
   ```python
   directory = '/your/path/to/training data'  # Update this path to your actual path
   ```

4. Run the application:
   ```
   python main.py
   ```

## 💻 Usage

1. Point your webcam at playing cards on a contrasting background.
2. The application will detect and identify the cards in real-time.
3. Press 'p' to save the current frame as an image.
4. Press 'q' to exit the application.

## 🔍 How It Works

1. **Image Preprocessing**: Each frame from the webcam is converted to grayscale, blurred to reduce noise, and thresholded to create a binary image.

2. **Card Detection**: Contours are detected in the binary image and filtered based on:
   - Area
   - Shape approximation (quadrilateral)
   - Aspect ratio
   - Convexity

3. **Card Recognition**: Each detected card is compared against template images using ORB feature matching:
   - Keypoints and descriptors are extracted from both the card and templates
   - Brute Force Matcher with Hamming distance is used to find matches
   - Lowe's ratio test is applied to filter good matches
   - The template with the highest match score is selected as the card's identity

4. **Visualization**: The identified cards are drawn on the frame with their contours and identities.

## 🔮 Future Directions

- Integration of YOLOv5 for more robust card detection
- Improved handling of varying lighting conditions
- Performance optimization for smoother real-time processing
- Enhanced UI for displaying game information
- Implementation of blackjack game logic

## 👥 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements or bug fixes.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgements

- OpenCV documentation and community
- [Add any other resources or inspirations]
