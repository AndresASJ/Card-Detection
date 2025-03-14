import cv2

def orb_feature_matching(card_image, templates):
    """
    Matches a card image against templates using ORB feature matching
    """
    card_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=2000)  # Increase the number of features for better accuracy

    best_match = None
    highest_score = 0

    for template_name, template_image in templates:  # Iterate through all template images
        # Find keypoints and descriptors in both images
        keypoints1, descriptors1 = orb.detectAndCompute(template_image, None)  # The reference template
        keypoints2, descriptors2 = orb.detectAndCompute(card_image, None)  # The source to compare

        if descriptors1 is None or descriptors2 is None:
            print(f"Warning: Descriptors not found for template {template_name} or card image.")
            continue  # Skip to the next template if descriptors are missing

        # Initialize Brute Force Matcher
        # BruteForceMatcher compares descriptors from the template_image to the source_image
        # and returns the closest match
        # NORM_HAMMING is better for binary descriptors
        # crossCheck reverses the check, ie, A checks B then crosscheck does B check A
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Compares descriptors | and k=2 refers to comparing 2 at a time
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Do the ratio test for filtering good matches
        good_matches = []  # List to store good matches
        for match in matches:
            if len(match) == 2:  # Ensure we have two matches for the ratio test
                m, n = match
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)  # add good matches to the list

        score = len(good_matches) / max(len(keypoints1), len(keypoints2))  # Calculate matching score

        if score > highest_score:
            highest_score = score
            best_match = template_name.split('.')[0]  # Assuming filename is "AceOfSpades.jpg"

    return best_match, highest_score

def identify_card(card, templates):
    """
    Identifies a card by matching it against templates
    """
    identity, confidence = orb_feature_matching(card.image, templates)  # Match the card against templates
    card.update_identity(identity, confidence)  # Update the card's identity and confidence
