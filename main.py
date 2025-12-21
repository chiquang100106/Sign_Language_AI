import cv2
import mediapipe as mp
import numpy as np


#Khởi tạo các thành phần của mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

#Tỷ lệ khung hình cho bouding box
DESIRED_ASPECT_RATIO = 1.0  # tỷ lệ khung hình 1:1
PADDING = 40 # Padding 40 cho bounding box

#Threshold cho detect chuyển động của tay
STILLNESS_THRESHOLD = 5

def process_frame(frame):
    '''
        Hàm Detect bàn tay
        :arg frame:
        :return:
    '''
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)

def calculate_bounding_box(hand_landmarks, frame_shape):
    '''
    Hàm tính bounding box
    :param hand_landmarks:
    :param frame_shape:
    :return:
    '''
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)

    # Add padding to the bounding box
    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(w, x_max + PADDING)
    y_max = min(h, y_max + PADDING)

    return x_min, y_min, x_max, y_max

#Đưa bounding box về dạng hình vuông
def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, desired_aspect_ratio):
    h, w, _ = frame_shape
    box_width = x_max - x_min
    box_height = y_max - y_min
    current_aspect_ratio = box_height / box_width

    if current_aspect_ratio < desired_aspect_ratio:
        # Height is too small, adjust the height
        new_height = int(box_width * desired_aspect_ratio)
        y_center = (y_min + y_max) // 2
        y_min = max(0, y_center - new_height // 2)
        y_max = min(h, y_center + new_height // 2)
    elif current_aspect_ratio > desired_aspect_ratio:
        # Width is too small, adjust the width
        new_width = int(box_height / desired_aspect_ratio)
        x_center = (x_min + x_max) // 2
        x_min = max(0, x_center - new_width // 2)
        x_max = min(w, x_center + new_width // 2)

    return x_min, y_min, x_max, y_max

def draw_bounding_box_and_landmarks(frame, x_min, y_min, x_max, y_max, hand_landmarks):
    # Draw the bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw landmarks on the frame
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Function to crop the hand region from the frame
def crop_hand(frame, x_min, y_min, x_max, y_max):
    return frame[y_min:y_max, x_min:x_max]


# Function to check if the hand is the right hand
def is_right_hand(handedness):
    return handedness.classification[0].label == 'Left'


# Function to detect if the hand is moving
def is_hand_moving(current_landmarks, previous_landmarks, threshold, frame_shape):
    if previous_landmarks is None:
        return True  # First frame, no previous landmarks to compare

    h, w, _ = frame_shape
    total_movement = 0
    num_landmarks = len(current_landmarks)

    for lm_current, lm_previous in zip(current_landmarks, previous_landmarks):
        x_current, y_current = int(lm_current.x * w), int(lm_current.y * h)
        x_previous, y_previous = int(lm_previous.x * w), int(lm_previous.y * h)

        # Calculate the Euclidean distance between the corresponding landmarks
        distance = np.sqrt((x_current - x_previous) ** 2 + (y_current - y_previous) ** 2)
        total_movement += distance

    # Average movement per landmark
    average_movement = total_movement / num_landmarks

    return average_movement > threshold  # Hand is moving if movement is greater than threshold

def main():
    #
    cap = cv2.VideoCapture(0)

    previous_hand_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Detect tay
        result = process_frame(frame)
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Detect only the right hand
                if is_right_hand(handedness):
                    # Check if the hand is moving
                    if is_hand_moving(hand_landmarks.landmark,previous_hand_landmarks, STILLNESS_THRESHOLD,
                                      frame.shape):
                        # Calculate the bounding box
                        x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)

                        # Enforce fixed aspect ratio for the bounding box
                        x_min, y_min, x_max, y_max = enforce_aspect_ratio(
                            x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO
                        )

                    else:
                        # Calculate the bounding box
                        x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)

                        # Enforce fixed aspect ratio for the bounding box
                        x_min, y_min, x_max, y_max = enforce_aspect_ratio(
                            x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO
                        )
                        # Crop the hand region
                        hand_crop = crop_hand(frame, x_min, y_min, x_max, y_max)
                        # Show the cropped hand
                        # cv2.imshow("Cropped Right Hand", hand_crop)
                        hand_crop = cv2.resize(hand_crop, (28, 28)) #size input 28x28

                    draw_bounding_box_and_landmarks(frame, x_min, y_min, x_max, y_max, hand_landmarks)
                    previous_hand_landmarks = hand_landmarks.landmark
        cv2.imshow('hand detect', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
