# Necessary libraries
import cv2 as cv
import torch  # framework for yolov5

#define autocast as True or False
autocast = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# check if cuda is available, if not use cpu
with torch.amp.autocast(device, enabled=autocast):
# Initialize small model for testing
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    
# Open the webcam (default camera 0)
    cap = cv.VideoCapture(0)

# Handle error if the camera can't be opened
if not cap.isOpened():
    print("Can't open camera")
    exit()

# Format input to fit with YOLOv5 requirements
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for video output
demo = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Video writer

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Cannot receive frame (maybe end of stream). Exiting...")
        break

    # Flip the frame horizontally for a mirror effect (optional)
    frame = cv.flip(frame, 1)

    # Convert the frame to RGB (since OpenCV uses BGR)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Convert the frame to a PyTorch tensor
    frame_tensor = torch.from_numpy(frame_rgb).float()

    # Normalize tensor to [0, 1] range
    frame_tensor /= 255.0

    # Add a batch dimension to the tensor (1, 3, H, W)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 480, 640]

    # Run inference (object detection) with YOLOv5
    results = model(frame_tensor)

    # Display resulting frame
    cv.imshow('frame', frame)

    # Write the frame to the output video file
    demo.write(frame)

    # Exit loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
demo.release()  # Release the video writer
cv.destroyAllWindows()

