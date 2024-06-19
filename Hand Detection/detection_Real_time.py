import cv2 
import torch 
import torchvision 


#making a yolo detector using pytorch 
hand_detector=torch.hub.load('ultralytics/yolov5','custom',path='best.pt')


img=cv2.imread('pexels-kevin-malik-9017018.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


output = hand_detector(img)

print(f'prediction: {output.pred}')
# output.show()

#now opening the webcam and applying the detetcion model 
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide the device index for other cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference using the YOLOv5 model
    results = hand_detector(frame_rgb)
    
    #resnet18 model(Sign MNIST)

    # Draw bounding boxes and labels on the frame
    results.render()  # This will draw the bounding boxes and labels on the frame
    
    # Get the frame with the drawn detections
    result_frame = results.ims[0]
    
    #inverting the image 
    # Display the frame in a window
    cv2.imshow('YOLOv5 Detection', result_frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()