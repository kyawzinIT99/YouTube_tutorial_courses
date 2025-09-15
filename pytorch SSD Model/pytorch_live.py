import torch
import torchvision
from torchvision import transforms as T
import cv2
import cvzone

# Load pre-trained model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

# Load class names (COCO dataset classes)
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Open webcam
cap = cv2.VideoCapture(0)  # use 0 for built-in camera

# Define torchvision transform
transform = T.ToTensor()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured, check camera.")
        break

    img = frame.copy()
    # Convert frame (H,W,C) → Tensor (C,H,W)
    tensor_img = transform(frame)

    # Inference
    with torch.no_grad():
        ypred = model([tensor_img])

        bbox, scores, labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']
        nums = torch.argwhere(scores > 0.60).shape[0]  # threshold = 0.6

        for i in range(nums):
            x, y, w, h = bbox[i].numpy().astype('int')
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

            classname = labels[i].item()
            classdetected = classnames[classname - 1] if classname - 1 < len(classnames) else "Unknown"
            cvzone.putTextRect(img, classdetected, [x, y - 10], scale=1, thickness=2, border=2)

    # Show output
    cv2.imshow('Live Detection', img)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()