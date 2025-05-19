import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('crowd.jpg')
people_boxes = [
    (100, 100, 50, 100), (200, 120, 50, 100), (300, 150, 50, 100),
    (400, 180, 50, 100), (500, 200, 50, 100), (600, 100, 50, 100),
    (150, 300, 50, 100), (250, 330, 50, 100), (350, 360, 50, 100),
    (450, 390, 50, 100), (550, 420, 50, 100), (650, 450, 50, 100)
]
for (x, y, w, h) in people_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
crowd_count = len(people_boxes)
cv2.putText(image, f"Estimated Crowd: {crowd_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()