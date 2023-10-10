import cv2
import numpy as np
import pygame
import os
def detect_features(img):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    return d
    
    
    
cap = cv2.VideoCapture('Mountain gliding on the island of Crete Greece [DUdcyVJmE1M].mp4')
success, img = cap.read()
clock = pygame.time.Clock()
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
F = float(os.getenv("F", "525"))
if os.getenv("SEEK") is not None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

clock = pygame.time.Clock()

if W > 1280:
    downscale = 1280/W
    F *= downscale
    H = int(H*downscale)
    W = 1280


K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)

wn = pygame.display.set_mode((W,H))

while success:
    clock.tick(30)
    success, img = cap.read()
    img = cv2.resize(img,(W,H))
    kps = detect_features(img)
    for i in kps:
        x,y = i.ravel()
        cv2.circle(img, (int(x), int(y)),  3, [0, 0, 255], -1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            success = False
    wn.blit(pygame.image.frombuffer(img.tobytes(), (W,H), "BGR"), (0, 0))
    pygame.display.update()

pygame.quit()
