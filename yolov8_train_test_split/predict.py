#!/usr/bin/env python
import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, car_model_path, armor_model_path):
        self.car_model = YOLO(car_model_path)
        self.armor_model = YOLO(armor_model_path)

    def car_detect(self, frame):
        results = self.car_model(frame)
        processed_frame = frame.copy()
        processed_boxes = []
        for result in results:
            boxes = result.boxes
            for box, cls, conf in zip(boxes.xywh, map(int, boxes.cls), boxes.conf): # YOLO gives the size of boxes, the type of the object the confidence and the ID
                x, y, w, h = map(int, box)
                processed_boxes.append(([x, y, w, h], cls, conf))
                cv2.rectangle(processed_frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 1)
                cv2.putText(processed_frame, f'car {conf:.2f}', (x-w//2, y-h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
        cv2.imshow("car_detect", processed_frame)
        cv2.waitKey(0)
        return processed_boxes

    def armors_detect(self, armor_frames):
        all_boxes = []
        for armor_frame in armor_frames:
            processed_boxes = []
            results = self.armor_model(armor_frame)
            for result in results:
                if not len(result):
                    continue
                boxes = result.boxes
                for box, cls, conf in zip(boxes.xywh, map(int, boxes.cls), boxes.conf): # YOLO gives the size of boxes, the type of the object the confidence and the ID
                    x, y, w, h = map(int, box)
                    processed_boxes.append(([x, y, w, h], cls, conf))

                all_boxes.append(processed_boxes)

        return all_boxes
        
    def car_2_armor(self, car_frame, boxes):
        armor_frames = []
        for box, cls, conf in boxes:
            x, y, w, h = box
            armor_frames.append(cv2.resize(car_frame[y-h//2:y+h//2, x-w//2:x+w//2], (w, h)))
            # cv2.imshow("car_detect", armor_frames[-1])
            # cv2.waitKey(0)
        
        return armor_frames
    
    def draw(self, frame, car_boxes, armors_boxes):
        for car_box, armors_box in zip(car_boxes, armors_boxes):
            car_box, _, conf = car_box
            x, y, w, h = car_box
            cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 1)
            cv2.putText(frame, f'car {conf:.2f}', (x-w//2, y-h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
            for box, cls, conf in armors_box:
                ax, ay, aw, ah = map(int, box)
                # cls = int(cls)
                cv2.rectangle(frame, (x-w//2+ax-aw//2, y-h//2+ay-ah//2), (x-w//2+ax+aw//2, y-h//2+ay+ah//2), (0, 255, 0), 1)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x-w//2+ax-aw//2, y-h//2+ay-ah//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,127,127), 1)


            
test_detector = YoloDetector("./car_best.pt", "./armor_best.pt")
test_frame = cv2.imread("./test_frame.jpg")

car_boxes = test_detector.car_detect(test_frame)

armor_frames = test_detector.car_2_armor(test_frame, car_boxes)

armors_boxes = test_detector.armors_detect(armor_frames)

test_detector.draw(test_frame, car_boxes, armors_boxes)

cv2.imshow("car_detect", test_frame)
cv2.waitKey(0)