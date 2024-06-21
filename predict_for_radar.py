#! /usr/bin/env python3

import rospy
import sensor_msgs
import enum
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from custom_msgs.msg import yolo_point, yolo_points
from sensor_msgs.msg import Image



bridge = CvBridge()

class YoloDetector:
    def __init__(self, car_model_path, armor_model_path):
        self.car_model = YOLO(car_model_path)
        self.armor_model = YOLO(armor_model_path)
        
        self.armor_cls_map = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'R1', 'R2', 'R3', 'R4', 'R5', 'R7', 'N0']

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
        # cv2.imshow("car_detect", processed_frame)
        # cv2.waitKey(0)
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
                cv2.rectangle(frame, (x-w//2+ax-aw//2, y-h//2+ay-ah//2), (x-w//2+ax+aw//2, y-h//2+ay+ah//2), (0, 255, 0), 1)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x-w//2+ax-aw//2, y-h//2+ay-ah//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,127,127), 1)

class Camra(enum.Enum):
    LEFT = 0
    RIGHT = 1

class Color(enum.IntEnum):
    BLUE = 0
    RED = 1

class ModelCallbacker:
    def __init__(self, camra):
        self.camra = camra
        if camra == Camra.LEFT:
            self.pub = rospy.Publisher('/left_yolo', yolo_points, queue_size=10)
        else:
            self.pub = rospy.Publisher('/right_yolo', yolo_points, queue_size=10)
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=10)
        
        self.detector = YoloDetector("/home/inno2/ws_radar/src/radar_stationRM2024/detection_pkg/models/car_only.pt", "/home/inno2/ws_radar/src/radar_stationRM2024/detection_pkg/models/armor_only.pt")
        # self.cls_map = {}

    def __call__(self, msg):
        img = bridge.imgmsg_to_cv2(msg, "rgb8")
        points = self._generate_msg(img)
        
        self.pub.publish(points)
        print("send points successfully!")
        # rospy.loginfo("send points successfully!")
        image_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        self.image_pub.publish(image_msg)

    def _generate_msg(self, img):
        # model inference
        car_boxes = self.detector.car_detect(img)
        armor_frames = self.detector.car_2_armor(img, car_boxes)
        armors_boxes = self.detector.armors_detect(armor_frames)
        
        all_results = yolo_points()
        for i in range(len(car_boxes)):
            one_result = yolo_point()
            
            car_box = car_boxes[i]
            car_box, _, conf = car_box
            x, y, w, h = car_box
            
            one_result.x = x
            one_result.y = y
            one_result.width = w
            one_result.height = h
            
            armors_box = armors_boxes[i] if i < len(armors_boxes) else []
            if len(armors_box):
                one_result.id = armors_box[0][1]
                cls = self.detector.armor_cls_map[armors_box[0][1]]
            else:
                one_result.id = 12
                cls = 12
            # if cls[0] == 'B':
            #     one_result.color = Color.BLUE
            # elif cls[0] == 'R':
            #     one_result.color = Color.RED
            # else:
            #     one_result.color = Color.GREY
                
            one_result.color = Color.BLUE if self.detector.armor_cls_map[one_result.id][0] == 'B' else Color.RED
            
            all_results.data.append(one_result)
            
            # cv output
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 1)
            cv2.putText(img, f'car {conf:.2f}', (x-w//2, y-h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
            for box, cls, conf in armors_box:
                ax, ay, aw, ah = map(int, box)
                cv2.rectangle(img, (x-w//2+ax-aw//2, y-h//2+ay-ah//2), (x-w//2+ax+aw//2, y-h//2+ay+ah//2), (0, 255, 0), 1)
                cv2.putText(img, f'{cls} {conf:.2f}', (x-w//2+ax-aw//2, y-h//2+ay-ah//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,127,127), 1)

        all_results.text = 'car' if len(all_results.data) else 'none'
        cv2.imshow("car_detect", img)
        # cv2.waitKey(0)
        return all_results


if __name__ == '__main__':
    # rospy.init_node('yolo_node')
    # CallbackFunc(Camra.LEFT)(0)
    rospy.init_node('yolo_node') 
    rospy.Subscriber('/hikrobot_camera/left/rgb', sensor_msgs.msg.Image, ModelCallbacker(Camra.LEFT))
    rospy.Subscriber('/hikrobot_camera/right/rgb', sensor_msgs.msg.Image, ModelCallbacker(Camra.RIGHT))
    rospy.spin()
