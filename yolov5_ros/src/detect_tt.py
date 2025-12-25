#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
from sensor_msgs.msg import Image, CompressedImage
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path
# import from yolov5 submodules
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (check_img_size, non_max_suppression)

def tran2rawsz(det, imw, imh):
    Kw = imw / 640
    Kh = imh / 480
    det[:,0] = det[:,0] * Kw
    det[:,1] = det[:,1] * Kh
    det[:,2] = det[:,2] * Kw
    det[:,3] = det[:,3] * Kh
    return det

@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        print(1)
        bs = 1 #batchsize
        self.max_det = 500
        self.iou_thres = 0.3
        self.conf_thres = rospy.get_param("~conf_thresh")
        self.img_size = [480,640] # compressed; [720,960] # raw

        self.half = False
        self.classes = None
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.agnostic_nms = True

        self.bridge = CvBridge()
        self.rate = rospy.Rate(60)
        weights = rospy.get_param("~weights")
        self.view_image = rospy.get_param("~view_image")
        self.hide_label = rospy.get_param("~hide_label")
        self.window_name = rospy.get_param("~window_name")
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.engine,
        )
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.half &= self.engine and self.device.type != "cpu"
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size), half=self.half)  

        # sub & pub
        # input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("/image_topic"), blocking = True)

        self.image_sub = rospy.Subscriber("/input", Image, self.callback, queue_size=10)

        self.pred_pub = rospy.Publisher("/yolo/bbx", BoundingBoxes, queue_size=10)
        
        self.img_pub = rospy.Publisher("/yolo/img", Image, queue_size=10)
        
        print(weights)

    def callback(self, data): 
        print(1)
        # start = time.time()
        self.img_pub.publish(data) 
        
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        except CvBridgeError as e:
            print(e)

        
        im, im0 = self.preprocess(cv_img) #* im为裁剪和转换颜色通道后的图，im0为原图
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
            
        pred = self.model(im, augment=False, visualize=False) 
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        det = pred[0].cpu().numpy() #* 基于resize后尺度的结果

        annotator = Annotator(im0, line_width=2.5, example=str(self.names))
        bbs = BoundingBoxes()
        bbs.header = data.header
        bbs.image_header = data.header
        if len(det):
            det[:, :4] = tran2rawsz(det[:,:4], im0.shape[1], im0.shape[0]) #* 将结果转化为基于原尺度的坐标 
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() ##源码
            for *xyxy, conf, cls in reversed(det):
                bb = BoundingBox()
                c = int(cls)
                bb.Class = self.names[c]
                bb.probability = conf 
                bb.xmin = int(xyxy[0]) 
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bbs.bounding_boxes.append(bb)
                if self.view_image:  # Add bbox to image
                    if self.hide_label:
                        label = False
                    else:
                        label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
 
        self.pred_pub.publish(bbs) # 发的bbox在原图上的坐标

        if self.view_image:
            cv2.imshow(self.window_name, im0) ## modified in 7.8
            cv2.waitKey(1)  # 1 millisecond

        self.rate.sleep()

    def preprocess(self, cv_img):
        img0 = cv_img.copy()
        resized_img = cv2.resize(cv_img, (640, 480))
        resized_img = np.array([letterbox(resized_img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        resized_img = resized_img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        resized_img = np.ascontiguousarray(resized_img)
        return resized_img, img0 


def main(args):
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
