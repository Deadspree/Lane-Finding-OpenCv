#Limit the number of cpu used by high performance libraries
#each library focuses on one task at a time, which can be helpful for debugging or ensuring consistent behavior in your code.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_imshow, xyxy2xywh, increment_path)

from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

#Lane finding

import pyshine as ps
import lane_finding
import numpy as np
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] #Yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) #Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok =\
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith('rtsp') or source.startwith('http') or source.endswith('.txt')
    #It checks if the source of input data is from a webcam (source == '0') or if it starts with 'rtsp', 'http', or ends with '.txt'. 
    #If so, it sets the webcam variable to True, indicating that the source is a webcam stream or a file containing video URLs.

    #Initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age = cfg.DEEPSORT.MAX_AGE, n_init = cfg.DEEPSORT.N_INIT, nn_budget = cfg.DEEPSORT.NN_BUDGET,
                        use_cuda= True) #This shit is customizable using argparser
    #Inititialize
    device = select_device(opt.device)
    half &= device.type != 'cpu' #half precision only supported on Cuda

    #The MOT16 evaluation runs multiple inference streams in paralle, each on wirting to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out) # delete output folder(This is done to ensure that any existing output from previous runs is removed before starting a new run.)
        os.makedirs(out) #make new output foloder

    #Lane finding parameter
    init = True
    mtx, dist = lane_finding.distortion_factors()

    #Directories
    save_dir = increment_path(Path(project) /name, exist_ok=exist_ok) #increment run
    save_dir.mkdir(parents=True, exist_ok = True) #make dir

    #Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device = device, dnn = opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    #pt:Pytorch model parameter, jit: Whether model is in Just in time model, onnx: whether model
    #is in Open Neural Network Exchange
    #JIT mode allows you to compile PyTorch models dynamically during runtime, optimizing them for execution on specific hardware or platforms
    #It provides an interoperable framework for exchanging models between different deep learning frameworks, such as PyTorch, TensorFlow, and MXNet. 
    imgsz = check_img_size(imgsz, s=stride) #check image size to ensure it meets certain requirements specified by the stride parameter. 

    #half
    half &= pt and device.type != 'cpu' #half precision only supported by Pytorch on Cuda
    #used in scenarios where memory usage or bandwidth is a concern, such as in graphics processing units (GPUs)
    #, machine learning accelerators, and deep learning frameworks
    if pt:
        model.model.half() if half else model.model.float()#whether the model is in JIT mode and the selected device is not a CPU.

    #Set Dataloader
    vid_path, vid_writer = None, None
    #Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    #DataLoader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True #Set True to spped up constant image size inference
        dataset = LoadStreams(source, img_size = imgsz, stride =stride, auto = pt and not jit)
        bs = len(dataset) # batch_size

    else:
        dataset = LoadImages(source, img_size = imgsz, stride= stride, auto=pt and not jit)
        bs = 1 #batch size
    vid_path, vid_writer = [None] * bs, [None] * bs
    #This line initializes vid_path and vid_writer as lists of length equal to the batch size (bs). 
    #Each element of these lists is initialized to None

    #Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # If model has a module attribute, it means the model is wrapped in a module
    #In this case, model.module.names retrieves the names of the classes from the module attribute.
    #  If the model object does not have a module attribute, it means the model is not wrapped in a module, and model.names retrieves the names of the classes directly from the model object.

    #exttract what is in between the last '/' and last'.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1,3, *imgsz).to(device).type_as(next(model.model.parameters()))) #warm up
        # The dimensions of the tensor are (batch_size, channels, *imgsz)
        #next(model.model.parameters()): This converts the data type of the tensor to match the data type of the parameters of the YOLOv5 model (model).
        # It does this by accessing the next parameter in the model's parameter list (next(model.model.parameters())) and using its data type as a reference.
        # model(...): This performs a forward pass through the YOLOv5 model with the input tensor.
    dt, seen = [0.0,0.0,0.0,0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()#Records the start time before preprocessing the image.
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() #uint 8 to fp16 or fp 32
        img /= 255.0
        if img.ndimension() == 3:
            time = img.unsqueeze(0)
        #Checks if the image tensor has three dimensions (indicating a single image). If it does, it adds an extra dimension at the beginning to represent the batch dimension. 
        #This step is necessary to ensure compatibility with models that expect batched input.
        t2 = time_sync()#Records the end time after preprocessing the image
        dt[0] += t2 - t1 #Computes the elapsed time for preprocessing the image and adds it to the list dt at index 0.

        #Inference
        visualize = increment_path(save_dir/Path(path).stem, mkdir=True) if opt.visualize else False
        #It checks the value of the opt.visualize flag. If it's True, it creates a directory for saving visualization images using increment_path,
        # otherwise, it sets visualize to False.
        pred = model(img, augment=opt.augment, visualize = visualize)
        t3 = time_sync()#Records the end time after performing inference.
        dt[1] += t3 - t2 #omputes the elapsed time for inference and adds it to the list dt at index 1.

        #Apply NMS(NMS removes redundant bounding boxes by suppressing those with low confidence scores or high overlap (IoU) with higher-scoring boxes.)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det = opt.max_det)
        #opt.conf_thres: This parameter specifies the confidence threshold. 
        #Predictions with confidence scores below this threshold will be discarded during NMS.
        #opt.iou_tthres: This parameter specifies the IoU (Intersection over Union) threshold. 
        #Bounding boxes with IoU greater than this threshold will be considered redundant during NMS.
        #opt.classes: This parameter contains the list of class names or IDs that the model can detect. 
        #Only predictions corresponding to these classes will be considered during NMS.
        #opt.agnostic_nms: This parameter controls whether class-agnostic NMS is applied. 
        #If set to True, NMS will be applied independently for each class, ignoring class information.( This means that bounding boxes from different object classes are treated equally during suppression.)
        #max_det=opt.max_det: This parameter specifies the maximum number of detections to keep after NMS. 
        #If there are more detections than this limit, only the top detections based on confidence scores will be retained.


        dt[2] += t3-t2 #This line computes the elapsed time for applying NMS and adds it to the list dt at index 2.


        #Process detections
        for i,det in enumerate(pred):
            #iterates over the predictions generated by the model (pred)
            #Each prediction (det) likely contains information about detected objects, such as bounding box coordinates, confidence scores, and class labels.

            seen += 1 #Increments the counter for the number of processed frames (seen).
            if webcam: #batch_size >=1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count # the code retrieves frame-related information (p, im0, _) based on the index i.
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0) #it assumes a single frame processing scenario and retrieves frame-related information (p, im0) accordingly.


            p = Path(p) # Converts the file path (p) to a Path object.
            save_path = str(save_dir / p.name) #im.jpg, vid.mp4, ...#Constructs the save path for the processed frame based on the output directory (save_dir) and the filename (p.name).
            s += "%gx%g" % img.shape[2:] #print string
            

            if seen ==1:#nitializes the video writer (vid_writer) accordingly.
                if isinstance(vid_writer, cv2.VideoWriter):#If the video writer already exists (isinstance(vid_writer, cv2.VideoWriter))
                    vid_writer.relsease() #release previous video writer
                if vid_cap: #VIf the input source is a video file (vid_cap),
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                else: #If the input source is a video stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0] # it sets default frame rate (fps), width (w), and height (h) values.

                ratio = w/200000

                vid_writer = cv2.VideoWriter('./yolo5/data/output/test_sample_result.avi', 
                                             cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
            
            #Lane finding

            img_out, angle, colorwarp, draw_poly_img = lane_finding.lane_finding_pipeline(im0, init, mtx, dist)

            if angle>1.5 or angle <-1.5: #If the angle is too steep, it indicates a significant change in lane direction-like when the car turn
                init = True
            else:
                init=False

            annotator = Annotator(img_out, line_width=2, pil = not ascii)
            #bject configured to annotate the provided image (img_out) with a line width of 2 pixels, 
            #and it specifies whether to use the Python Imaging Library based on the image format indicated by the ascii variable.
                
            if det is not None and len(det):
                #Rescale boxes from ims_size to im0size
                det[:,:4] = scale_boxes(img.shape[2:], det[:,:4], im0.shape).round()

                #Print results
                for c in det[:,:-1].unique():
                    n = (det[:, :-1]==c).sum() #detection per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4] #Extract confident score from detection
                clss = det[:, 5] #Ectract class ID from detection

                #Initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                #

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                #The detections, along with their bounding box coordinates, confidence scores, and class IDs, 
                #are passed to the DeepSORT tracker (deepsort.update)
                t5 = time_sync()
                dt[3] += t5 - t4
                #draw boxes for visualization

                if len(outputs) > 0: #If there are outputs from the DeepSORT tracker 
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = outputs[5]

                        #color with cmap

                        color_ = colors[int(id) % len(colors)]#This line assigns a color to the bounding box based on the object's ID (id)
                        color_ = [i * 255 for i in color_] #This line scales each RGB value in the selected color to the range [0, 255]. 

                        c = int(cls) #inter class
                        label = f'{id} {names[c]} {conf:.2f}'  
                        annotator.box_label(bboxes, label, color=color_) #The box_label function is likely responsible for drawing the label on the image.

                        #Distance calculation
                        if cls == 2 or cls == 5 or cls == 7:#vehicles like cars, buses, and trucks)
                            mid_x = (bboxes[0]+bboxes[2])/2
                            mid_y = (bboxes[1]+bboxes[3])/2
                            apx_distance = round((((h-bboxes[3]))*ratio)*4.5,1)#height of image - height of bounding box * a certain ratio
                            mid_xy = [mid_x, mid_y]
                            #update for text input
                            annotator.dist(mid_xy, apx_distance)

                            if apx_distance <=1:#if too close, makes a warning
                                if (mid_x) > w*0.3 and (mid_x) < w*0.7:
                                    warn_xy = [400, 150]
                                    annotator.dist(warn_xy, cls, id)

                        if save_txt:
                            #to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]-output[0]
                            bbox_h = output[3]-output[1]
                            #Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g' *10 +'\n') %(frame_idx +1, id, bbox_left, #MOT format
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1) )
                                #the detected object's bounding box coordinates and other relevant information are formatted in MOT (Multiple Object Tracking) compliant format
                                # and appended to a text file (txt_path). 
                LOGGER.info(f'{s}Done. YOLO:({t3-t2:.3f}s), DeepSort:({t5-t4:.3f}s)')
                #This line logs an informational message
            
            else: # If there are no detections
                deepsort.increment_ages()
                #For each frame where an object is successfully associated with a track, the age of the track is incremented by 1. 
                #This indicates how many consecutive frames the object has been tracked without interruption.

                LOGGER.info('No detections')
            

            if show_vid:

                cv2.imshow('result', im0)

                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
                
    #Print results
    t = tuple(x / seen * 1E3 for x in dt) # speeds per image

    if save_txt or save_vid:
        print('Results saved to %s' % save_path) #he program prints a message indicating that the results are saved to a specified path (save_path).
        if platform == 'darwin': # MacOS
            os.system('open ' + save_path)

#How to use argumentParser(https://www.youtube.com/watch?v=88pl8TuuKz0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser() #It provides the option of changing the input using command line instead of changing directly in the code
    parser.add_argument('--yolo_model', nargs='+', type = str, default='yolov5m.pt', help = 'model.pt path(s)')
    #nargs='+': This specifies that the argument can accept one or more values
    #help='model.pt path(s)': This provides a brief description of the argument that will be displayed when the user requests help information for your script. It helps users understand what the argument is for and how to use it.
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    #If not specify in terminal, is gonna be false
    #if specify is gonna be true, is NEVER None
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1 
    #if opt.imgsz is a single value, it will be multiplied by 2, effectively doubling its value. 
    #If opt.imgsz is already a tuple of two values, its value remains unchanged.
    #ensuring that opt.imgsz is a tuple of two values (width and height) rather than a single value
    with torch.no_grad():
        detect(opt)
    #he torch.no_grad() context manager is used in PyTorch to disable gradient calculation, 
    #which is useful during inference when you don't need to compute gradients for optimization
    #Save computational power







