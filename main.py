import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *
import time

list_of_vehicles = ["bicycle","car","motorbike","bus","truck", "train"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416
Incout=1
Framecount=0
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()

line = [(23, 543), (650, 543)]

line1 = [(400, 350), (650, 350)]
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
from Sort import *
tracker = Sort()
memory = {}
SingleFrameout=0
CurrentInCountList=[0,1,0,0,0]
def displayVehicleCount(frame, vehicle_count,CurrentVehicleListCount,Framecount,CurrentVehicleListInCount,CurrentVehicleOutListCount,vehicle_outcount):
        j=20
        global counter,Incout,CurrentInCountList
        timer=0.0
        print(CurrentVehicleListInCount)
        print(CurrentVehicleOutListCount)
        for i in range(len(CurrentInCountList)):
                CurrentInCountList[i]=(CurrentInCountList[i]+CurrentVehicleListInCount[i])-CurrentVehicleOutListCount[i]
                if(CurrentInCountList[i]==-1):
                        CurrentInCountList[i]=0
        vehicepriority=[0.12,0.13,0.02,0.03,0.4,0.9]
        if(Framecount%30==0):
                for j in range(len(CurrentVehicleListCount)-1):\
                    timer=timer+((CurrentInCountList[j])*vehicepriority[j])
                
        if(Framecount%30==0):
                time.sleep(2)
                cv2.putText(
		frame, #Image
		'Traffic Signal Timer  ON: ' + str(timer), #Label
		(600, 50), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(255, 255, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
                time.sleep(5)
                
                
       
        cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
        cv2.putText(
		frame, #Image
		'In Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
       
        for i in  range(len(CurrentInCountList)):
                j=j+20;
                cv2.putText(
		frame, #Image
		'On Detected '+list_of_vehicles[i]+': ' + str(CurrentInCountList[i]), #Label
		(20, j), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.5, #Size
		(0,0,0xFF), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
        cv2.putText(
		frame, #Image
		'Out Vehicles Count: ' + str(counter), #Label
		(20, 150), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0xFF, 255 , 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
        cv2.putText(
		frame, #Image
		'Total In Vehicles Count: ' + str(Incout), #Label
		(600, 150), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(255, 255 , 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
       

def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
	x1_line, y1_line, x2_line, y2_line = line_coordinates #Unpacking

	if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
		(y_mid_point >= y1_line and y_mid_point <= y2_line+5):
		return True
	return False


def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		os.system('clear') # Equivalent of CTRL+L on the terminal
		print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames


def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
        dets = []
        bicycle=0
        car=0
        motorbike=0
        bus=0
        truck=0
        train=0
        bicyclein=0
        carin=0
        motorbikein=0
        busin=0
        truckin=0
        
        global memory,counter,Incout
        if len(idxs) > 0:
                for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        dets.append([x, y, x+w, y+h, confidences[i]])
                        
                        
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        print(previous)
        for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                memory[indexIDs[-1]] = boxes[-1]
        #print(memory)
        if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
                        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                        cv2.rectangle(frame, (x, y), (w, h), color, 2)
                        if indexIDs[i] in previous:
                                
                                previous_box = previous[indexIDs[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                #print(x2, y2,w2, h2)
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                cv2.line(frame, p0, p1, color, 3)
                                
                                if intersect(p0, p1, line[0], line[1]):
                                        print(p0, p1, line[0], line[1])
                                        counter += 1
                                        if(LABELS[classIDs[i]]=="bicycle"):
                                                bicycle=bicycle+1
                                        elif(LABELS[classIDs[i]]=="car"):
                                                car+=1
                                        elif(LABELS[classIDs[i]]=="motorbike"):
                                                motorbike+=1
                                        elif(LABELS[classIDs[i]]=="bus"):
                                                bus+=1
                                        elif(LABELS[classIDs[i]]=="truck"):
                                                truck+=1
                                        elif(LABELS[classIDs[i]]=="train"):
                                                train+=1
                                        
                                if intersect(p0, p1, line1[0], line1[1]):
                                        Incout += 1
                                        if(LABELS[classIDs[i]]=="bicycle"):
                                                bicyclein=bicyclein+1
                                        elif(LABELS[classIDs[i]]=="car"):
                                                carin+=1
                                        elif(LABELS[classIDs[i]]=="motorbike"):
                                                motorbikein+=1
                                        elif(LABELS[classIDs[i]]=="bus"):
                                                busin+=1
                                        elif(LABELS[classIDs[i]]=="truck"):
                                                truckin+=1
                                      
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        i += 1
        cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
        cv2.line(frame, line1[0], line1[1], (0, 255, 255), 5)
        CurrentVehicleListOutCount=[bicycle,car,motorbike,bus,truck,train]
        CurrentVehicleListInCount=[bicyclein,carin,motorbikein,busin,truckin]
        
        return CurrentVehicleListInCount,CurrentVehicleListOutCount
        


def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)


def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
counter=0
def count_vehicles(idxs, boxes, classIDs, vehicle_count,vehicle_outcount, previous_frame_detections, frame,previous):
        current_detections = {}
        list_of_vehicles_count = ["bicycle","car","motorbike","bus","truck", "train"]
        bicycle=0
        car=0
        motorbike=0
        bus=0
        truck=0
        train=0
        
        if len(idxs) > 0:
                for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        centerX = x + (w//2)
                        centerY = y+ (h//2)
                        

				
                        if (LABELS[classIDs[i]] in list_of_vehicles):
                                
                                current_detections[(centerX, centerY)] = vehicle_count
                                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                                        vehicle_count += 1
                                       
                                
                               
                                        
                                if(LABELS[classIDs[i]]=="bicycle"):
                                        bicycle=bicycle+1
                                elif(LABELS[classIDs[i]]=="car"):
                                        car+=1
                                elif(LABELS[classIDs[i]]=="motorbike"):
                                        motorbike+=1
                                elif(LABELS[classIDs[i]]=="bus"):
                                        bus+=1
                                elif(LABELS[classIDs[i]]=="truck"):
                                        truck+=1
                                elif(LABELS[classIDs[i]]=="train"):
                                        train+=1
                                ID = current_detections.get((centerX, centerY))
                                if (list(current_detections.values()).count(ID) > 1):
                                        current_detections[(centerX, centerY)] = vehicle_count
                                        vehicle_count += 1
                                cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
        CurrentVehicleListCount=[bicycle,car,motorbike,bus,truck,train]
        return vehicle_count,vehicle_outcount, current_detections,CurrentVehicleListCount


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = None
video_height =None



#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count,vehicle_outcount = 0, 0,0
writer = None
start_time = int(time.time())
while True:
	print("================NEW FRAME================")
	
	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], []
	
	vehicle_crossed_line_flag = False
	previous = memory.copy()
	Framecount=Framecount+1

	
	(grabbed, frame) = videoStream.read()
        
	
	if not grabbed:
		break
	if video_width is None or video_height is None:
		(video_height, video_width) = frame.shape[:2]
	
	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for i, detection in enumerate(output):
			
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			

			
			if confidence > preDefinedConfidence:
				
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
                            
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	
	CurrentVehicleListInCount,CurrentVehicleListOutCount=drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

	vehicle_count,vehicle_outcount, current_detections,CurrentVehicleListCount = count_vehicles(idxs, boxes, classIDs, vehicle_count,vehicle_outcount, previous_frame_detections, frame,previous)

	
                
                
	displayVehicleCount(frame, vehicle_count,CurrentVehicleListCount,Framecount,CurrentVehicleListInCount,CurrentVehicleListOutCount,vehicle_outcount)
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/output.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		

	# write the output frame to disk
	writer.write(frame)

   
	

	cv2.imshow('Frame', frame)
        
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
	
	previous_frame_detections.pop(0) #Removing the first frame from the list
	
	previous_frame_detections.append(current_detections)


print("[INFO] cleaning up...")
writer.release()
videoStream.release()
