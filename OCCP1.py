import cv2
from ultralytics import YOLO
import imutils

# Load the YOLOv8 model
model = YOLO("./model/l8x32.pt")
dim = (640, 640)
# Open the video
cap = cv2.VideoCapture(0)
cap2 = cap
cap2.set(cv2.CAP_PROP_EXPOSURE, -15)
cap2.set(cv2.CAP_PROP_BRIGHTNESS, 60)
cap2.set(cv2.CAP_PROP_SATURATION, 40)
cap2.set(cv2.CAP_PROP_CONTRAST, -10 )
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    _, filter = cap2.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                x1 = r[1]-20
                y1 = r[3]+20
                x2 = r[0]-20
                y2 = r[2]+20
                crop = cv2.resize(frame[x1:y1, x2:y2],dim)
                gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                print(thres.shape)
                contours= cv2.findContours(thres.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                contours = sorted(contours, key=cv2.contourArea, reverse=False)
                number = 0
                ma =[[],[],[],[]]
                td =[] 
                for c in contours:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if (3<w<25) and(3<h<25):
                        ma[0].append(x)
                        ma[1].append(y)
                        a =[x+w/2,y+h/2] 
                        td.append(a)
                        ma[2].append(w)
                        ma[3].append(h)
                        number +=1
                print(ma[0])
                print(ma[1])
                print(ma[2])
                print(ma[3])
                xmin = min(ma[0])
                xmax = max(ma[0])
                ymin = min(ma[1])
                ymax = max(ma[1])
                print('ymin==',ymin)
                print('ymax==',ymax)
                bit=[]
                dx = (xmax-xmin)/(8-1)
                dy = (ymax-ymin)/(8-1)
                c = ymin
                while c <= ymax+1: 
                    b = xmin
                    while b <= xmax+1:
                        for j in td:
                            if b <= j[0] <= b+dx and c <= j[1] <= c+dy:
                                d=1
                                break
                            else:
                                d=0
                        bit.append(d)
                        b = b+dx
                    c +=dy

        print("Number of Contours found = " + str(number))
        tem = bit[8:16]
        tem1 = sum(val*(2**idx) for idx, val in enumerate(reversed(tem)))
        hum = bit[16:24]
        hum1 = sum(val*(2**idx) for idx, val in enumerate(reversed(hum)))
        if bit[0:8]==[1,0,0,1,1,0,0,1]:
            cv2.putText(crop, "Temperature: " + str(tem1) +"*C", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(crop, "Humidity: " + str(hum1) + "%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)        
        #cv2.imshow('frame',frame)
        # Display the annotated frame
        cv2.imshow("Crop", crop)
        cv2.imshow("Filter", filter)
        cv2.imshow("Thres", thres)
        cv2.imshow("Frame", frame)
        #cv2.imshow("Edges", contours)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
