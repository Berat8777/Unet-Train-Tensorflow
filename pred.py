#! /usr/bin/python3

#Ali Berat Algün

import cv2
from keras.models import load_model
import time
import pandas as pd
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from simple_pid import PID
import time

time.sleep(1)
fps_start_time = 0
fps = 0


def calculate_angle(A, B, C):
    # Vektörleri oluştur
    AB = B - A
    AC = C - A

    # Vektörlerin normalizasyonu
    AB_normalized = AB / np.linalg.norm(AB)
    AC_normalized = AC / np.linalg.norm(AC)

    # İç çarpımı al
    dot_product = np.dot(AB_normalized, AC_normalized)

    # Arccos fonksiyonu ile açıyı hesapla
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def oranlayici(eski_aralik, yeni_aralik, deger):
    eski_min, eski_max = eski_aralik
    yeni_min, yeni_max = yeni_aralik
    
    # Eski aralıktaki değerin normalize edilmiş halini hesaplayın
    normalize_deger = (deger - eski_min) / (eski_max - eski_min)
    
    # Yeni aralıktaki değeri hesaplayın
    yeni_deger = yeni_min + (normalize_deger * (yeni_max - yeni_min))
    
    return yeni_deger


eski_aralik = (-90, 90)
yeni_aralik = (-1, 1)

model = load_model("/home/berat/Desktop/Modeller/512Berat.h5", compile=False)  #Derleme işlemini devre dışı bırakıyoruz çünkü bu zaten hazır bir model
size= 512
bridge = CvBridge()

cap = cv2.VideoCapture("/home/berat/Desktop/test/challenge_video.mp4")
if cap.isOpened() == 0:
    exit(-1)

while True:
    
    ret, frame = cap.read()
    basla = time.perf_counter()

    fps_end_time =time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (5,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    
    frame=cv2.resize(frame,(size,size))  
    img = frame.copy()
    img = img/255
    img = np.expand_dims(img,axis=0)  
    
    pred = model.predict(img)         
    
    pred = pred.reshape(size,size,1)   
    

    h, w, d = pred.shape             
    image=frame.copy()              
    
    mask0=pred.copy()                 
    mask6=pred.copy()                   
    #segmentasyon=mask.copy()
    sagserit=pred.copy()
    solserit=pred.copy()
   


    # image roi in front of the camera
    #search_top0 = h//2           #512              
    #search_bot0 = h//2 + 100     #612   
    #mask0[0:search_top0, 0:w] = 0       
    #mask0[search_bot0:h, 0:w] = 0      
    mask_sol_karsi =mask0.copy()     
    mask_sag_karsi =mask6.copy()       
    

    def mask_of_sol(mask_sol_karsi):
        mask_sol_karsi=cv2.resize(mask_sol_karsi,(512,512))
        polygons = np.array([[(0, 290), (197, 290), (52, 290)],[(197,290),(0,290),(0,512)],[(197,512),(0,512),(197,290)]]) 
        mask = np.zeros_like(mask_sol_karsi) 
        cv2.fillPoly(mask, polygons, 1) 
        mask_sol_karsi = cv2.bitwise_and(mask_sol_karsi, mask)  
        return mask_sol_karsi
    
    def mask_of_sag(mask_sag_karsi):
        mask_sag_karsi=cv2.resize(mask_sag_karsi,(512,512))
        polygons = np.array([[(440,290),(512,290),(315,290)],[(315,290),(512,290),(512,512)],[(315,512),(512,512),(315,290)]]) 
        mask = np.zeros_like(mask_sag_karsi)  
        cv2.fillPoly(mask, polygons, 1)   #Biz resmi 255e böldük o yüzden 1 olmalı burası 
        mask_sag_karsi = cv2.bitwise_and(mask_sag_karsi, mask)  
        return mask_sag_karsi
    
    
    cv2.line(image,(0,290), (52,290), (69,15,28), 3)      
    cv2.line(image,(52,290),(197,290),(69,15,28), 3)
    cv2.line(image,(197,290),(197,512),(69,15,28),3)
    
    cv2.line(image,(315,290),(440,290),(69,15,28),3)
    cv2.line(image,(512,290),(440,290),(69,15,28),3)
    cv2.line(image,(315,290),(315,512),(69,15,28),3)

    solserit=mask_of_sol(mask_sol_karsi)
    sagserit=mask_of_sag(mask_sag_karsi)
    
    
    
    #-------------------------------------------------------------------------------------------------
    ret, solserit_thresh = cv2.threshold(solserit, 0.01, 1, cv2.THRESH_BINARY) #En doğru değer 0.0001de verdi  nesne sayısı olarak da 0.01 doğru değeri veriyor
    solserit_thresh = solserit_thresh.astype(np.uint8)

    ret, sagserit_thresh = cv2.threshold(sagserit, 0.01, 1, cv2.THRESH_BINARY) #En doğru değer 0.0001de verdi  nesne sayısı olarak da 0.01 doğru değeri veriyor
    sagserit_thresh = sagserit_thresh.astype(np.uint8)


    #-----------------------------------------------------------------------------------------------
    
    interested_value = 1  

    #Sol pikseller
    interested_pixels = np.where(solserit_thresh == interested_value)
    pixel_sol = len(interested_pixels[0])
    print(f"sol piksel sayısı: {pixel_sol}")

    #Sağ piseller
    interested_pixels = np.where(sagserit_thresh == interested_value)
    pixel_sag = len(interested_pixels[0])
    print(f"sağ piksel sayısı: {pixel_sag}")

    if pixel_sol>pixel_sag:
        print("Sol şerittesin")

    elif pixel_sag>pixel_sol:
        print("Sağ şerittesin")
    

    #------------------------------------------------------------------------------------------------------

    M = cv2.moments(solserit)              
    N = cv2.moments(sagserit)             

    
    # lane detection imshow
    pred *=255                                                
    pred = pred.astype(np.uint8)           
    red=np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.uint8)
    cv2.rectangle(red,(0,0),(red.shape[1],red.shape[0]),(0,0,164),-1)      
    maskbgr= cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)                        
    redmask=cv2.bitwise_and(maskbgr,red)                                  

    # serit silme
    bitnot=cv2.bitwise_not(pred)                    
    bitnot3= cv2.cvtColor(bitnot, cv2.COLOR_GRAY2BGR)
    bitw=cv2.bitwise_and(image,bitnot3)
    image=cv2.add(bitw,redmask)                      
     
    if pixel_sol>800 and pixel_sag>800:
        cx1 = int(M['m10']/M['m00'])                 
        cy1 = int(M['m01']/M['m00'])                 
        cv2.circle(image,(cx1,cy1),3,(255,255,255),-1) 
        cv2.putText(image,"Sol M",(cx1,cy1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1,cv2.LINE_8)     
    
        cx2 = int(N['m10']/N['m00'])                                      
        cy2 = int(N['m01']/N['m00'])                                       
        cv2.circle(image,(cx2,cy2),3,(255,255,255),-1)                       
        cv2.putText(image,"Sag M",(cx2,cy2),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1,cv2.LINE_8)  

    
        ax1= (cx1+cx2)//2                  
        ay1= (cy1+cy2)//2                              
        

    elif pixel_sol>800 :
        cx1 = int(M['m10']/M['m00'])                 
        cy1 = int(M['m01']/M['m00'])                 
        cv2.circle(image,(cx1,cy1),3,(255,255,255),-1) 
        cv2.putText(image,"Sol M",(cx1,cy1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1,cv2.LINE_8) 

        ax1= cx1+200
        ay1= cy1
        

    elif pixel_sag>800:
        cx2 = int(N['m10']/N['m00'])                                      
        cy2 = int(N['m01']/N['m00'])                                       
        cv2.circle(image,(cx2,cy2),3,(255,255,255),-1)                       
        cv2.putText(image,"Sag M",(cx2,cy2),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1,cv2.LINE_8)  

        ax1= cx2-200
        ay1= cy2

    else:
        print("şerit yok")                                    

    #-------------------------------------------------------------------------------------------------
        
    try:
    
        A = np.array([256, 512]) 
        B = np.array([256, 420])  
        C = np.array([ax1, ay1])  

        angle = calculate_angle(A, B, C)
        cv2.line(image, (ax1, 512), (ax1,ay1+100), (0, 255, 255), 3)   
        cv2.line(image, (256, 512), (ax1,ay1+100), (0, 255,0), 3)  
        
        
        if ax1<256:
            angle=-(angle//1)
            cv2.putText(image, f"{angle:.2f} derece", (ax1, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            print(f"Açı: {angle} derece")
            
        
        else:
            
            angle= (angle//1)
            cv2.putText(image, f"{angle:.2f} derece", (ax1, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            print(f"Açı: {angle} derece")
            

        pid = PID(1.8, 1.5 ,4.5, setpoint=0)   #P=1.8      I=1.5        D=4.5    mükemmel simülasyon değerleri  
        pid.output_limits = (-15, 15) 
        
        output=pid(angle)

        if angle>8:
            print(output)
            output=output-6 
            print(f"Direksiyon Açısı: {output}")
            steering_angle = oranlayici(eski_aralik, yeni_aralik, output)
            
        
        elif angle<-8:
            print(output)
            output=output+6  
            print(f"Direksiyon Açısı: {output}")
            steering_angle = oranlayici(eski_aralik, yeni_aralik, output)

            
        else:
            output= output
            print(f"Direksiyon Açısı: {output}")
            steering_angle = oranlayici(eski_aralik, yeni_aralik, output)
            
            
    except:
        print("şerit tespit edilemedi")
    image=cv2.resize(image,(1500,700))
    bitir = time.perf_counter()
    zaman = bitir-basla
    cv2.imshow("Lane segmentation", image)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()
