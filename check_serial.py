import serial
import time

ser = serial.Serial('/dev/ttyACM1', 115200, timeout=.1)

def write_read(x): 
    x = str(x)
    ser.write(bytes(x, 'utf-8')) 
    time.sleep(0.05) 
    data = ser.readline() 
    return data 

while True: 
    num = 180 # Taking input from user 
    value = write_read(num) 
    print(value) # printing the value 

