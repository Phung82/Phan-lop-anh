import os  
import shutil

def rename_f(echo):
    #path="./__pycache__/"+str(echo)+".pth"
    shutil.copyfile("C:/Users/SB/Desktop/Do-An-MNR/bin/pycache/MH/cifar_e"+str(echo)+".pth", "C:/Users/SB/Desktop/Do-An-MNR/cifar_e"+str(echo)+".pth")
 
#echo=3
#rename_f(echo)
import time
def countdown(echo):
    if echo <=8:
        t=int(echo)*50
    if echo >8:
        t=int(echo)*120
    #print(type(t))
    #print('This window will remain open for 3 more seconds...')
    while t >= 0:
        #print(t, end='...')
        time.sleep(1)
        t -= 1
    return 0


    
