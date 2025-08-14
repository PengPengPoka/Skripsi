import cv2 as cv
import os
from matplotlib import pyplot as plt

def main():
    directory = os.path.expanduser('~')
    tea_class = ("DUST", "DUST2", "DUST3", "BOHEA", "BOP", "BOPF", "BOPF1", "FI", "F2", "PF", "PF2", "PF3", "BP")
    
    i = 1
    j = 1
    
    tea_file = directory + "\\Repositories\\Skripsi\\Data_NEW\\" + tea_class[0] + "\\DUST_001.jpg"
    print(tea_file)
    
    image = cv.imread(tea_file)
    
    if image is None:
        print("No image")
    
    cv.imshow("DUST", image)
    key = cv.waitKey()
    
    if key == ord('q'):
        exit()
    
if __name__ == "__main__":
    main()