import cv2 as cv
import pandas as pd
import time

def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("camera device not detected")
        exit()

    start_readParameter = time.time()
    params = {}
    with open('Camera_Parameters.txt', 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=')
                params[key] = float(value) if '.' in value or '-' in value else int(value)

    df = pd.DataFrame([params])
    row = df.iloc[0]

    # Camera Parameters
    cap.set(cv.CAP_PROP_SETTINGS, 1)
    # cap.set(cv.CAP_PROP_BRIGHTNESS, row['brightness'])
    # cap.set(cv.CAP_PROP_CONTRAST, row['contrast'])
    # cap.set(cv.CAP_PROP_SATURATION, row['saturation'])
    # cap.set(cv.CAP_PROP_SHARPNESS, row['sharpness'])
    # cap.set(cv.CAP_PROP_WHITE_BALANCE_BLUE_U, row['white_balance'])
    # cap.set(cv.CAP_PROP_GAIN, row['gain'])
    # cap.set(cv.CAP_PROP_ZOOM, row['zoom'])
    # cap.set(cv.CAP_PROP_FOCUS, row['focus'])
    # cap.set(cv.CAP_PROP_EXPOSURE, row['exposure'])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    end_readParameter = time.time()

    while cap.isOpened():
        _, img = cap.read()

        cv.imshow("video", img)
        key = cv.waitKey(30)

        if key == ord('q'):
            break
        elif key == ord('t'):
            name = input('image name: ')

            start_saveImage = time.time()

            cv.imwrite(name+'.jpg', img)
            print('image saved succesfully')

            end_saveImage = time.time()
            time_saveImage = end_saveImage - start_saveImage

    time_parameter = end_readParameter - start_readParameter
    print(f"Time to initialize parameter: {time_parameter:.2f} seconds")
    print(f"Time to save image: {time_saveImage:.2f} seconds")

main()