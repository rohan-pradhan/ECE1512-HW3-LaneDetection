import cv2 as cv
import os

for filename in os.listdir('./Night'):
    filename = './Night/' + filename
    print(filename)
    cap = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow(filename, cap)
    if cv.waitKey(0) & 0xFF== ord('q'):
        cap.release()
        cv.destroyAllWindows()
