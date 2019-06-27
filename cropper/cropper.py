import sys
import os
import magic
import cv2

def main():
	inFolder = sys.argv[1]
	outFolder = sys.argv[2]
	cascadeFile = "lbpcascade_animeface.xml"
	cascade = cv2.CascadeClassifier(cascadeFile)


	for file in os.listdir(inFolder):
		fileName = inFolder+"/"+file
		outName = outFolder+"/"+file
		extension = magic.from_file(fileName,mime=True)

		if (extension == "image/jpeg" or extension=="image/png"):
			image = cv2.imread(fileName,cv2.IMREAD_COLOR)
			grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			grayImg = cv2.equalizeHist(grayImg)
			faces = cascade.detectMultiScale(grayImg,
											# detector options
											scaleFactor = 1.1,
											minNeighbors = 5,
											minSize = (24, 24))

			i = 0
			for (x, y, w, h) in faces:
				new_x = x+int(w/2)-int(h/2)
				croppedImg = image[y:y+h, x:x+h]
				resizedImg = cv2.resize(croppedImg, (64,64))

				i=i+1
				cv2.imwrite(outName+"_"+str(i)+".png",resizedImg)	


if __name__ == '__main__':
	main()