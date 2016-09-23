from PIL import Image
import sys

def imgRotate(imgFile, outputFile):
	img = Image.open(imgFile)
	img = img.rotate(180)
	img.save(outputFile)

if __name__ == "__main__":

	imgFile = sys.argv[1]
	outputFile = sys.argv[2]

	imgRotate(imgFile, outputFile)
