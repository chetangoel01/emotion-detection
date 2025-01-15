from PIL import Image
import cv2 as cv
import numpy
import os
import pywhatkit
from tensorflow import keras
import matplotlib as plt

def casc():
	face_cascade = cv.CascadeClassifier("C:\\Users\\sanje\\Desktop\\emotion\\haarcascade_frontalface_default.xml")
	cap = cv.VideoCapture(0)
	while 1:
		ret, img = cap.read()
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			im = [x, y, w, h]
		cv.imshow('img', img)

		if cv.waitKey(30) & 0xFF == 27:
			break
		elif cv.waitKey(30) == 13:
			x, y, w, h = im
			cropped = img[y:y + h, x:x + w]
			cv.imwrite('C:\\Users\\sanje\\Desktop\\emotion\\input_haar_cropped.jpeg', cropped)
			break
	
	cap.release()
	cv.destroyAllWindows()

def preprocess(img):
	image = Image.open(img)
	image = image.resize((48, 48))
	image = image.convert('L')
	image.save('C:\\Users\\sanje\\Desktop\\emotion\\face_grayscale.jpeg')
	return numpy.array(image)

def youplay(emotion):
	emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
	if emotion == "angry":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=NvPZ-wvUDdM")
	elif emotion == "neutral":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=Q6MemVxEquE&t=8385s")
	elif emotion == "fear":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=MvHVzOREGUQ")
	elif emotion == "happy":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=xiUhNx24iZs")
	elif emotion == "sad":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=NkstXAUSpyM")
	elif emotion == "surprise":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
	elif emotion == "disgust":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=VDa5iGiPgGs")
	return 0

def emotionprediction(face):
	model = keras.models.load_model("my_model.h5")
	data = model.predict(face)
	fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
	emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    axs[0].imshow(face, 'gray')
    
    axs[1].bar(bar_label, data, color='orange', alpha=0.7)
    axs[1].grid()
	plt.show()

	# load model here
	# return emotion predicted by our model
	return "happy"

def main():
	casc()
	img = "C:\\Users\\sanje\\Desktop\\emotion\\input_haar_cropped.jpeg"
	face = preprocess(img)
	os.remove(img)

	emotion = emotionprediction(face)
	youplay(emotion)
	print("pe")
	
main()
