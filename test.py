#Caricamento Librerie
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import tensorflow as tf
print("Versione tensorflow")
print(tf.__version__)
print("Versione h5py")
import h5py
print(h5py.__version__)
import keras
print("Versione keras")
print(keras.__version__)
from keras.models import load_model
# Qui viene importato il classificatore per il riconoscimento del volto
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Siglia di probabilità
threshold=0.90

cap=cv2.VideoCapture(0)
# Imposta la largezza del frame a 640
cap.set(3, 640)
# Imposta l'altezza del frame a 480px
cap.set(4, 480)
# Imposta il font
font=cv2.FONT_HERSHEY_COMPLEX
# Carica in "model" il modello allenato
model = load_model("MyTrainingModel.h5")
# Operazione di preprocessing
def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

# Corrispopndenza tra numero e etichetta
def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"


while True:
	#suces è 1 se il frame viene catturato
	#la variabile imgOriginal contiene la matrice dell'immagine catturata
	sucess, imgOrignal=cap.read()
	#detectMultiScale accetta come parametri: l'immagine, il fattore di scala e il fattore minNeighbors.
#	Se riesce ad individuare il volto, restituisce i parametri (x,y,w,h) all'interno di faces (Rect(x,y,w,h))
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		# cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
		# cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
		# Ritaglia l'immagine solo in corrispondenza del volto
		crop_img=imgOrignal[y:y+h,x:x+h]
		# Ridimensiona l'immagine poichè dovrà essere elbaorata dal modello
		img=cv2.resize(crop_img, (32,32))
		img=preprocessing(img)
		img=img.reshape(1, 32, 32, 1)
		# cv2.putText(imgOrignal, "Class" , (20,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
		# cv2.putText(imgOrignal, "Probability" , (20,75), font, 0.75, (255,0,255),2, cv2.LINE_AA)
		# Predizione sull'immagine. Restituisce la probabilità che l'immagine appartenga alle classi 0 e 1
		prediction=model.predict(img)
		#print(prediction)
		# Predizione classe
		classIndex=model.predict_classes(img)
		# Ritorna il valore massimo dell'array, poichè siamo interessati alla classe con probabilità più alta
		probabilityValue=np.amax(prediction)
		# Se il valore di probabilità è maggiore della soglia impostata allora visualizza il rettangolo di delimitazione e la classe di apparteneza
		if probabilityValue>threshold:
			if classIndex==0:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif classIndex==1:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

			# cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
# Rilascio delle risorse
cap.release()
cv2.destroyAllWindows()














