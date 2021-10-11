# Importazione OpenCv
import cv2
# Viene richiamata la classe per la cattura del flusso dalla fotocamera
video=cv2.VideoCapture(0)
 # Viene richiamata la classe CascadeClassifier che prende come input il classificatore xml
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0

while True:
#ret è 1 se il frame viene catturato
#la variabile frame contiene la matrice dell'immagine catturata
	ret,frame=video.read()
#detectMultiScale accetta come parametri: l'immagine, il fattore di scala e il fattore minNeighbors.
#Se riesce ad individuare il volto, restituisce i parametri (x,y,w,h) all'interno di faces (Rect(x,y,w,h))
	faces=facedetect.detectMultiScale(frame,1.3, 5)
	for x,y,w,h in faces:
		count=count+1
		name='./images/1/'+ str(count) + '.jpg'
		print("Creating Images........." +name)
		#Scrive il file immagine solamente nella parte dove è presente il volto
		cv2.imwrite(name, frame[y:y+h,x:x+w])
		#Disegna il rettangolo dove è presente il volto
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	#Visualizza l'immagine
	cv2.imshow("WindowFrame", frame)
	cv2.waitKey(1)
	#Ottenute 500 catture interrompe l'esecuzione
	if count>500:
		break
#Rilascio delle
video.release()
cv2.destroyAllWindows()