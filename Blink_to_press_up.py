import dlib,cv2
import imutils as im
import pyautogui as kbd
from imutils import face_utils
from scipy.spatial import distance as d

def eye_aspect_ratio(eye):
	x = d.euclidean(eye[1], eye[5])
	y = d.euclidean(eye[2], eye[4])
	z = d.euclidean(eye[0], eye[3])
	ear = (x + y) / (2.0 * z)
	return ear

threshold = 0.2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(leftS, leftE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightS, rightE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cam = cv2.VideoCapture(0)
while True:
	_,frame = cam.read()
	frame = im.resize(frame, width=450)
	gscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gscale, 0)

	for face in faces:
		shape = predictor(gscale, face)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[leftS:leftE]
		rightEye = shape[rightS:rightE]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (255,0,255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (255,0,255), 1)
		if ear < threshold:
			cv2.putText(frame, "jump!!", (15, 80),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0, 255), 3)
			kbd.press('up')
	cv2.imshow("Blink to play!", frame)
	if cv2.waitKey(1) == 27:
		break
cam.release()
cv2.destroyAllWindows()