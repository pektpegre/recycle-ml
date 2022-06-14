#!/usr/bin/python

import sys
import cv2
import time
import math
import smbus
#from gpiozero import Servo
#from gpiozero import DistanceSensor
#from time import sleep
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


#model = "efficientnet_lite0.tflite"
model = "model.tflite"
enable_edgetpu = False
num_threads = 1
max_results = 4
score_threshold = 0.2

width = 640
height = 480

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


# Initialize the image classification model
base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)

# Enable Coral by this setting
classification_options = processor.ClassificationOptions(max_results=max_results, score_threshold=score_threshold)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)

classifier = vision.ImageClassifier.create_from_options(options)
print(classifier)
# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()






# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================

class PCA9685:

  # Registers/etc.
  __SUBADR1            = 0x02
  __SUBADR2            = 0x03
  __SUBADR3            = 0x04
  __MODE1              = 0x00
  __PRESCALE           = 0xFE
  __LED0_ON_L          = 0x06
  __LED0_ON_H          = 0x07
  __LED0_OFF_L         = 0x08
  __LED0_OFF_H         = 0x09
  __ALLLED_ON_L        = 0xFA
  __ALLLED_ON_H        = 0xFB
  __ALLLED_OFF_L       = 0xFC
  __ALLLED_OFF_H       = 0xFD

  def __init__(self, address=0x40, debug=False):
    self.bus = smbus.SMBus(1)
    self.address = address
    self.debug = debug
    if (self.debug):
      print("Reseting PCA9685")
    self.write(self.__MODE1, 0x00)
	
  def write(self, reg, value):
    "Writes an 8-bit value to the specified register/address"
    self.bus.write_byte_data(self.address, reg, value)
    if (self.debug):
      print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
	  
  def read(self, reg):
    "Read an unsigned byte from the I2C device"
    result = self.bus.read_byte_data(self.address, reg)
    if (self.debug):
      print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
    return result
	
  def setPWMFreq(self, freq):
    "Sets the PWM frequency"
    prescaleval = 25000000.0    # 25MHz
    prescaleval /= 4096.0       # 12-bit
    prescaleval /= float(freq)
    prescaleval -= 1.0
    if (self.debug):
      print("Setting PWM frequency to %d Hz" % freq)
      print("Estimated pre-scale: %d" % prescaleval)
    prescale = math.floor(prescaleval + 0.5)
    if (self.debug):
      print("Final pre-scale: %d" % prescale)

    oldmode = self.read(self.__MODE1);
    newmode = (oldmode & 0x7F) | 0x10        # sleep
    self.write(self.__MODE1, newmode)        # go to sleep
    self.write(self.__PRESCALE, int(math.floor(prescale)))
    self.write(self.__MODE1, oldmode)
    time.sleep(0.005)
    self.write(self.__MODE1, oldmode | 0x80)

  def setPWM(self, channel, on, off):
    "Sets a single PWM channel"
    self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
    self.write(self.__LED0_ON_H+4*channel, on >> 8)
    self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
    self.write(self.__LED0_OFF_H+4*channel, off >> 8)
    if (self.debug):
      print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))
	  
  def setServoPulse(self, channel, pulse):
    "Sets the Servo Pulse,The PWM frequency must be 50HZ"
    pulse = pulse*4096/20000        #PWM frequency is 50HZ,the period is 20000us
    self.setPWM(channel, 0, int(pulse))

if __name__=='__main__':
 
  pwm = PCA9685(0x40, debug=False)
  pwm.setPWMFreq(50)

  basketPos = 1200
  basketCPos = 1200
  pwm.setServoPulse(0,1200)
  pwm.setServoPulse(1,800)

  garbageType = ['Aluminium', 'Plastic', 'Paper', 'Background']

# Continuously capture images from the camera and run inference
imN = 0
while (imN < 20):
  time.sleep(5)
  # Start capturing video input from the camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  time.sleep(0.5)

  if (cap.isOpened()):
    imN = imN + 1
    print('######### run ', imN)
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create TensorImage from the RGB image
    tensor_image = vision.TensorImage.create_from_array(rgb_image)
    # List classification results
    categories = classifier.classify(tensor_image)

    
    categories.classifications[0].classes.sort(key=lambda x: x.score, reverse=False)
    print(categories.classifications[0].classes)

    # Show classification results on the image
    for idx, category in enumerate(categories.classifications[0].classes):
      #print(category)
      class_name = category.index
      #print(class_name)
      score = round(category.score, 2)
      result_text = str(class_name) + ' (' + str(score) + ')'
      text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    cv2.imwrite('ELLAK-' + str(class_name) + '-'+ str(class_name) + '.jpg', image)
#    cv2.imwrite('bg-0-'+ str(imN) + '.jpg', image)
    print(class_name)
    print(score)

    #################### turn servo based on index
    basket = class_name
    basketPos = 200+500*(basket)
    if (basket != 0): # Background - no object
      if (basketPos > basketCPos):
        step = 20
      else:
        step = -20

      print('Garbage Type: ', garbageType[basket])
      print('Basket: ', basket)
      print('Basket Pos: ', basketPos)
      print('Step: ', basketPos)
      time.sleep(0.5)

      for i in range(basketCPos,basketPos,step):
        pwm.setServoPulse(0,i)
        time.sleep(0.05)
      basketCPos = basketPos


      for i in range(800,1000,20):
        pwm.setServoPulse(1,i)
        time.sleep(0.05)
      time.sleep(1)
      for i in range(1000,800,-20):
        pwm.setServoPulse(1,i)
        time.sleep(0.05)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(33) == ord('a'):
      print('#################### key pressed')
      break

  cap.release()

