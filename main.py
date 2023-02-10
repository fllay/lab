import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        #print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            
            try:
              #print(member[0].text)
              value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
              xml_list.append(value)
            except IndexError:
              pass
    
            
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def runcmd(cmd):
  print("##################### run cmd ################")
  print(cmd)
  os.system(cmd)
  output = [""] #get_ipython().getoutput(cmd)
  print(output)

#we build config dynamically based on number of classes
#we build iteratively from base config files. This is the same file shape as cfg/yolo-obj.cfg
def file_len(fname):
  print(fname)
  with open(fname) as f:
    print(fname)
    i = 0
    for i, l in enumerate(f):
      pass
    return i + 1

import glob
import cv2


import os
import pickle
import shutil
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

dirs = ['train', 'val']
#classes = list(set(xml_df['class']))

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path, classes):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    img_file = dir_path + '/' + basename_no_ext + '.png'
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    #size = root.find('size')
    #w = int(size.find('width').text)
    #h = int(size.find('height').text)
    #print(img_file)
    im = cv2.imread(img_file)
    h, w, c = im.shape
    #print(width, height)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

cfg_data = '''
[net]
# Testing
#batch=1
#subdivisions=1
# Training
#batch=64
#subdivisions=24

batch=16
subdivisions=8
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = {max_batches}
policy=steps
steps={steps_str}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={num_filters}
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
nms_kind=greedynms
beta_nms=0.6

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={num_filters}
activation=linear

[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
nms_kind=greedynms
beta_nms=0.6
'''


import random
def stop():
  global log, stop_threads, is_done
  if stop_threads:
    log("Training is stopped!")
    is_done = True
    raise SystemExit() 

def training():
  global gl, run, log, stop, is_done
  xml_df=xml_to_csv('/content/images')
  print(list(set(xml_df['class']))) 
  runcmd('mkdir -p /content/darknet/data/obj')
  classes = list(set(xml_df['class']))

  full_dir_path = '/content/images'
  output_path = '/content/darknet/data/obj/'

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  image_paths = getImagesInDir(full_dir_path)
  list_file = open(full_dir_path + '.txt', 'w')
  print(list_file)

  for image_path in image_paths:
    list_file.write(image_path + '\n')
    convert_annotation(full_dir_path, output_path, image_path, classes)
    _, image_name = os.path.split(image_path)
    shutil.copyfile(image_path, os.path.join(output_path,image_name))

  list_file.close()



  print("Finished processing: ")

  image_paths = getImagesInDir(output_path)



  
  percent_val_size = 10
  number_of_samples = int(len(image_paths)*percent_val_size/100)
  print(number_of_samples)
  valid_set = random.choices(population=image_paths, k=number_of_samples)
  print(valid_set)
  train_set = [ele for ele in image_paths if ele not in valid_set]
  print(train_set)
  # 
  textfile1 = open("/content/darknet/data/valid.txt", "w")
  for element in valid_set:
    textfile1.write(element + "\n")
  textfile1.close()


  textfile2 = open("/content/darknet/data/train.txt", "w")
  for element in train_set:
    textfile2.write(element + "\n")
  textfile2.close()

  #
  classes_name = list(set(xml_df['class']))
  textfile = open("/content/darknet/backup/coco.names", "w")
  for element in classes_name:
    textfile.write(element + "\n")
  textfile.close()
  #n_class = 'classes = ' + str(len(classes_name) + 1)
  n_class = 'classes = ' + str(len(classes_name))
  obj_data = [n_class,
    'train = /content/darknet/data/train.txt',
    'valid = /content/darknet/data/valid.txt',
    'names = /content/darknet/data/obj.names',
    'backup = /content/darknet/backup']

  textfile3 = open("/content/darknet/data/obj.data", "w")
  for element in obj_data:
    textfile3.write(element + "\n")
  textfile3.close()

  textfile4 = open("/content/darknet/data/obj.names", "w")
  for element in classes_name:
    textfile4.write(element + "\n")
  textfile4.close()


  num_classes = file_len('/content/darknet/data/obj.names')
  max_batches = num_classes*2000
  steps1 = .8 * max_batches
  steps2 = .9 * max_batches
  #steps1 = .1 * max_batches
  #steps2 = .2 * max_batches
  #steps_str = str(steps1)+','+str(steps2)
  steps_str = "2000"
  num_filters = (num_classes + 5) * 3


  print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

  #Instructions from the darknet repo
  #change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
  #change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
  if os.path.exists('/content/darknet/cfg/custom-yolov4-tiny-detector.cfg'): os.remove('/content/darknet/cfg/custom-yolov4-tiny-detector.cfg')
  varc = cfg_data.format(num_filters = num_filters, max_batches = max_batches, steps_str=steps_str, num_classes = num_classes)
  #open text file
  cfg_file = open("/content/darknet/cfg/custom-yolov4-tiny-detector.cfg", "w")
 
  #write string to file
  n = cfg_file.write(varc)
 
  #close file
  cfg_file.close()
  runcmd('/content/darknet/darknet detector train /content/darknet/data/obj.data /content/darknet/cfg/custom-yolov4-tiny-detector.cfg /content/darknet/yolov4-tiny.conv.29 -dont_show -map')
  print("[Training Thread] Done! Ready for download!...")
  is_done = True
  stop_threads = True
  stop()
 
import tuna
import os
from flask import Flask, flash, request, redirect, url_for, send_file, jsonify
from flask_cors import CORS
import threading
import time


UPLOAD_FOLDER = '/content'
ALLOWED_EXTENSIONS = {'zip'}



gl = []
is_done = True

def run(cmd):
  global gl
  os.system(cmd)
  output = [""] #get_ipython().getoutput(cmd)
  print(output)
  gl = gl + output

def log(text):
  global gl
  print(text)
  gl = gl + [text]

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
# app.debug = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#run_with_ngrok(app)   #starts ngrok when the app is run



stop_threads = False

def upload():
    log("[Upload] Unzipping...")
    #run('rm -rf /root/tfdata/data/object-detection.pbtxt')
    run('rm /content/darknet/backup/*')
    run('rm -rf /content/images')
    run('mkdir /content/images')
    #run('mkdir /root/tfdata/images/test')
    run('unzip -j /content/package.zip -d /content/images')
    #run('ls -Q /root/tfdata/images/train | head -10 | xargs -i mv /root/tfdata/images/train/{} /root/tfdata/images/test/')
    xmlfiles = []
    pngfiles = []
    for root, dirs, files in os.walk("/content/images"):
      for filename in files:
        if filename.endswith(".xml"):
          xmlfiles.append(os.path.splitext(filename)[0])
        if filename.endswith(".png"):
          pngfiles.append(os.path.splitext(filename)[0])

    diff = set(pngfiles) - set(xmlfiles)

    print(xmlfiles)
    print(pngfiles)

    print(diff)
    for root, dirs, files in os.walk("/content/images"):
      for filename in files:
        for x in diff:
          if filename.startswith(x):
            print(os.path.join(root,filename))
            os.remove(os.path.join(root,filename))

    log("[Upload] Training Thread Starting...");
    x = threading.Thread(target=training)
    x.start()
    log("[Upload] Training Thread Started.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=['POST'])
def upload_file():
    global gl, run, log, stop_threads, is_done
    gl = []
    is_done = False

    log("[Upload] File uploading...")
    print("[Upload] File uploading...")
    stop_threads = False
    # check if the post request has the file part
    if 'file' not in request.files:
        log("[Upload] Error: No file content.")
        print("[Upload] Error: No file content.")
        return "No file content."
    print("Going!!!!!!!")
    file = request.files['file']
    print(file)
    print(file.filename)
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        log("[Upload] Error: No filename.")
        return "No filename."
    if file and allowed_file(file.filename):
        log("[Upload] Upload completed.")
        filename = os.path.join(app.config['UPLOAD_FOLDER'], "package.zip")
        log("[Upload] Saving... "+ filename)       
        file.save(filename)
        y = threading.Thread(target=upload)
        y.start()
        log("Done upload function... ") 
        response = jsonify({'res': "Upload completed,"+filename+". Now it is training..."})
        response.headers.add('Access-Control-Allow-Origin', '*')
        #return "Upload completed,"+filename+". Now it is training..."
        return response 

@app.route("/train/stop")
def train_stop():
  global stop_threads
  stop_threads=True
  response = jsonify({'res': "Stop command is executing..., Waiting for the running instruction is completed."})
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

@app.route("/train/log")
def train_log():
  response = jsonify({'res': "\n".join(map(lambda x: str(x), gl))})
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

@app.route("/train/is_running")
def train_is_running():
  global is_done
  if is_done == True:
    print("!!!!! Done training !!!!!!!")
    run('cp /content/darknet/backup/custom-yolov4-tiny-detector_best.weights /content/darknet/backup/custom-yolov4-tiny-detector.weights')
    run('tar -czvf /content/darknet/backup/model.tar.gz /content/darknet/backup/custom-yolov4-tiny-detector.weights /content/darknet/cfg/custom-yolov4-tiny-detector.cfg /content/darknet/backup/coco.names')
    run('cp /content/darknet/backup/model.tar.gz /content/drive/MyDrive')
  response = jsonify({'res': str(not is_done)})
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

@app.route("/download")
def download():
  print("####### Downloading #######")
  if not os.path.exists("/content/darknet/backup/model.tar.gz"):
    return "Frozen model is not yet available."
  return send_file("/content/darknet/backup/model.tar.gz", as_attachment=True)

@app.route("/")
def home():
    return "AI Training Server on Google Colab!"
  


if __name__ == '__main__':
    with open('/content/tt.txt') as f:
        lines = f.readlines()
    print(lines[0])
    tuna.run_tuna(5000,lines[0])
    app.run(host="0.0.0.0",debug=True)


