import akida
import cv2
import math
import time
import signal
import threading
import sys
import numpy as np

# For I2C communication
import board
import busio
import time
import os

from queue import Queue
from scipy.special import softmax
from flask import Flask, render_template, Response
from picamera2 import MappedArray, Picamera2, Preview

# Preview Resolution
normalSize = (640 , 640)
#normalSize = (1920 , 1080)
# Model image size requeriment
lowresSize = (224, 224)
dotsResSize = (224, 224)

app = Flask(__name__, static_folder='templates/assets')

i2c = busio.I2C(board.SCL, board.SDA, 400000)
        
EI_CLASSIFIER_INPUT_WIDTH  = 224
EI_CLASSIFIER_INPUT_HEIGHT = 224
EI_CLASSIFIER_LABEL_COUNT = 1
EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD = 0.90
categories = ['piece']
inference_speed = 0
power_consumption = 0
piece_count = 0
akida_fps = 0


def ei_cube_check_overlap(c, x, y, width, height, confidence):
    is_overlapping = not ((c['x'] + c['width'] < x) or (c['y'] + c['height'] < y) or (c['x'] > x + width) or (c['y'] > y + height))

    if not is_overlapping:
         return False

    if x < c['x']:
        c['x'] = x
        c['width'] += c['x'] - x

    if y < c['y']:
        c['y'] = y;
        c['height'] += c['y'] - y;

    if (x + width) > (c['x'] + c['width']):
        c['width'] += (x + width) - (c['x'] + c['width'])

    if (y + height) > (c['y'] + c['height']):
        c['height'] += (y + height) - (c['y'] + c['height'])

    if confidence > c['confidence']:
        c['confidence'] = confidence

    return True

def ei_handle_cube(cubes, x, y, vf, label, detection_threshold):
    if vf < detection_threshold:
        return

    has_overlapping = False
    width = 1
    height = 1

    for c in cubes:
        # not cube for same class? continue
        if c['label'] != label:
             continue

        if ei_cube_check_overlap(c, x, y, width, height, vf):
            has_overlapping = True
            break

    if not has_overlapping:
        cube = {}
        cube['x'] = x
        cube['y'] = y
        cube['width'] = 1
        cube['height'] = 1
        cube['confidence'] = vf
        cube['label'] = label
        cubes.append(cube)

def fill_result_struct_from_cubes(cubes, out_width_factor):
    result = {}
    bbs = [];
    results = [];
    added_boxes_count = 0;
  
    for sc in cubes:
        has_overlapping = False;
        for c in bbs:
            # not cube for same class? continue
            if c['label'] != sc['label']:
                continue

            if ei_cube_check_overlap(c, sc['x'], sc['y'], sc['width'], sc['height'], sc['confidence']):
                has_overlapping = True
                break

        if has_overlapping:
            continue

        bbs.append(sc)

        results.append({
            'label'  : sc['label'],
            'x'      : int(sc['x'] * out_width_factor),
            'y'      : int(sc['y'] * out_width_factor),
            'width'  : int(sc['width'] * out_width_factor),
            'height' : int(sc['height'] * out_width_factor),
            'value'  : sc['confidence']
        })

        added_boxes_count += 1
    result['bounding_boxes'] = results
    result['bounding_boxes_count'] = len(results)
    return result

def fill_result_struct_f32_fomo(data, out_width, out_height):
    cubes = []

    out_width_factor = EI_CLASSIFIER_INPUT_WIDTH / out_width;

    for y in range(out_width):
        for x in range(out_height):
            for ix in range(1, EI_CLASSIFIER_LABEL_COUNT + 1):
                vf = data[y][x][ix];
                ei_handle_cube(cubes, x, y, vf, categories[ix - 1], EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD);

    result = fill_result_struct_from_cubes(cubes, out_width_factor)

    return result


def inferencing(model_file, queueOut):
    akida_model = akida.Model(model_file)
    devices = akida.devices()
    print(f'Available devices: {[dev.desc for dev in devices]}')
    device = devices[0]
    device.soc.power_measurement_enabled = True
    akida_model.map(device)
    akida_model.summary()
    i_h, i_w, i_c = akida_model.input_shape
    o_h, o_w, o_c = akida_model.output_shape
    scale_x = int(i_w/o_w)
    scale_y = int(i_h/o_h)
    scale_out_x = dotsResSize[0]/EI_CLASSIFIER_INPUT_WIDTH
    scale_out_y = dotsResSize[1]/EI_CLASSIFIER_INPUT_HEIGHT

    global inference_speed
    global power_consumption
    global piece_count
    global akida_fps


    picam2 = Picamera2()
    #picam2.start_preview(Preview.DRM, x=0, y=0, width=1920, height=1080)
    picam2.start_preview(Preview.NULL)
    for i in picam2.sensor_modes:
        print("Sensor Mode: ", i)

    mode = picam2.sensor_modes[3]
    config = picam2.create_preview_configuration(sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']})
    
    picam2.configure(config)
    #print(picam2.video_configuration)
    #stride = picam2.stream_configuration("lores")["stride"]
    #stride = picam2.stream_configuration("main")["size"]

    #picam2.post_callback = DrawRectangles

    picam2.start()

        
    resize_dim = (EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)

    while True:
        #frame = picam2.capture_array("lores")
        frame = picam2.capture_array()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized_img = cv2.resize(img, resize_dim)
        
        input_data = np.expand_dims(resized_img, axis=0)
        
        start_time = time.perf_counter()
        logits = akida_model.predict(input_data)
        end_time = time.perf_counter()
        inference_speed = (end_time - start_time) * 1000

        pred = softmax(logits, axis=-1).squeeze()

        floor_power = device.soc.power_meter.floor
        power_events = device.soc.power_meter.events()
        
        active_power = 0
        for event in power_events:
            active_power += event.power
    
        power_consumption = f'{(active_power/len(power_events)) - floor_power : 0.2f}' 
        akida_fps = akida_model.statistics
        
        #print(akida_model.statistics)

        result = fill_result_struct_f32_fomo(pred, int(EI_CLASSIFIER_INPUT_WIDTH/8), int(EI_CLASSIFIER_INPUT_HEIGHT/8))
        
        #print(result)
        picTwo = [255]*64
  
        for bb in result['bounding_boxes']:
            img = cv2.circle(img, (int((bb['x'] + int(bb['width']/2)) * scale_out_x), int((bb['y'] + int(bb['height']/2)) * scale_out_y)), 8, (57, 255, 20), 2)
            img = cv2.circle(img, (int((bb['x'] + int(bb['width']/2)) * scale_out_x), int((bb['y'] +  int(bb['height']/2)) * scale_out_y)), 4, (255, 165, 0), 2)

            x = bb['x']
            y = 224 - bb['y']

            x = int(x*8/224)
            y = int(y*8/224)

            picTwo[xytoIndex(x,y)] = 55
             
        
        displayFrames(picTwo, 10, True, 1)

        piece_count = result['bounding_boxes_count']
        #piece_count = len(result['bounding_boxes'])
        #print(piece_count)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if not queueOut.full():
            queueOut.put(img)

        
        
def gen_frames():
    #resize_stream = (640, 480)
    while True:
        if queueOut.empty():
            time.sleep(0.01)
            continue
        img = queueOut.get()
        #resized_img = cv2.resize(img, resize_stream)
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
def get_inference_speed():
    while True:
        yield f"data:{inference_speed:.2f}\n\n"
        time.sleep(0.1)

def get_power_consumption():
    while True:
        yield "data:" + str(power_consumption) + "\n\n"
        time.sleep(0.1)

def get_piece_count():
    while True:
        yield "data:" + str(piece_count) + "\n\n"
        time.sleep(0.1)

def get_fps():
    while True:
        yield "data:" + str(akida_fps) + "\n\n"
        time.sleep(0.1)

def getDeviceVID():
    i2c.writeto(matrix, bytes([0x00]))
    result = bytearray(1)
    i2c.readfrom_into(matrix, result)
    vid = hex(int.from_bytes(result))
    print("Seeed Matrix ID: ", vid)
    return vid

def stopDisplay():
    i2c.writeto(matrix, bytes([0x06]))


def displayColorBlock(rgb, duration_time, forever_flag):
    data=[0]*7
    data[0] = 0x0d #I2C_CMD_DISP_COLOR_BLOCK
    data[1] = (rgb >> 16) & 0xff
    data[2] = (rgb >> 8) & 0xff
    data[3] = rgb & 0xff
    data[4] = (duration_time & 0xff)
    data[5] = ((duration_time >> 8) & 0xff)
    data[6] = forever_flag
    result = [hex(i) for i in data]
    i2c.writeto(matrix, bytes(data))
    #print(result)

def displayFrames(buffer, duration_time, forever_flag, frames_number):
    data=[0]*72
    if frames_number > 5:
        frames_number = 5
    elif frames_number == 0:
        return
    
    data[0] = 0x05 #I2C_CMD_DISP_CUSTOM
    data[1] = 0x0
    data[2] = 0x0
    data[3] = 0x0
    data[4] = frames_number

    for i in range(frames_number):
        data[5] = i
        for j in range(64):
            data[8+j] = buffer[j+i*64]
        if i == 0:
            data[1] = (duration_time & 0xff)
            data[2] = ((duration_time >> 8) & 0xff)
            data[3] = forever_flag

        result = [hex(i) for i in data]
        #print(result)
        i2c.writeto(matrix, bytes(data))


def xytoIndex(x, y):
    if x > 8 or x < 0 or y > 8 or y < 0:
        print("Axis out of range")
        return
    row = (8-y)*8 - 8
    index = row + x
    return index

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model_inference_speed')
def model_inference_speed():
	return Response(get_inference_speed(), mimetype= 'text/event-stream')

@app.route('/model_power_consumption')
def model_power_consumption():
	return Response(get_power_consumption(), mimetype= 'text/event-stream')

@app.route('/model_piece_count')
def model_piece_count():
	return Response(get_piece_count(), mimetype= 'text/event-stream')

@app.route('/model_fps')
def model_fps():
	return Response(get_fps(), mimetype= 'text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':

    model_file = './model/akida_model.fbz'
    
    os.system("i2cdetect -y 1") # this command scan and wakes up the I2C peripheral

    scan_result = i2c.scan()

    print("I2C devices found: ", [hex(i) for i in scan_result])
    time.sleep(1)
    matrix = 0x65

    if not matrix in scan_result:
        print("Could not find Seeed Matrix")
        sys.exit()

    stopDisplay()
    time.sleep(1)

    queueOut = Queue(maxsize = 24)

    t2 = threading.Thread(target=inferencing, args=(model_file,queueOut))
    t2.start()
    app.run(host = '0.0.0.0', port = 8080)
    t2.join()
