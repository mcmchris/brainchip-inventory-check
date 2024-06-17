import sys
import board
import busio
import time

i2c = busio.I2C(board.SCL, board.SDA)

scan_result = i2c.scan()

print("I2C devices found: ", [hex(i) for i in scan_result])

matrix = 0x65

if not matrix in scan_result:
    print("Could not find Seeed Matrix")
    sys.exit()

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
 
if __name__ == "__main__":
    VID = getDeviceVID()
    if VID != '0x86':
        print("Could not detect led matrix!!!")
        sys.exit()
    
    print("Matrix init success")

    stopDisplay()

    time.sleep(2)
    
    RGB = (0 << 16) | (255 << 8) | 0

    displayColorBlock(RGB,0,True)

    time.sleep(2)

    stopDisplay()

    time.sleep(2)

    picTwo = [255]*64
    
    for i in range(7):
        picTwo[xytoIndex(i,i)] = 55

    displayFrames(picTwo, 2000, True, 1)

    # source .venv/bin/activate  #enter virtual env