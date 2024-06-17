import sys
import board
import busio

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
    print(result)



if __name__ == "__main__":
    VID = getDeviceVID()
    if VID != '0x86':
        print("Could not detect led matrix!!!")
        sys.exit()
    
    print("Matrix init success")

    stopDisplay()
    RGB = (0 << 16) | (255 << 8) | 0
    displayColorBlock(RGB,0,True)
