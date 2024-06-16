import sys
import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)

print("I2C devices found: ", [hex(i) for i in i2c.scan()])

matrix = 0x65

if not matrix in i2c.scan():
    print("Could not find Seeed Matrix")
    sys.exit()

def getDeviceVID():
    i2c.writeto(matrix, bytes([0x00]))
    result = bytearray(2)
    i2c.readfrom_into(matrix, result)
    vid = hex(int.from_bytes(result,"big"))
    print("Seeed Matrix ID: ", vid)
    return vid

if __name__ == "__main__":
    VID = getDeviceVID()
    if VID != 0x8628:
        print("Could not detect led matrix!!!")
        sys.exit()
    
    print("Matrix init success")