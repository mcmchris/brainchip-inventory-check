import sys
import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)

print("I2C devices found: ", [hex(i) for i in i2c.scan()])

Seeed_Matrix = 0x65

if not Seeed_Matrix in i2c.scan():
    print("Could not find Seeed Matrix")
    sys.exit()

def getDeviceVID():
    i2c.writeto(Seeed_Matrix, bytes([0x00]))
    result = bytearray(1)
    i2c.readfrom_into(Seeed_Matrix, result)
    print("Seeed Matrix ID: ", int.from_bytes(result,"big"))

if __name__ == "__main__":
    getDeviceVID()