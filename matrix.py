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
    result = bytearray(1)
    i2c.readfrom_into(matrix, result)
    print("Seeed Matrix ID: ", hex(int.from_bytes(result,"big")))

if __name__ == "__main__":
    getDeviceVID()