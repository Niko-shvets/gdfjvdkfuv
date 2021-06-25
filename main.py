import socket
import time
from coordinates_extractor import *
from math_transformation import *
import sys

def main(argv):

    """
    В качестве аргумента идет имя картинки
    """

    input_image = sys.argv[-1]
    # establish connection to a server
    address = ('127.0.0.1', 8052)
    # address = ("localhost", 8052)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    coordinates = coords_create(input_image,show = False,save = False)
    angles = human_angles(coordinates)# выход словарь!


    # !!!!!нужно сопоставить координаты или углы которые передаем мо мкриптом передачи!!!!


    # armX_1 = new_coord['left_wrist'][0]
    # armY_1 = new_coord['left_wrist'][1]
    # armX_2 = new_coord['right_wrist'][0]
    # armY_2 = new_coord['right_wrist'][0]

    while True:
        # send massage to a server
        msg = '%.4f %.4f %.4f %.4f'%(armX_1,armY_1,armX_2,armY_2)
        s.send(bytes(msg, "utf-8"))
        print(msg)
        # armX_1 += 1 ДОБАВИТЬ!!
        # armY_1 += 1
        # armX_2 -= 1
        # armY_2 -= 1


        time.sleep(1)

    s.close()

if __name__ == '__main__':
    main(sys.argv[-1])
