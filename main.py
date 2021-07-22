import socket
import time
from coordinates_extractor import *
from math_transformation import *
from funcs import *
import sys

def main(argv):

    """
    В качестве аргумента идет имя картинки
    """

    address = ('127.0.0.1', 8052)
# address = ("localhost", 8052)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    if argv[2] == 'photo':
        input_image = sys.argv[1]
        image = cv2.imread(input_image)
        # boxes, coord,shape = solver(image)
        # init_marker = initial_check(boxes, coord,shape)
        # assert init_marker == True, 'Initial check found error'

        coords3d,coords2d = coords_create(image)
        second_marker = check_sides(coords2d,50)

        assert second_marker == True, 'Second check found Error'


        coords3d = knee_check(coords3d)
        coords3d = normalize_body(coords3d,unity_body_length)
        
        # angle = human_angles(coords3d)

        coords_arr = np.array(list(coords3d.values()))
        coords_list = [round(num, 4) for num in coords_arr.reshape(-1)]
        msg = "".join(str(x) + " " for x in coords_list) # create massege 

        while True:
        # send message to a server
        # TODO: send message only once and make sure that it was received
            # msg = '%.4f %.4f %.4f %.4f'%(armX_1,armY_1,armX_2,armY_2)
            s.send(bytes(msg, "utf-8"))
            time.sleep(1)
        s.close()



    elif argv[2] == 'video':
        input_video = sys.argv[1]

        vidcap = cv2.VideoCapture(input_video)
        success,image = vidcap.read()

        # boxes, coord,shape = solver(image)
        # init_marker = initial_check(boxes, coord,shape)
        # assert init_marker == True, 'Initial check found error'

        coordinates3D =[]

        count = 0
        while success:
            

            coords3d,coords2d = coords_create(image)

            if count % 10 == 0:
                second_marker = check_sides(coords2d,50)
                assert second_marker == True, 'Second check found Error'

            # show video
            cv2.imshow('Frame',image)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # coordinates3D.append(сoords3d) # не знаю что лучше сначала в список потом передача, или предавать поочередно, пусть будет поочередно
            success,image = vidcap.read()
            count += 1
        
            coords3d = knee_check(coords3d)
            coords3d = normalize_body(coords3d,unity_body_length)
            # angle = human_angles(сoords3d)

            coords_arr = np.array(list(coords3d.values()))
            coords_list = [round(num, 4) for num in coords_arr.reshape(-1)]
            msg = "".join(str(x) + " " for x in coords_list) # create massege 
            s.send(bytes(msg, "utf-8"))

        s.close()

if __name__ == '__main__':
    main(sys.argv)
