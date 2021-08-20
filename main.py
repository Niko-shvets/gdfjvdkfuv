import socket
import time
import pickle   
import cv2
import sys
import numpy as np

import coordinates_extractor 
import math_transformation
import music

def main(file_name, recompute_flag=True):
    """
    В качестве аргумента идет имя картинки/ видео
    """

    # настройка сервера
    address = ('127.0.0.1', 8052) 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    # считка аргумента
    input_file = file_name[1]

    vidcap = cv2.VideoCapture(input_file)# считка файла
    success,image = vidcap.read()
    img_shape = np.array(image.shape[:2])# размер картинки
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))# определение фпс
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # количество фреймов

    if frame_count > 1:
        music_name = music.music_extractor(input_file) # достаем музыку

    
    if recompute_flag:
        
        coordinates3D =[]

        count = 0
        while success:
            
            coords3d,coords2d = coordinates_extractor.coords_create(image) # выделение 2д и 3д координат тела

           
            # show video
            cv2.imshow('Frame',image)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            success,image = vidcap.read() # итерирование к следующему фрейму видео
            count += 1
        
            coords3d = math_transformation.knee_check(coords3d) # проверка адекватнотного положения коленей
            coords3d = math_transformation.normalize_body(coords3d)
            coords3d['ass'] = math_transformation.get_ass_location(coords3d, coords2d, img_shape)
            coordinates3D.append(coords3d)
            

        coordinates3D = math_transformation.Kalman_preprocessing(coordinates3D) # сглаживание координат с помощью фильтра Калмана

        with open('outfile', 'wb') as fp:
            pickle.dump(coordinates3D, fp)

    else:
        with open ('outfile', 'rb') as fp:
            coordinates3D = pickle.load(fp)
    # return coordinates3D


    # отправка координат на юнити
    for coords3d in coordinates3D:
        coords_arr = np.array(list(coords3d.values()))
        coords_list = [round(num, 4) for num in coords_arr.reshape(-1)]
        msg = "".join(str(x) + " " for x in coords_list) # create massege 
        s.send(bytes(msg, "utf-8"))
        time.sleep(1/fps)

    s.close()

    video_name = 'name_example' # нужно чтобы виидео сохранялось под каким то именем
    music.add_music(video_name,music_name) # сохранение нового видео с музыкой
    
if __name__ == '__main__':
    recompute_flag = True # set False to use precomputed outfile with coordinates
    main(sys.argv, recompute_flag)
