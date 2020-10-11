from multiprocessing import Pipe, Process
from server import Server
import numpy as np
import msvcrt
import time
import csv
import cv2
import os

class ManualHandler:

    car_controls = {b'w':b"FWD", b's':b"BKW", b'a':b"LF", b'd':b"RG", b' ':b"STP"}
    direction = {b'w':b'F', b's':b'B'}
    ndv = {False:b"NDV000", True:b"NDV001"}
    connection = {b'q': b"END", b'n': b"NDV"}

    def __init__(self, server_ip, server_port, speed):
        self.server = Server(server_ip, server_port)
        self.speed = bytes(speed, "utf-8")
        self.current_direction = b'w'
        self.one_way_comm = False

    def start(self):
        self.server.startServer()

    def stop(self):
        self.server.stopServer()

    def translate(self, key):
        if(key == b'w'):
            self.current_direction = b'w'
            return ManualHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b's'):
            self.current_direction = b's'
            return ManualHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b'a' or key == b'd'):
            direct = ManualHandler.direction[self.current_direction]
            return direct + ManualHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b' '):
            return ManualHandler.car_controls[key] + b"000" + b'\n'
        elif(key == b'n'):
            if(self.one_way_comm == False):
                self.one_way_comm = True
                return ManualHandler.ndv[self.one_way_comm] + b'\n'
            else:
                self.one_way_comm = False
                return ManualHandler.ndv[self.one_way_comm] + b'\n'
        elif(key == b'q'):
            return ManualHandler.connection[key] + b"000" + b'\n'

    def send(self, msg):
        print("Sent:", msg)
        if(msg[:3] == b"NDV" or msg[:3] == b"END"):
            self.server.send(msg)
            return None
        elif(self.one_way_comm == True):
            self.server.send(msg)
            return None
        else:
            self.server.send(msg)
            return self.server.recv(31)
            #return "Distance Vector"

    def keyReaderAndSender(self):
        previous_char = b"69"
        brake_delay = 0.070 # 70 ms 
        key_list = list(ManualHandler.car_controls.keys())+list(ManualHandler.connection.keys())
        while(True):
            char = msvcrt.getch()
            if(char == b'n' or char != previous_char):
                #send the new value to client 
                #append to pipe or queue for logging here
                if(char not in key_list):
                    continue
                recvd = self.send(self.translate(char))        
                if(recvd):
                    #push this on the queue or pipe for logging the recvd data 
                    print("recvd:", recvd)
                if(char == b'q'):
                    return 
                previous_char = char

    @staticmethod
    def getCameraView(cam_link):
        cap = cv2.VideoCapture(cam_link)
        i=0
        while(True):
            ret, frame = cap.read()
            if(i == 0):
                print("Camera Properties, shape={}, fps={}".format(frame.shape, cap.get(cv2.CAP_PROP_FPS)))
                i+=1
                continue
            if(not ret or cv2.waitKey(1) == ord('q')):
                break
            cv2.imshow('Car Camera View', frame)


    @staticmethod
    def saveFile(csv_filename, pipe):
        print("Saving CSV File Started")
        with open(csv_filename, 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(['key', 'decision', 'video_file'])
            while(True):
                recvd_row = pipe.recv()
                if(type(recvd_row) == type("EOF") and recvd_row[:3] == 'EOF'):
                    break
                csvwriter.writerow(recvd_row)
        print("Saving CSV File Ended")

    @staticmethod
    def showAndRecordCamera(cam_link, pipe):
        print("showAndRecordCamera Started")
        break_status, i, recording = False, 0, False
        cap, fourcc = cv2.VideoCapture(cam_link), cv2.VideoWriter_fourcc(*'XVID')
        frame_shape, frame_fps, frame_dtype = None, None, None
        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            if(i==0):
                frame_dtype = frame.dtype
                frame_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_shape = frame.shape
                print("Frame info setup done")
                print("Frame Shape: {}, Frame dtype: {}, Frame FPS: {}\n".format(frame_shape, frame_dtype, frame_fps))
                pipe.send("FISS") #Frame Info Setup Successful
                i+=1
                continue
            if(pipe.poll()):
                recvd = pipe.recv()
                if(recvd == 'CNV'): #Capture New Video 
                    filename = pipe.recv()
                    out = cv2.VideoWriter(filename, fourcc, frame_fps, (frame_shape[1], frame_shape[0]), True)
                    recording = True
                elif(recvd == 'ECV'): #End Capturing Video
                    out.release()
                    recording = False
                else:
                    if(recording == True):
                        out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    print("showAndRecordCamera ended")
                    return
            if(recording == True):
                out.write(frame)
                title = "Car Camera *Recording"
            else:
                title = "Car Camera"
            cv2.imshow(title, frame)
            cv2.waitKey(1)
           
    def spawnProcesses(self, cam_link, csv_filename):
        file, _file = Pipe()
        cam, _cam = Pipe()
        file_proc = Process(target=self.saveFile, args=(csv_filename, _file))
        camera_proc = Process(target=self.showAndRecordCamera, args=(cam_link, _cam))
        camera_proc.start()
        file_proc.start()
        if(cam.recv() == "FISS"):
            return cam, file
        else:
            print("ERROR: spawnProcesses: Wrong Code Received from showAndRecordCamera")
            file_proc.terminate()
            camera_proc.terminate()
            exit()

    def record(self, cam_link, img_dir, csv_filename):
        previous_char = b"69"
        key_list = list(ManualHandler.car_controls.keys())+list(ManualHandler.connection.keys())+[b'r']
        recording_status, vid_no, filename = False, 1, None
        cam, file = self.spawnProcesses(cam_link, csv_filename)
        print("Enter Here:\n")
        while(True):
            char = msvcrt.getch()
            if(((char == b'n' or char == b'r') or char != previous_char) and (char in key_list)):
                if(char == b'r'):
                    if(recording_status == True):
                        cam.send("ECV")
                        recording_status = False
                    previous_char = char
                    continue
                car_cmd = self.translate(char)
                recvd = self.send(car_cmd)
                if(char == b'q'):
                    if(recording_status == True):
                        cam.send("ECV")
                    cam.send("EOF")
                    file.send("EOF")
                    return
                if(recvd):
                    print("recvd:", recvd)
                if(char == b'n'):
                    previous_char = char
                    continue
                filename = img_dir+"\\vid{}.avi".format(vid_no)
                if(recording_status == True):
                    cam.send('ECV')
                    cam.send('CNV')
                    cam.send(filename)
                else:
                    cam.send("CNV")
                    cam.send(filename)
                    recording_status=True
                file.send([char, car_cmd, filename])
                vid_no+=1
                previous_char = char

    @staticmethod
    def makeDataDir(master_directory, scene_dir):
        scene_dir_path = master_directory+"\\"+scene_dir
        vid_dir_path = scene_dir_path+"\\"+"videos"
        if(os.path.isdir(master_directory) == False):
            os.mkdir(master_directory)
            os.mkdir(scene_dir_path)
            os.mkdir(vid_dir_path)
        elif(os.path.isdir(scene_dir_path) == False):
            os.mkdir(scene_dir_path)
            os.mkdir(vid_dir_path)
        elif(os.path.isdir(vid_dir_path) == False):
            os.mkdir(vid_dir_path)
        return vid_dir_path, scene_dir_path+"\\"+"file.csv"
                

def test():
    handle = ManualHandler("123", 312412, "100")
    while(True):
        char = msvcrt.getch() 
        msvcrt.ungetch(char)
        print(char, "=", handle.translate(char))
        if(char == b'u'):
            break

#test()

def test2():
    ManualHandler("123", 312412, "100").keyReaderAndSender()

#test2()

def test3(speed):
    cam_link = "http://{}:{}/stream.mjpg".format("192.168.43.6", "8001")
    handler = ManualHandler("", 8000, speed)
    cam_proc = Process(target=handler.getCameraView, args=(cam_link, ))
    handler.start()
    print("Client Connected")
    cam_proc.start()
    handler.keyReaderAndSender()
    handler.stop()
    cam_proc.terminate()
    print("Server Stopped")

#test3()

def main():
    os.chdir('C:\\Users\\KIIT\\Documents\\AnacondaProjects\\SDC laptop')
    print("Current Working Directory:", os.getcwd())
    cam_link = "http://{}:{}/stream.mjpg".format("192.168.43.6", "8001")
    handler = ManualHandler("", 8000, "062")
    vid_dir, file_path = handler.makeDataDir(master_directory="data_collage_1920", scene_dir="clg_scene_7")
    print("----------------------------------------------------------------------------------------------------------------")
    print("Storage Info:")
    print("Current Scene Dir:", vid_dir.split('\\')[-2])
    print("Video Directory Path:", vid_dir)
    print("File Path:", file_path)
    print("----------------------------------------------------------------------------------------------------------------")
    handler.start()
    print("Client Connected")
    handler.record(cam_link, vid_dir, file_path)
    handler.stop()
    print("Server Stopped")

if __name__ == "__main__":
    #test3('057')
    main()
    #while(True):
    #    speed = input("Enter speed (XXX) format: ")
    #    if(speed == 'q'):
    #        break
        #test3("050")

    #
