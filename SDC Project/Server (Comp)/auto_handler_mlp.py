from sklearn.neural_network import MLPClassifier
from multiprocessing import Pipe, Process
from server import Server
from joblib import load
import numpy as np
import msvcrt
import time
import cv2
import csv
import os


class AutoHandler:

    car_controls = {b'w':b"FWD", b's':b"BKW", b'a':b"LF", b'd':b"RG", b' ':b"STP"}
    direction = {b'w':b'F', b's':b'B'}
    ndv = {False:b"NDV000", True:b"NDV001"}
    connection = {b'q': b"END", b'n': b"NDV"}

    def __init__(self, server_ip, server_port, speed, cam_link):
        self.server = Server(server_ip, server_port)
        self.cam_link = cam_link
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
            return AutoHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b's'):
            self.current_direction = b's'
            return AutoHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b'a' or key == b'd'):
            direct = AutoHandler.direction[self.current_direction]
            return direct + AutoHandler.car_controls[key] + self.speed + b'\n'
        elif(key == b' '):
            return AutoHandler.car_controls[key] + b"000" + b'\n'
        elif(key == b'n'):
            if(self.one_way_comm == False):
                self.one_way_comm = True
                return AutoHandler.ndv[self.one_way_comm] + b'\n'
            else:
                self.one_way_comm = False
                return AutoHandler.ndv[self.one_way_comm] + b'\n'
        elif(key == b'q'):
            return AutoHandler.connection[key] + b"000" + b'\n'

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

    def controller(self):
        cap = cv2.VideoCapture(self.cam_link)
        int2cmd_dict = load('int2cmd_dict.jobfile')
        print(int2cmd_dict)
        nn = load('nn.joblib')
        self.send(self.translate(b'n')) #NDV Enabled
        while(True):
            ret, frame = cap.read()
            if(not ret):
                print("ERROR: Camera Connection Lost")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = gray.flatten().astype(np.float32).reshape(1, -1)
            pred = nn.predict(gray)
            decision = int2cmd_dict[int(pred)]
            decision = bytes(decision[2:-3]+"\n", 'utf-8')
            print(decision)
            self.send(decision)
            print(decision)
            cv2.imshow('Car Camera View', frame)
            if(not ret or cv2.waitKey(1) == ord('q')):
                self.send(self.translate(b'q'))
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    cam_link = "http://{}:{}/stream.mjpg".format("192.168.43.6", "8001")
    ah = AutoHandler("", 8000, "065", cam_link)
    ah.start()
    print("Client Connected")
    ah.controller()
    ah.stop()
    print("Server Stopped")

main()    
