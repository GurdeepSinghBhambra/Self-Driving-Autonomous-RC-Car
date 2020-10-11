#####################################################################################
#    Computer        Arduino                                                        #
#    Connects   ->   Serial Connection Established                                  #
#    Receives   <-   Ready                                                          #
#    Decision   ->   Receives                                                       #
#    Receives   <-   Sensor Data                                                    #
#    OFF000\n   ->   End Transmission, Reset All motors                             #
#                                                                                   #
#   Decision: Every command is followed by motor speed                              #
#             Commands:                                                             #
#                       FWD[MOTORSPEED (000-100 range)]\n: Forward                  #
#                       BKW[MOTORSPEED (000-100 range)]\n: Backward                 #
#                       FLF[MOTORSPEED (000-100 range)]\n: Forward Left Turn        #
#                       FRG[MOTORSPEED (000-100 range)]\n: Forward Right Turn       #
#                       BLF[MOTORSPEED (000-100 range)]\n: Backward Left Turn       #
#                       BRG[MOTORSPEED (000-100 range)]\n: Backward Right Turn      #
#                       STP000\n: Stop the car                                      #
#                       OFF000\n: End Transmission, Switch off all motors           # 
#                       RDY(Arduino -> Computer): Arduino Ready                     # 
#                       END(Arduino -> Computer): Arduino Serial Connection off     #
#                       ERR(Arduino -> Computer): Error in Arduino                  #
#                       NDV001\n: No Distance Vectors, Disables Incoming Messages   #
#                       NDV000\n: Enables Distance Vectors                          #
#                                                                                   #
#   NOTE: If the Distance vectors are disabled, then the incoming messages are also #
#         disabled. So Err, messages are also disabled.                             #
#####################################################################################


import serial
import serial.tools.list_ports as lp

#sample_commands = [b'FWD090\n', b"BKW090\n", b"FLF075\n", b"BRG075\n", b"STP000\n"]

class ArduinoSerial:
    def __init__(self, baudrate):
        self.port = self.findArduinoPort()
        self.baudrate = baudrate
        self.one_way_communication = False

    def establishConnection(self):
        try:
            self.serial_com = serial.Serial(self.port, self.baudrate)
        except serial.SerialException as error:
            print("ERROR:", error)
            exit()
        #print("Waiting for response ...")
        if(self.serial_com.readline() == b"RDY"):
            return True
        else:
            return False

    def send(self, decision):
        self.serial_com.write(decision)
        self.serial_com.flush()
        if(self.one_way_communication==False):
            return self.serial_com.readline()
        else:
            return None

    def setOneWayComm(self, bool_val):
        if(bool_val == False):
            self.serial_com.write(b"NDV000\n")
            self.serial_com.flush()
        else:
            self.serial_com.write(b"NDV001\n")
            self.serial_com.flush()
        self.one_way_communication = bool_val

    def endConnection(self):
        self.serial_com.write(b"OFF000")
        if(self.serial_com.readline() == b"END"):
            self.serial_com.close()
            return True
        return False

    def findArduinoPort(self):
        found = False
        i=0
        arduino_port = None
        try:
            for port in lp.comports():
                if("Arduino" in port.description):
                    arduino_port = port.device
                    found=True
                    i+=1

            if(found == False):
                raise Exception("No Arduino Detected")
            elif(i>1):
                raise Exception("Multiple Arduinos Detected")
        except Exception as error:
            print("ERROR:", error)
        return arduino_port


#print("Arduino Port:", ArduinoSerial('com3', 38400).findArduinoPort())

#def Test():
#    from random import choice
    #import time
#    ard = ArduinoSerial('COM3', 38400)
#    ard.establishConnection()
#    ard.setOneWayComm(True)
#    while(1):
#        try:
#            print(ard.send(choice(sample_commands)))
#        except KeyboardInterrupt:
#            print("Ending Connection")    
#            print(ard.endConnection())
#            print("Connection closed")
#            break
        #time.sleep(0.5)



