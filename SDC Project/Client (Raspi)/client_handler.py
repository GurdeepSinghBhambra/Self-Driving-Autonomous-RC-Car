from client import Client
from arduino import ArduinoSerial


class RaspiHandler:
    def __init__(self, server_ip, server_port, baudrate):
        self.client = Client(server_ip, server_port) #args type: all strings
        self.arduino = ArduinoSerial(baudrate) #args type: string, int
        self.one_way_comm = False

    @staticmethod
    def getZeros(no):
        return "".join(['0' for i in range(no)])

    def formatVector(self, dist_vector):
        dist_vector = str(dist_vector[:-2], 'utf-8')
        format_vector = list()
        #format list format = [0000.00, 0000.00, 0000.00, 0000.00]
        for no in dist_vector.split(','):
            no_split_lst = no.split('.')
            normalised_no = [self.getZeros(4-len(no_split_lst[0]))+no_split_lst[0], self.getZeros(2-len(no_split_lst[1]))+no_split_lst[1]]
            new_no = ".".join(normalised_no)
            format_vector.append(new_no)
        return bytes(",".join(format_vector), 'utf-8')

    def start(self):
        self.arduino.establishConnection()
        self.client.connect()

    def stop(self):
        self.client.disconnect()
        self.arduino.endConnection()

    def pipeline(self):
        while(True):
            recvd_cmd = self.client.recv(7)
            print("Server =", recvd_cmd)
            print("______________________________")
            if(recvd_cmd[:3] == b"NDV" or recvd_cmd[:3] == b"END"):
                if(recvd_cmd == b"NDV000\n"):
                    self.one_way_comm = False
                    print("NDV Disabled")
                    self.arduino.setOneWayComm(False)
                elif(recvd_cmd == b"NDV001\n"):
                    self.one_way_comm = True
                    print("NDV Enabled")
                    self.arduino.setOneWayComm(True)
                else:
                    self.one_way_comm = False
                    print("END Command recvd")
                    return
            elif(self.one_way_comm == True):
                self.arduino.send(recvd_cmd)
                print("Through NDV enabled:", recvd_cmd)
            else:
                dv = self.formatVector(self.arduino.send(recvd_cmd))
                #dv = self.formatVector(b"3.4,323.32,2560.99,0.00\r\n")
                print("Through NDV disabled:", recvd_cmd)
                self.client.send(dv)
            print("______________________________\n\n")

#print(RaspiHandler("dasd", "dasdas", "asdas", 8907098).formatVector(b'12.6,2.0,00.0,14.4\r\n'))

def test():
    handler = RaspiHandler("127.0.0.1", 8000, 38400)
    handler.start()
    handler.pipeline()
    handler.stop()

test()
