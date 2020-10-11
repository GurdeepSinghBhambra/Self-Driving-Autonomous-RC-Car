__author__="Gurdeep"

import socket

class Client:
    def __init__(self, server_ip, server_port):
        self.server_port = server_port
        self.server_ip = server_ip
        self.socket = None
    
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.socket.connect((self.server_ip, self.server_port))

    def send(self, msg):
        self.socket.send(msg)

    def recv(self, size):
        return self.socket.recv(size)

    def disconnect(self):
        self.socket.close()


def test():
    client = Client("127.0.0.1", 8000)
    client.connect()
    client.send(bytes(input("MSG: "), 'utf-8'))
    print(client.recv(1024))
    client.disconnect()

#test()
