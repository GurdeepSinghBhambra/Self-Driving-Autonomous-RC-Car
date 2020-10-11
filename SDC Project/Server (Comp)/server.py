import socket

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None
        self.client_addr = None

    def startServer(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.conn, self.client_addr = self.socket.accept()

    def send(self, msg):
        self.conn.send(msg)

    def recv(self, size):
        return self.conn.recv(size)

    def stopServer(self):
        self.socket.close()

def test():
    server = Server("127.0.0.1", 8000)
    server.startServer()
    print(server.recv(1024))
    server.send(bytes(input("MSG: "), 'utf-8'))
    server.stopServer()

#test()