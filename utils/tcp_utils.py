import socket


def initTCP(address):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(address)
  sock.listen()
  sock.setblocking(False)
  return sock


def getConnections(sock, recipients, block=True):
  if block:
    sock.setblocking(True)
    conn, addr = sock.accept()
    recipients.append((conn, addr))
    sock.setblocking(False)
  else:  # assumes we're not blocking...
    try:
      conn, addr = sock.accept()
      recipients.append((conn, addr))
      print(f'Added recipient {addr} to socket serving on {sock.getsockname}')
    except BlockingIOError:
      pass

# NOTE: I think that conn.shutdown is proper form, but keep getting errors
# this might be related to socket.SO_REUSEADDR?


def sendData(data, recipients):
  for i, (conn, addr) in reversed(list(enumerate(recipients))):
    try:
      conn.send(data)
    except (ConnectionAbortedError, ConnectionResetError):
      # conn.shutdown(socket.SHUT_RDWR)
      print(
          f'Recipient {addr} disconnected from socket serving on {conn.getsockname()}')
      conn.close()  # will this error out?
      del recipients[i]
    except BlockingIOError:
      print('BlockingIOError while trying to send data.')
      # pass


def doShutdown(sock, recipients):
  for conn, _ in recipients:
    # conn.shutdown(socket.SHUT_RDWR)
    conn.close()
  # sock.shutdown(socket.SHUT_RD)  # this is just a listener socket, no writes
  sock.close()
