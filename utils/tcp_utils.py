import socket


def initTCP(address):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(address)
  sock.listen()
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
    except BlockingIOError:
      pass


def sendData(data, recipients):
  for i, (conn, _) in reversed(list(enumerate(recipients))):
    try:
      conn.send(data)
    except (ConnectionAbortedError, ConnectionResetError):
      conn.shutdown(socket.SHUT_RDWR)  # will this error out?
      conn.close()
      del recipients[i]
    except BlockingIOError:
      pass


def doShutdown(sock, recipients):
  for conn, _ in recipients:
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()
  sock.shutdown(socket.SHUT_RDWR)
  sock.close()
