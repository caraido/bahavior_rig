# from setup import setup
# from callbacks import initCallbacks
import sys
from socketApp import initServer


# from threading import Thread

if __name__ == "__main__":
  # Thread(target=socketio.run, args=(sockets,), kwargs={'port': 5001}).start()
  # server.run(host='localhost', port=5000, debug=False,
  #            use_reloader=False)
  if len(sys.argv) > 1:
    if sys.argv[1] == 'mock':
      from mockSetup import setup
      from mockCallbacks import initCallbacks
    else:
      raise Exception('Unclear run type')
  else:
    from setup import setup
    from callbacks import initCallbacks

  ag, status = setup()
  initCallbacks(ag, status)
  app, socket = initServer(ag, status)
  print('Serving on port 5001')
  socket.run(app, port=5001)
