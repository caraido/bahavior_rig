from flask import Flask
from flask_socketio import SocketIO, emit


def initServer(ag, status):
  app = Flask(__name__)
  socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

  def printToGUI(*args):
    socketio.emit('message', ' '.join([str(arg)
                                       for arg in args]), broadcast=True)
    print(*args)

  @socketio.on('connect')
  def handle_new_connection():
    emit('broadcast', status.update)
    print('new client registered')
    ag.print = printToGUI

  @socketio.on('disconnect')
  def handle_closed_connection():
    ag.print = print  # TODO: only if the number of connections is now zero!

  @socketio.on('get')
  def parse_request(request_type):
    print('Requested resource: ' + request_type)

    if request_type == 'allowed':
      print('sending allowed settings dictionary')
      return status.allowed

    elif request_type == 'current':
      return status.update

    elif request_type == 'processing': #give me a handful of rootfilename
      return { #might request files 0 to 30, may have only recorded 29
          'first': args[0], #if args[0] == 0, then I want the most recent rootfilename
          'sessions': [{
              'name': getrootfilename(i),  # get the actual rootfile name
              'status': getRandomStatus('testingA')} for i in range(args[0], min(args[0] + args[1], numberofrecordedsessions))]
        #a=rgs[1] = number of rootfilenames
      }  # suppose 110 is the number of actual sessions in existence

    elif request_type == 'processing categories':
      return {
          'headers': ['session', 'calibration', 'deepsqueak', 'deeplabcut', 'migration'],
          'info': [
              [{'name': 'Session name',
                'description': 'Set by the root file name during the recording', 'icon': None}],
              [
                  {'name': 'Configuration file',
                   'description': 'Copies the configuration file to the session directory', 'icon': 59181},
                  {'name': 'Undistortion',
                      'description': 'Uses the most recent calibration to correct for lens distortion', 'icon': 58868},
                  {'name': 'Coordinate extraction',
                      'description': 'Uses the arena-mounted markers to detect the location of the aerna', 'icon': 58950},
                  {'name': '3D alignment', 'description': 'Uses the extrinsic calibration data to extract the position of the cameras relative to the arena', 'icon': 59735}
              ],
              [{'name': 'DeepSqueak', 'description': 'Parses the microphone data for squeaks',
                'icon': 59288}],
              [
                  {'name': 'DeepLabCut', 'description': 'Extracts 2D pose estimates of mice in the arena',
                   'icon': 59813},
                  {'name': '3D pose estimation',
                      'description': 'Transforms the DeepLabCut results into 3D coordinates using the calibration data', 'icon': 60097}
              ],
              [
                  {'name': 'Upload', 'description': 'Uploads the data to the Schwartz Lab server',
                   'icon': 58981},
                  {'name': 'HDD', 'description': 'Transfers the data to long-term storage on this computer', 'icon': 59809}
              ]
          ]
      }



  @socketio.on('post')
  def parse_update(update):
    print('Requested change: ' + str(update))
    # the update is a dictionary of status:value pairs

    for k, v in update.items():
      status[k](v)

    # optional: send a string message to all clients that gets displayed on the gui
    emit('message', 'Requested change: ' + str(update), broadcast=True)

    # send the new status to all other clients as a 'broadcast' event
    emit('broadcast', status.update, broadcast=True, include_self=False)

    # return the new status to the requesting client
    print(f'returning status change: {status.update}')
    return status.update

  return app, socketio
