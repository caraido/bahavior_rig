import SLCam
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

cg = SLCam.CameraGroup()

# this section is a placeholder
# we will want to use the GUI to manage these settings
cg.start(
    filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov'], isDisplayed=[True])

# run collection in the background -- this should ultimately be initiated by a gui button
grabber = threading.Thread(target=cg.cameras[0].run)
grabber.start()  # will run until the stop() method is called

emitter = threading.Thread(target=cg.nidaq.display, args=(socketio))
emitter.start()

# api_switch = {
#     'start_camera_group': cg.start,
# }

# @app.route('/api', methods=['POST'])
# def apiRouter():
#   if request.method == 'POST':
#     api_switch.get(request.form['action'])()

# this is a placeholder, mimicking a post request using a get request


@socketio.on('connect')
def connected():
  emit('settings', {'center': 0, 'fs': cg.nidaq.sample_rate})
  # don't actually know what center should be...


@app.route('/api/stop')
def stop_running():
  cg.cameras[0].stop()
  # should close sockets?


@app.route('/video/<int:cam_id>')
def generate_frame(cam_id):
  return Response(cg.cameras[cam_id].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '__main__':
  socketio.run(app, host='127.0.0.1', port=3001,
               debug=True, use_reloader=False)
