import SLCam
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

ag = SLCam.AcquisitionGroup(frame_rate=30, audio_rate = int(3e5))

# this section is a placeholder
# we will want to use the GUI to manage these settings
ag.start(
    filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov', 'C:\\Users\\SchwartzLab\\Desktop\\Testing.tdms'], isDisplayed=[True, True])

# run collection in the background -- this should ultimately be initiated by a gui button
ag.run()

emitter = threading.Thread(target=ag.nidaq.display, args=(socketio))
emitter.start()

# api_switch = {
#     'start_camera_group': ag.start,
# }

# @app.route('/api', methods=['POST'])
# def apiRouter():
#   if request.method == 'POST':
#     api_switch.get(request.form['action'])()

# this is a placeholder, mimicking a post request using a get request


@socketio.on('connect')
def connected():
  emit('settings', {'center': 0, 'fs': ag.nidaq.sample_rate})
  # don't actually know what center should be...


@app.route('/api/stop')
def stop_running():
  ag.stop()
  socketio.emit('stopped')


@app.route('/video/calibration')
def calibration_switch():
  cg.cameras[0].calibration_switch()


@app.route('/video/<int:cam_id>')
def generate_frame(cam_id):
  return Response(ag.cameras[cam_id].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '__main__':
  socketio.run(app, host='127.0.0.1', port=3001,
               debug=True, use_reloader=False)
