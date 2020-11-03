import SLCam
import threading
from flask import Flask, Response, render_template, request
# from flask_socketio import SocketIO, emit

audio_settings = {
    'fs': 3e5,  # sample rate
    'fMin': 20000,
    'fMax': 115000,
    'nFreq': 5e3,  # number of frequencies to plot
    'fScale': 'log',  # frequency spacing, linear or log
    'window': .0032,  # length of window in seconds
    'overlap': .875,  # fractional overlap
    'correction': True,  # whether to correct for 1/f noise
    'readRate': 1,  # how frequently to read data from the Daq's off-board data buffer

    # notes on parameters:

    # window, overlap, fMin, fMax are taken from deepsqueak
    # fs needs to be at least twice the highest frequency of interest, ideally higher
    # window*fs should be a power of 2 (or at least even) for optimal computation of fft
    # the higher the readRate, the better the performance, but the latency of plotting increases

    # nFreq determines the number of frequencies that are plotted (by cubic interpolation), not calculated
    # the browser also performs some interpolation, so nFreq should be as low as possible to see features of interest



}
app = Flask(__name__)
# socketio = SocketIO(app, async_mode='threading', cors_allowed_origins=[])

ag = SLCam.AcquisitionGroup(frame_rate=30, audio_settings=audio_settings)

# this section is a placeholder
# we will want to use the GUI to manage these settings
# ag.start(
# filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov', 'C:\\Users\\SchwartzLab\\Desktop\\Testing.tdms'], isDisplayed=[True, True])

# run collection in the background -- this should ultimately be initiated by a gui button
ag.run()

emitter = threading.Thread(target=ag.nidaq.display)
emitter.start()


# sendSize = int(settings['nFreq'] / settings['window'] / settings['overlap'])


api_switch = {
    'start_acquisition': lambda: ag.start(
        filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov', 'C:\\Users\\SchwartzLab\\Desktop\\Testing.tdms'], isDisplayed=[True, True]
    ),
    'stop_acquisition': ag.stop,
}


@app.route('/api', methods=['POST'])
def apiRouter():
  if request.method == 'POST':
    api_switch.get(request.form['action'])()

# this is a placeholder, mimicking a post request using a get request


# @app.route('/api/stop')
# def stop_running():
#   ag.stop()
  # socketio.emit('stopped')


@app.route('/video/calibration')
def calibration_switch():
  cg.cameras[0].calibration_switch()


@app.route('/video/<int:cam_id>')
def generate_frame(cam_id):
  return Response(ag.cameras[cam_id].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audio')
def generate_spectrogram():
  return Response(ag.nidaq.display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=3001, debug=True, use_reloader=False)
