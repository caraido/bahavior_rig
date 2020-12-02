import SLCam
import threading
from flask import Flask, Response, render_template, request, redirect, url_for
# from flask_socketio import SocketIO, emit
import cgi,cgitb
from utils import path_operation_utils as pop

audio_settings = {
    'fs': 3e5,  # sample rate
    'fMin': 200,
    'fMax': 80000,
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

#default filepath
#filepath = ['C:\\Users\\SchwartzLab\\Desktop\\Testing_Female2Record.mov',
#            None,
#            None,
#            'C:\\Users\\SchwartzLab\\Desktop\\Testing_Female2Record.tdms'
#]
model_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_first_try-Devon-2020-11-24\exported-models\DLC_Alec_first_try_resnet_50_iteration-0_shuffle-1'


# this section is a placeholder
# we will want to use the GUI to manage these settings
# ag.start(
# filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov', 'C:\\Users\\SchwartzLab\\Desktop\\Testing.tdms'], isDisplayed=[True, True])

# run collection in the background -- this should ultimately be initiated by a gui button
# ag.run()

# emitter = threading.Thread(target=ag.nidaq.display)
# emitter.start()
# sendSize = int(settings['nFreq'] / settings['window'] / settings['overlap'])

def record_switch():
    ag.cameras[0].saving_switch_on()
    ag.nidaq.saving_switch_on()

def dlc_switch():
    ag.cameras[0].dlc_switch(model_path=model_path)

def ex_calibration_switch():
    result = ag.cameras[0].extrinsic_calibration_switch()
    # TODO: change mimetype to display on webpage based on the returned value type
    if isinstance(result,str):
        data_type = 'text/html'
    else:
        data_type = 'multipart/x-mixed-replace; boundary=frame'
    return Response(result, mimetype=data_type)


def in_calibration_switch():
    result = ag.cameras[0].intrinsic_calibration_switch()
    # TODO: change mimetype to display on webpage based on the returned value type
    if isinstance(result, str):
        data_type = 'text/html'
    else:
        data_type = 'multipart/x-mixed-replace; boundary=frame'
    return Response(result, mimetype=data_type)

def get_filepath():
    path=request.values['folderpath']
    name=request.values['foldername']
    nCamera=ag.nCameras
    camera_list = []
    for i in range(nCamera):
        camera_list.append(ag.cameras[i].device_serial_number)
    path = pop.reformat_filepath(path,name,camera_list)
    ag.filepaths = path


api_switch = {
    'confirm and submit': get_filepath,
    'start': lambda: ag.start(isDisplayed=True),
    'trace': dlc_switch,
    'record': record_switch,
    'save and quit': ag.stop,
    'intrinsic_calibration': in_calibration_switch,
    'extrinsic_calibration': ex_calibration_switch
}


@app.route('/api', methods=['GET','POST'])
def apiRouter():
    # print(request.method)
    if request.method == 'POST':
        print(request.form['action'])
        api_switch.get(request.form['action'])()
        ag.run()
    if request.method == 'GET':
        api_switch.get(request.values['action'])()
    return redirect(url_for('index'), code=302)

# this is a placeholder, mimicking a post request using a get request
# @app.route('/api/stop')
# def stop_running():
#   ag.stop()
  # socketio.emit('stopped')


@app.route('/video/ex-calibration')
def ex_calibration_switch():
    ag.cameras[0].extrinsic_calibration_switch()
    return Response(ag.cameras[0].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/in-calibration')
def in_calibration_switch():
    ag.cameras[0].intrinsic_calibration_switch()
    return Response(ag.cameras[0].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
  app.run(host='127.0.0.1', port=3001, debug=False, use_reloader=False)
