import SLCam
import threading
from flask import Flask, Response, render_template

cg = SLCam.CameraGroup()

# this section is a placeholder
# we will want to use the GUI to
cg.start(filepaths=['C:\\Users\\SchwartzLab\\Desktop\\Testing.mov'], isDisplayed=[True])
grabber = threading.Thread(target=cg.cameras[0].run)
grabber.start()

app = Flask(__name__)


# api_switch = {
#     'start_camera_group': cg.start,
# }

# @app.route('/api', methods=['POST'])
# def apiRouter():
#   if request.method == 'POST':
#     api_switch.get(request.form['action'])()

# this is a placeholder, mimicking a post request using a get request
@app.route('/api/stop')
def stop_running():
	cg.cameras[0].stop()


@app.route('/video/calibration')
def calibration_switch():
	cg.cameras[0].calibration_switch()


@app.route('/video/<int:cam_id>')
def generate_frame(cam_id):
	return Response(cg.cameras[cam_id].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
	return render_template('index.html')


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=3001, debug=True, use_reloader=False)
