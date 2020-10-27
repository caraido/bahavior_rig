import SLCam
from flask import Flask, Response, render_template


cg = SLCam.CameraGroup()
cg.start(isDisplayed=[True])  # placeholder... one camera this is displayed

app = Flask(__name__)


# api_switch = {
#     'start_camera_group': cg.start,
# }

# @app.route('/api', methods=['POST'])
# def apiRouter():
#   if request.method == 'POST':
#     api_switch.get(request.form['action'])()


@app.route('/video/<cam_id>')
def generate_frame(cam_id):
  return Response(cg.cameras[cam_id].display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=3001, debug=True, use_reloader=False)
