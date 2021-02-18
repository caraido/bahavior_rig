# behavior_rig
Schwartz Lab Behavior Rig Project

## Web gui
To run demo:
1. Install node/npm
2. `cd` into `client` and run `npm install` followed by `npm run build`
3. `cd` into `server` and run `python3 demo_webserving.py`
4. Open a browser directed at `localhost:3001` and you should see 4 video streams playing white noise
    - Pressing the button will send a `POST` request to the Flask server which can be used to execute callbacks such as to start recording