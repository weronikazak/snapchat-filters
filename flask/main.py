from flask import Flask, render_template, Response, request
from camera import Camera

app = Flask(__name__)
camera = Camera()

def gen(camera):
    while True:
        frame = camera.return_effect()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == "POST":
		effect = request.form['btn']
		camera.effect = effect
	return render_template('index.html')


@app.route('/video_feed')
def start_streaming():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

