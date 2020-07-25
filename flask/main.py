from flask import Flask, render_template, Response, request
from camera import Camera


app = Flask(__name__)


def gen(camera, effect="mirrors"):
    while True:
        if effect == "contours":
            frame = camera.effect_canny()
        elif effect == "baby":
            frame = camera.effect_baby_face()
        elif effect == "blurr":
            frame = camera.effect_bluring_face()
        elif effect == "cartoon":
            frame = camera.effect_cartoon()
        elif effect == "doggy":	
            frame = camera.effect_dog_face()
        elif effect == "large":	
            frame = camera.effect_enlarged()
        elif effect == "mirrors":	
            frame = camera.effect_mirror()
        elif effect == "triangle":	
            frame = camera.effect_delaunay_triangle()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/', methods=["GET", "POST"])
def button():
    if request.method == "POST":
        print(request.form["btn"])
        return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def start_streaming():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

