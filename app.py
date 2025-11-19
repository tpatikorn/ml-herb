import cv2
from flask import Flask, render_template, Response, Blueprint

bp = Blueprint('ml-herb', __name__, template_folder='templates', static_folder='static')

cap = cv2.VideoCapture(0)


@bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 255), 2)
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@bp.route('/')
@bp.route('/home')
def index():
    return render_template('index.html')


@bp.route('/data')
def data():
    return render_template('data.html')


@bp.route('/camera')
def camera():
    return render_template('camera.html')


@bp.route('/get_plant/<plant>')
def get_plant(plant):
    # "Lam":  # ขี้เหล็ก
    # "Wildbetat":  # ชะพลู
    # 'Kaffir':  # มะกรูด
    # 'Senegalia':  # ชะอม
    # 'Cilantro':  # ผักชี
    # 'basil':  # โหระพา
    # 'Ocimum':  # กะเพรา
    # 'Celery':  # ขึ้นฉ่าย
    # 'Leek':  # กุยช่าย
    # 'Pandan':  # เตย
    # 'Ivy':  # ตำลึง
    # 'Galanga':  # ข่า
    # 'Turmeric':  # ขมิ้น
    # 'Fingerroot':  # กระชาย
    # 'Ginger':  # ขิง
    # 'Lapine':  # ตะไคร้
    # 'Tonhom':  # ต้นหอม
    # 'Lime':  # มะนาว
    # 'Mamuang':  # มะม่วงหาวมะนาวโห่
    # 'Marakinoc':  # มะระขี้นก
    template_file = f'plants/{plant}.html'
    return render_template(template_file)


@bp.route('/about')
def about():
    return render_template('about.html')


@bp.route('/mobile')
def mobile():
    return render_template('mobile.html')


app = Flask(__name__)
app.register_blueprint(bp, url_prefix='/ml-herb')
app.config['APPLICATION_ROOT'] = '/ml-herb'

if __name__ == "__main__":
    app.run(debug=True, port=18080)
