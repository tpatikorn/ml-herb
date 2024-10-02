from flask import Flask, render_template, Response, Blueprint
import cv2

bp = Blueprint('ml-herb', __name__, template_folder='templates', static_folder='static')

# camera_url = ''
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
@bp.route('/Home')
def index():
    return render_template('index.html')


@bp.route('/Data')
def data():
    return render_template('Data.html')


@bp.route('/scan')
def scan():
    return render_template('scan.html')


@bp.route('/test')
def test():
    return render_template('test.html')


@bp.route('/About')
def about():
    return render_template('About.html')


@bp.route('/camera')
def camera():
    return render_template('camera.html')


@bp.route('/Lam')  # ขี้เหล็ก
def Lam():
    return render_template('plants/Lam.html')


@bp.route('/Wildbetat')  # ชะพลู
def Wildbetat():
    return render_template('plants/Wildbetat.html')


@bp.route('/Kaffir')  # มะกรูด
def Kaffir():
    return render_template('plants/Kaffir.html')


@bp.route('/Senegalia')  # ชะอม
def Senegalia():
    return render_template('plants/Senegalia.html')


@bp.route('/Cilantro')  # ผักชี
def Cilantro():
    return render_template('plants/Cilantro.html')


@bp.route('/basil')  # โหระพา
def basil():
    return render_template('plants/basil.html')


@bp.route('/Ocimum')  # กะเพรา
def Ocimum():
    return render_template('plants/Ocimum.html')


@bp.route('/Celery')  # ขึ้นฉ่าย
def Celery():
    return render_template('plants/Celery.html')


@bp.route('/Leek')  # กุยช่าย
def Leek():
    return render_template('plants/Leek.html')


@bp.route('/Pandan')  # เตย
def Pandan():
    return render_template('plants/Pandan.html')


@bp.route('/Ivy')  # ตำลึง
def Ivy():
    return render_template('plants/Ivy.html')


@bp.route('/Galanga')  # ข่า
def Galanga():
    return render_template('plants/Galanga.html')


@bp.route('/Turmeric')  # ขมิ้น
def Turmeric():
    return render_template('plants/Turmeric.html')


@bp.route('/Fingerroot')  # กระชาย
def Fingerroot():
    return render_template('plants/Fingerroot.html')


@bp.route('/Ginger')  # ขิง
def Ginger():
    return render_template('plants/Ginger.html')


@bp.route('/Lapine')  # ตะไคร้
def Lapine():
    return render_template('plants/Lapine.html')


@bp.route('/Tonhom')  # ต้นหอม
def Tonhom():
    return render_template('plants/Tonhom.html')


@bp.route('/Lime')  # มะนาว
def Lime():
    return render_template('plants/Lime.html')


@bp.route('/Mamuang')  # มะม่วงหาวมะนาวโห่
def Mamuang():
    return render_template('plants/Mamuang.html')


@bp.route('/Marakinoc')  # มะระขี้นก
def Marakinoc():
    return render_template('plants/Marakinoc.html')


@bp.route('/About')
def About():
    return render_template('About.html')


app = Flask(__name__)
app.register_blueprint(bp, url_prefix='/ml-herb')
app.config['APPLICATION_ROOT'] = '/ml-herb'

if __name__ == "__main__":
    app.run(debug=True, port=8083)
