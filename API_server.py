from flask import Flask, request, jsonify
import os
import sys
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# Tentukan base path aplikasi (apakah dijalankan sebagai script .py atau .exe hasil compile)
if getattr(sys, 'frozen', False):
    resource_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    executable_path = os.path.dirname(sys.executable)
else:
    resource_path = os.path.dirname(os.path.abspath(__file__))
    executable_path = os.path.dirname(os.path.abspath(__file__))

# Load variabel dari file .env berdasarkan lokasi resource
dotenv_path = os.path.join(resource_path, '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(executable_path, 'api_uploads').replace("\\", "/")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- PENGATURAN DATABASE MENGGUNAKAN ENV ---
# Sekarang tidak ada lagi password yang tertulis langsung di sini
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'use_pure': True
}


@app.route('/register', methods=['POST'])
def register_user():
    try:
        # 1. Buka koneksi menggunakan parameter dari DB_CONFIG
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        data = request.json
        input_username = data.get('username')
        input_password = data.get('password')

        # 2. Cek apakah username sudah ada di database
        check_sql = "SELECT * FROM user WHERE username = %s"
        cursor.execute(check_sql, (input_username,))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Username sudah terpakai! Silakan gunakan nama lain."}), 400

        # 3. Jika belum ada, masukkan data user baru ke tabel
        insert_sql = "INSERT INTO user (username, password, status) VALUES (%s, %s, 'Offline')"
        cursor.execute(insert_sql, (input_username, input_password))
        conn.commit()

        # 4. Tutup koneksi agar tidak memory leak
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Registrasi berhasil! Silakan Login."}), 201

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500


@app.route('/login', methods=['POST'])
def login_user():
    try:
        # 1. Buka koneksi menggunakan parameter dari DB_CONFIG
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # Menerima data dari Jendela Login GUI
        data = request.json
        input_username = data.get('username')
        input_password = data.get('password')

        # Cek ke database (Untuk keamanan nyata, password harus di-hash)
        sql = "SELECT * FROM user WHERE username = %s AND password = %s"
        cursor.execute(sql, (input_username, input_password))
        user = cursor.fetchone()

        if user:
            # Jika cocok, ubah status menjadi Online dan catat waktu login
            update_sql = "UPDATE user SET status = 'Online', last_login = NOW() WHERE user_id = %s"
            cursor.execute(update_sql, (user['user_id'],))
            conn.commit()

            cursor.close()
            conn.close()

            return jsonify({"status": "success", "message": "Login berhasil!"}), 200
        else:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Username atau Password salah!"}), 401

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500
@app.route('/upload', methods=['POST'])
def receive_alert():
    # Menangkap API Key yang dikirim dari GUI
    client_key = request.headers.get('x-api-key')

    # --- 1. PROSES VALIDASI API KEY ---
    valid_key = False
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Ambil daftar API Key dari tabel api_connection
        cursor.execute("SELECT API FROM api_connection")
        api_records = cursor.fetchall()

        for row in api_records:
            if client_key == row[0]:
                valid_key = True
                break

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"=> ERROR DATABASE (VALIDASI API): {err}")
        return jsonify({'status': 'error', 'pesan': 'Gagal koneksi database'}), 500

    # Jika kunci tidak valid, langsung hentikan di sini
    if not valid_key:
        print(f"Peringatan: Akses ditolak untuk Key: {client_key}")
        return jsonify({'status': 'error', 'pesan': 'API Key tidak valid'}), 401

    # --- 2. MENANGKAP DATA (Sudah diperbaiki posisinya) ---
    # Kode ini sekarang berada di luar blok 'if not valid_key' agar bisa jalan
    room_name = request.form.get('room', 'Ruangan Tidak Diketahui')
    confidence_score = request.form.get('confidence', 0.0)

    # Cek foto
    if 'photo' not in request.files:
        return jsonify({'status': 'error', 'pesan': 'Tidak ada file foto'}), 400

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'status': 'error', 'pesan': 'File tidak valid'}), 400

    # --- 3. PROSES SIMPAN FOTO & DATA ---
    try:
        # Simpan Foto Lokal
        filename = photo.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename).replace("\\", "/")
        photo.save(save_path)

        # Simpan ke Database
        waktu_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ambil status kejadian langsung dari data form yang dikirim oleh GUI
        status_val = request.form.get('status', 'fire')

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
                INSERT INTO fire_screenshot (camera, photos, location, confidence_score, status, waktu_kejadian)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
        values = (room_name, save_path, room_name, float(confidence_score), status_val, waktu_sekarang)

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()

        # Log Sukses
        print(f"\n[FIRE] ALERT TERDAFTAR: {room_name} ({confidence_score}%)")
        return jsonify({'status': 'sukses', 'pesan': 'Data berhasil disimpan'}), 200

    except Exception as err:
        print(f"=> ERROR UPLOAD/DATABASE: {err}")
        return jsonify({'status': 'error', 'pesan': f'Gagal simpan data: {str(err)}'}), 500


@app.route('/get_history', methods=['GET'])
def get_history():
    # Menangkap API Key yang dikirim dari GUI
    client_key = request.headers.get('x-api-key')

    # --- 1. PROSES VALIDASI API KEY (Sama seperti fungsi upload) ---
    valid_key = False
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT API FROM api_connection")
        api_records = cursor.fetchall()

        for row in api_records:
            if client_key == row[0]:
                valid_key = True
                break

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"=> ERROR DATABASE (VALIDASI API): {err}")
        return jsonify({'status': 'error', 'pesan': 'Gagal koneksi database'}), 500

    if not valid_key:
        return jsonify({'status': 'error', 'pesan': 'API Key tidak valid'}), 401

    # --- 2. AMBIL DATA DARI DATABASE ---
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # Gunakan dictionary=True agar outputnya langsung menjadi JSON key:value
        cursor = conn.cursor(dictionary=True)

        # Mengambil dari tabelmu yang bernama fire_screenshot
        query = "SELECT fire_id, camera, photos, location, confidence_score, waktu_kejadian FROM fire_screenshot ORDER BY fire_id DESC"
        cursor.execute(query)

        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(data), 200

    except Exception as err:
        print(f"=> ERROR DATABASE (GET HISTORY): {err}")
        return jsonify({'status': 'error', 'pesan': f'Gagal mengambil data: {str(err)}'}), 500


if __name__ == '__main__':
    from waitress import serve
    # Mengambil port dari .env, jika tidak ada pakai default 5000
    port_api = int(os.getenv('API_PORT'))
    print(f"Server API sedang berjalan")
    print("Tekan CTRL+C di jendela ini untuk mematikan server.")

    # Menggunakan Waitress sebagai mesin server utama
    serve(app, host='0.0.0.0', port=port_api)