@echo off
:: Mengubah directory aktif ke folder script batch ini berada
cd /d "%~dp0"

echo ------------------------------------------
echo Menjalankan Sistem Deteksi Kebakaran YOLO
echo ------------------------------------------

:: 0. Coba nyalakan Database MySQL (memerlukan hak akses Admin jika service mati)
echo [0/2] Memeriksa dan Memulai MySQL Database...
net start MySQL80 >nul 2>&1
net start MySQL >nul 2>&1

:: Jeda singkat agar DB siap
ping -n 2 127.0.0.1 >nul

:: 1. Jalankan API Server (Waitress/Flask)
echo [1/2] Memulai API Server...
start /d "dist\api_server" api_server.exe

:: Beri jeda 1 detik agar server siap sebelum GUI terbuka
ping -n 2 127.0.0.1 >nul

:: 2. Jalankan GUI Deteksi Kebakaran (PyQt6)
echo [2/2] Memulai Aplikasi Utama...
start /d "dist\Fire_photo" Fire_photo.exe

echo Semuanya telah berjalan!
exit
