# แอปพยากรณ์สภาพอากาศ จังหวัดพระนครศรีอยุธยา (Streamlit + Machine Learning)

พยากรณ์ล่วงหน้า 7 วัน: อุณหภูมิสูงสุด/ต่ำสุด, ปริมาณฝนรายวันและฝนสะสม, PM2.5 เฉลี่ยรายวัน
แสดงฝนสะสมจากข้อมูลจริง (7 วัน, 30 วัน, ตั้งแต่ต้นเดือนเทียบค่าเฉลี่ย, ตั้งแต่ต้นปี)

## แหล่งข้อมูล
- สภาพอากาศ: Open-Meteo Archive API และ Forecast API
- PM2.5: Open-Meteo Air Quality API (ค่าจากแบบจำลอง CAMS ไม่ใช่ค่าจากสถานีวัดภาคพื้นดิน)
- ไม่ต้องใช้ API key

## รันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud ผ่าน GitHub
1. สร้าง repository ใหม่บน GitHub แล้วอัปโหลดไฟล์ `app.py`, `requirements.txt`, `README.md`, `.gitignore` ไว้ที่ root
2. เข้า https://share.streamlit.io แล้วเข้าสู่ระบบด้วยบัญชี GitHub
3. เลือก **Create app** → เลือก repository, branch (`main`) และ Main file path เป็น `app.py`
4. กด **Deploy**

## หมายเหตุ
- ผลลัพธ์ถูกแคชไว้ 1 ชั่วโมง (`ttl=3600`)
- ข้อมูล PM2.5 ย้อนหลังได้ตั้งแต่ ส.ค. 2565 จึงมีข้อมูลฝึกโมเดลน้อยกว่าตัวแปรอุณหภูมิและฝน
- เกณฑ์ระดับ PM2.5 ในโค้ด (`PM25_LEVELS`) ควรตรวจสอบกับกรมควบคุมมลพิษก่อนนำไปใช้จริง
