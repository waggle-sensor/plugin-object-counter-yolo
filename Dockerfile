FROM waggle/plugin-base:1.1.1-ml

COPY requirements.txt app.py /app/
COPY detection/ /app/detection
COPY tool/ /app/tool
COPY yolov4.weight yol
ov4.cfg /app/


RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /app/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
