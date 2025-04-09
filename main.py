from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import nest_asyncio
import uvicorn
import cv2
import numpy as np
import os
import uuid
import yaml
import torch

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Create output directory if it doesn't exist
OUTPUT_DIR = "output_videos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize FastAPI App
app = FastAPI(title="Traffic Sign Recognition API")

# CORS setup (Frontend Communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory to serve videos
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Load YOLOv8 Trained Model
model_path = "/content/drive/MyDrive/German/yolov8_low_visibility_trained.pt"
model = YOLO(model_path)

# Load class names from data.yaml
with open("/content/drive/MyDrive/German/YoloOutputs/yolo-data-augmented/data.yaml", 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

@app.get("/")
def home():
    return {"message": "Traffic Sign Recognition API is running! Use /predict/image or /predict/video endpoints."}

@app.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    """Process a single image and return traffic sign detections"""
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Perform detection
    results = model(image)

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls.cpu().numpy())  # Class ID
        class_name = class_names[class_id]     # Get Class Name from YAML

        detections.append({
            "class": class_name,  # Send Class Name directly
            "confidence": float(box.conf.cpu().numpy()),  # Confidence score
            "x_min": float(box.xyxy[0][0].cpu().numpy()),  # Bounding box coordinates
            "y_min": float(box.xyxy[0][1].cpu().numpy()),
            "x_max": float(box.xyxy[0][2].cpu().numpy()),
            "y_max": float(box.xyxy[0][3].cpu().numpy()),
        })

    return {"detections": detections}

@app.post("/predict/video/")
async def predict_video(file: UploadFile = File(...)):
    """Process a video and return the path to processed video with detections"""
    # Generate unique filenames for input and output videos
    unique_id = str(uuid.uuid4())
    input_video_path = f"input_{unique_id}.mp4"
    output_filename = f"output_{unique_id}.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, output_filename)

    # Read uploaded video
    video_bytes = await file.read()

    # Save uploaded video
    with open(input_video_path, "wb") as f:
        f.write(video_bytes)

    # Open video for processing
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output Video Writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict using YOLOv8
        results = model(frame)

        # Collect frame detections
        frame_detections = []
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy())
            class_name = class_names[class_id]

            frame_detections.append({
                "frame": frame_count,
                "class": class_name,
                "confidence": float(box.conf.cpu().numpy()),
                "x_min": float(box.xyxy[0][0].cpu().numpy()),
                "y_min": float(box.xyxy[0][1].cpu().numpy()),
                "x_max": float(box.xyxy[0][2].cpu().numpy()),
                "y_max": float(box.xyxy[0][3].cpu().numpy()),
            })

        all_detections.extend(frame_detections)

        # Draw Detections on Frame
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    # Clean up input file
    if os.path.exists(input_video_path):
        os.remove(input_video_path)

    # Create download URLs
    download_url = f"/download/{output_filename}"
    direct_file_url = f"/files/{output_filename}"

    return {
        "message": "Video processed successfully!",
        "frames_processed": frame_count,
        "download_url": download_url,
        "direct_file_url": direct_file_url,
        "detections": all_detections
    }

# Endpoint to download the processed video
@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a processed video file"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="video/mp4"
        )
    return {"error": "File not found"}







