import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wavfile
from transformers import pipeline

# Load pipelines
narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# Function to generate audio from text
def generate_audio(text):
    narrated_text = narrator(text)
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"], data=narrated_text["audio"][0])
    return "output.wav"

# Function to read and summarize detected objects
def read_objects(detection_objects):
    object_counts = {}
    for detection in detection_objects:
        label = detection['label']
        object_counts[label] = object_counts.get(label, 0) + 1

    response = "This picture contains"
    labels = list(object_counts.keys())
    for i, label in enumerate(labels):
        response += f" {object_counts[label]} {label}"
        if object_counts[label] > 1:
            response += "s"
        if i < len(labels) - 2:
            response += ","
        elif i == len(labels) - 2:
            response += " and"
    response += "."
    return response

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, detections):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"
        text_size = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image

# Main function to process the image
def detect_object(image):
    detections = object_detector(image)
    processed_image = draw_bounding_boxes(image, detections)
    description_text = read_objects(detections)
    processed_audio = generate_audio(description_text)
    return processed_image, processed_audio

# Gradio interface
description_text = """
# Multi-Object Detection with Audio Narration

Upload an image to detect objects and hear a natural language description.

### Credits:
Developed by Taizun S
"""

demo = gr.Interface(
    fn=detect_object,
    inputs=gr.Image(label="Upload an Image", type="pil"),
    outputs=[
        gr.Image(label="Processed Image", type="pil"),
        gr.Audio(label="Generated Audio")
    ],
    title="Multi-Object Detection and Narration",
    description=description_text,
)

demo.launch()
