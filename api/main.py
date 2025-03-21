from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

# FastAPI uygulamasını başlat
# Start FastAPI application
app = FastAPI()

# Model ve sınıf isimlerini yükleme
# Loading the model and class labels
MODEL_FILE_PATH = "../software proje/api/model.keras"
try:
    # Modeli belirtilen dosya yolundan yükler
    # Load the model from the specified file path
    MODEL = tf.keras.models.load_model(MODEL_FILE_PATH)
except Exception as err:
    # Model yüklenemezse hata fırlatır
    # Raise an error if the model cannot be loaded
    raise RuntimeError(f"Model yüklenemedi: {err}")

# Sınıf etiketleri
# Class labels
CLASS_LABELS = ['Early Blight', 'Healthy', 'Late Blight']

@app.get("/check")
def check():
    """
    API'nin çalışıp çalışmadığını kontrol etmek için basit bir uç nokta.
    Simple endpoint to check if the API is working.
    """
    return {"status": "OK"}

def preprocess_image(file_data: bytes) -> str:
    """
    Görüntü verisini numpy formatına dönüştür ve geçici bir dosyaya kaydet.
    Convert image data to numpy format and save to a temporary file.
    """
    try:
        # Görüntüyü RGB formatına dönüştürür ve boyutlandırır
        # Convert image to RGB format and resize
        img = Image.open(BytesIO(file_data)).convert("RGB")
        img = img.resize((128, 128))  # Modelin giriş boyutuna uygun şekilde yeniden boyutlandır
        temp_path = "temp_image.jpg"
        img.save(temp_path)  # Geçici dosyaya kaydet
        return temp_path
    except Exception as processing_err:
        # Görüntü işleme sırasında bir hata oluşursa bildirin
        # Report error if an issue occurs during image processing
        raise ValueError(f"Görüntü işleme hatası: {processing_err}")

def model_prediction(test_image: str, number: int = 0):
    """
    Modeli kullanarak tahmin yapar ve sonucu döner.
    Make predictions using the model and return the result.
    """
    model = tf.keras.models.load_model(MODEL_FILE_PATH)  # Modeli yükler
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))  # Görüntüyü model girişine uygun hale getirir
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Görüntüyü numpy array'e dönüştürür
    input_arr = np.array([input_arr])  # Tek görüntüyü bir batch'e dönüştür
    predictions = model.predict(input_arr)  # Modelden tahmin alır
    conf = np.max(predictions[number])  # En yüksek güven skorunu alır
    return np.argmax(predictions), conf  # Max elemanın indeksini ve güven skorunu döner

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Kullanıcıdan bir görüntü alır, modeli kullanarak tahmin yapar ve sonucu döner.
    Receive an image from the user, make predictions using the model, and return the result.
    """
    try:
        # Görüntüyü işleme
        # Process the image
        temp_image_path = preprocess_image(await file.read())

        # Tahmin yapma
        # Make predictions
        predicted_index, confidence_score = model_prediction(temp_image_path)
        predicted_label = CLASS_LABELS[predicted_index]

        # Confidence skoruna göre yanıt oluştur
        if float(confidence_score) > 0.9900:
            return {
                "predicted_label": predicted_label,  # Tahmin edilen sınıf etiketi
                "confidence_score": float(confidence_score) * 100  # Güven skoru (% formatında)
            }
        else:
            return {
                "predicted_label": "model tahmin edilemedi",  # Tahmin edilemediği için bilinmeyen sınıf
                "confidence_score": float(confidence_score) * 100  # Mevcut güven skoru (% formatında)
            }
        
    except Exception as prediction_err:
        # Tahmin sırasında hata oluşursa bunu döner
        # Return an error if any issue occurs during prediction
        return {"error": f"Tahmin sırasında bir hata oluştu: {prediction_err}"}

if __name__ == "__main__":
    # yerel sunucunun IP'si ile çalışıyor
    # Runs with the local server's IP
    uvicorn.run(app, host="192.168.196.88", port=8080)  # Telefon A24
    # uvicorn.run(app, host="172.21.197.5", port=8080) 
    #uvicorn.run(app, host="10.44.240.106", port=8080)  # KYK

