import os
import numpy as np
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat

# Set paths to your dataset
data_dir = 'fabric_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Load and split dataset
train_data = DataLoader.from_folder(train_dir)
validation_data = DataLoader.from_folder(val_dir)

# Model configuration
model_spec = image_classifier.ModelSpec(
    uri='https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/5',
    input_image_shape=[224, 224]  # Standard size for MobileNet
)

# Training configuration
model = image_classifier.create(
    train_data,
    model_spec=model_spec,
    validation_data=validation_data,
    epochs=15,  # Increase if you have more data
    batch_size=32,
    train_whole_model=True,
    dropout_rate=0.2  # Helps prevent overfitting
)

# Evaluate the model
print('Evaluating model...')
loss, accuracy = model.evaluate(validation_data)
print(f'Validation accuracy: {accuracy*100:.2f}%')

# Export the model
print('Exporting model...')
model.export(
    export_dir='./stain_model/',
    tflite_filename='fabric_stain_classifier.tflite',
    label_filename='labels.txt',
    export_format=[ExportFormat.TFLITE, ExportFormat.LABEL]
)

# Optional: Export a quantized model (smaller, faster)
print('Exporting quantized model...')
model.export(
    export_dir='./stain_model/',
    tflite_filename='fabric_stain_classifier_quant.tflite',
    label_filename='labels.txt',
    quantized=True,
    quantization_config=QuantizationConfig.for_float16()  # Good balance of speed/accuracy
)

print('Training complete! Models saved in ./stain_model/')