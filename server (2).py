from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import numpy as np
import SimpleITK as sitk
import nrrd
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Activation
from tensorflow.keras import Model

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)
app.config['DEBUG'] = False
CORS(app)  # Enable CORS for all routes

# Load the trained model
# model = load_model(r"resnet152v2_without_data_augumentation.h5")

def build_autoencoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
   
    # Encoder
    x = Conv3D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    # Decoder
    x = Conv3DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    x = Conv3DTranspose(2, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   
    # Output
    outputs = Conv3DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
   
    return Model(inputs, outputs)

input_shape = (128, 128, 128, 1)  # Assuming input dimensions are 128x128x128x1
model = build_autoencoder(input_shape)
model.load_weights(r"test_autoencoder.h5")

#PREPROCESSING PIPELINE
#OTSU THRESHOLDING
import SimpleITK as sitk
import numpy as np
import os
from math import floor, ceil
import nrrd
from scipy.ndimage import zoom
import nibabel as nib

def calculate_otsu_threshold(input_file_path,output_file_path):
    
    image = sitk.ReadImage(input_file_path)
    histogram = sitk.GetArrayViewFromImage(image).flatten()
    hist, bins = np.histogram(histogram, bins=256, range=(0, 256), density=True)
    sum_total = np.sum(hist)
    sum_intensity = np.dot(bins[:-1], hist)
    sumB = 0
    wB = 0
    maximum = 0
    threshold = 0
    
    for i in range(0, 256):
        wB += hist[i]
        if wB == 0:
            continue
        
        wF = sum_total - wB
        if wF == 0:
            break
        
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_intensity - sumB) / wF
        
        between = wB * wF * (mB - mF) * (mB - mF)
        
        if between >= maximum:
            threshold = i
            maximum = between
    
    binary_mask = sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=255, insideValue=1, outsideValue=0)
    skull_only = sitk.Mask(image, binary_mask)
    skull_array = sitk.GetArrayFromImage(skull_only)
    upper_eye_boundary = min(int(skull_array.shape[0] * 0.4), skull_array.shape[0] - 1)
    skull_region = skull_array.copy()
    skull_region[:upper_eye_boundary, :, :] = 0
    skull_region_image = sitk.GetImageFromArray(skull_region)
    skull_region_image.CopyInformation(image) 
    sitk.WriteImage(skull_region_image, output_file_path)
    print('Size after processing thresholding:', skull_region_image.GetSize()) 

def denoise_skull(input_file_path,output_file_path):
    image = sitk.ReadImage(input_file_path)
    denoised_image = sitk.Median(image)
    sitk.WriteImage(denoised_image, output_file_path)
    print('Size after processing denoising:', denoised_image.GetSize())
    
def crop_nrrd_files(input_file_path, output_file_path):
    image = sitk.ReadImage(input_file_path)
    cropped_image = image[:, :, 128:384]
    sitk.WriteImage(cropped_image, output_file_path)
    print('Size after processing cropping :', cropped_image.GetSize())


def pad_nrrd_files(input_path, output_path):
    data, header = nrrd.read(input_path)
    array = np.array(data)
    diff = 256 - array.shape[2]
    top_slices = diff // 2
    bottom_slices = diff - top_slices
    padded_array = np.pad(array, ((0, 0), (0, 0), (top_slices, bottom_slices)))
    print(f"Size after processing padding : {padded_array.shape}")
    nrrd.write(output_path, padded_array, header)


def adjust_slices(input_file_path,output_path):
    target_size_x = 512
    target_size_y = 512
    data, header = nrrd.read(input_file_path)
    current_size_x, current_size_y = data.shape[0], data.shape[1]

    diff_x = target_size_x - current_size_x
    diff_y = target_size_y - current_size_y

    if diff_x > 0:
        left_slices_x = diff_x // 2
        right_slices_x = diff_x - left_slices_x
        data = np.pad(data, ((left_slices_x, right_slices_x), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif diff_x < 0:
        left_slices_x = abs(diff_x) // 2
        right_slices_x = abs(diff_x) - left_slices_x
        data = data[left_slices_x:-right_slices_x, :, :]

    if diff_y > 0:
        top_slices_y = diff_y // 2
        bottom_slices_y = diff_y - top_slices_y
        data = np.pad(data, ((0, 0), (top_slices_y, bottom_slices_y), (0, 0)), mode='constant', constant_values=0)
    elif diff_y < 0:
        top_slices_y = abs(diff_y) // 2
        bottom_slices_y = abs(diff_y) - top_slices_y
        data = data[:, top_slices_y:-bottom_slices_y, :]
    print(f"Size after slice adjustment : {data.shape}")
    nrrd.write(output_path, data, header)

def downsample_nrrd(input_path, output_path):
    target_dimensions = (128, 128, 128) 
    data, header = nrrd.read(input_path)

    factor = [target_dim / current_dim for target_dim, current_dim in zip(target_dimensions, data.shape)]

    downsampled_data = zoom(data, factor, order=3)

    print(f"Size after downsampling : {downsampled_data.shape}")
    
    nrrd.write(output_path, downsampled_data)
    nrrd_img = sitk.ReadImage(output_path)
    sitk.WriteImage(nrrd_img, output_path)

def nrrd_to_nii(nrrd_file, nii_file):
    nrrd_img = sitk.ReadImage(nrrd_file)
    sitk.WriteImage(nrrd_img, nii_file)
    
def newAxis(input_file):
    file = nib.load(input_file)
    file = file.get_fdata()
    file = file[..., np.newaxis]
    print('Final Size of preprocessed image: ' ,file.shape)
    return file

def preprocess_image(file):
    output_folder = "/media/computer/a454b81a-4532-48c4-a200-7aad03b060e2/skull Reconstruction/servertest"

    try:
        image = sitk.ReadImage(file)
        sitk.WriteImage(image, os.path.join(output_folder, 'original_image.nrrd'))
        print('Original Size:', image.GetSize())

        calculate_otsu_threshold(output_folder + '/original_image.nrrd', output_folder+'/thresholded_image.nrrd')
        denoise_skull(output_folder + '/thresholded_image.nrrd', output_folder+'/denoised_image.nrrd')
        crop_nrrd_files(output_folder + '/denoised_image.nrrd', output_folder+'/cropped_image.nrrd')
        pad_nrrd_files(output_folder + '/cropped_image.nrrd', output_folder+'/padded_image.nrrd')
        adjust_slices(output_folder+'/padded_image.nrrd',output_folder+'/final_image.nrrd')
        downsample_nrrd(output_folder+'/final_image.nrrd', output_folder+'/final_downsampled_image.nrrd')
        nrrd_to_nii(output_folder+'/final_downsampled_image.nrrd',output_folder+'/final_downsampled_image.nii')

        preprocessed_file = newAxis(output_folder+'/final_downsampled_image.nii')
        return preprocessed_file
    except Exception as e:
        raise ValueError("Error preprocessing image: " + str(e))
        
file = '/media/computer/a454b81a-4532-48c4-a200-7aad03b060e2/skull Reconstruction/Pre Processing Final Outputs/defects_nii_test/defected_A0368.nii'


def predictor(image):
    image = np.array([image])
    prediction = model.predict(image)
    prediction = np.squeeze(prediction[0],-1)
    thresh = np.mean(prediction)
    prediction = tf.where(prediction > thresh, tf.ones_like(prediction),tf.zeros_like(prediction))
    prediction = tf.cast(prediction, tf.float64)
    print(prediction.shape)
    ni_img = nib.Nifti1Image(prediction, affine=np.eye(4))
    nib.save(ni_img, "Result.nii")
    return prediction

# Define route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("hello from flask server")
        print(request.files)
        print("yo")
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400  # Bad request
        
        # Get the file from the request
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return jsonify({'print(prediction)error': 'No selected file'}), 400  # Bad request
        
        # Preprocess the image
        image = preprocess_image(file)
        
        # Log the image array shape
        print("Image array shape:", image.shape)
        
        # Make prediction using the loaded model
        prediction = predictor(image)

        # Format prediction as JSON response
        return jsonify({'prediction': str(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal server error

if __name__ == '__main__':
    # app.run(debug=True)
    image = preprocess_image(file)
    print("Image array shape:", image.shape)
    predictor(image)