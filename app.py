from flask import Flask, request, jsonify
from flask_cors import CORS   # import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os
import io
import base64
from PIL import Image, ImageDraw
from matplotlib import cm

app = Flask(__name__)
CORS(app)

# Load your trained model (update the path to your model)
MODEL_PATH = './retinopathy_model.h5'
model = load_model(MODEL_PATH)

# Class labels (modify based on your training)
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']


def _connected_components(binary_mask):
    """Return connected components for a boolean mask using 8-connectivity."""
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            points = []

            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append(points)

    return components


def extract_patch_intensity(img_path, patch_size=64, percentile=92, min_component_area=12):
    """
    Compute patch intensity metrics for a retinal image.
    Returns both fixed-grid patch means and bright connected patch statistics.
    """
    rgb = np.array(Image.open(img_path).convert('RGB'))
    gray = np.array(Image.fromarray(rgb).convert('L'), dtype=np.float32)

    # Retinal disc mask: ignore near-black pixels around circular border/background.
    fundus_mask = (rgb[:, :, 0] > 10) | (rgb[:, :, 1] > 10) | (rgb[:, :, 2] > 10)
    if not np.any(fundus_mask):
        return {
            'patch_size': patch_size,
            'threshold_percentile': percentile,
            'threshold_value': None,
            'grid_patches': [],
            'bright_patches': [],
            'summary': {
                'grid_patch_count': 0,
                'bright_patch_count': 0,
                'mean_grid_intensity': None,
                'max_grid_intensity': None,
                'mean_bright_patch_intensity': None,
                'max_bright_patch_intensity': None
            }
        }

    h, w = gray.shape

    # 1) Fixed grid patch means.
    grid_patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            patch_gray = gray[y:y2, x:x2]
            patch_mask = fundus_mask[y:y2, x:x2]
            valid_ratio = float(np.mean(patch_mask))

            # Keep patches that substantially belong to the retina region.
            if valid_ratio < 0.5:
                continue

            values = patch_gray[patch_mask]
            mean_intensity = float(np.mean(values))
            grid_patches.append({
                'bbox': [int(x), int(y), int(x2), int(y2)],
                'mean_intensity': round(mean_intensity, 3),
                'pixel_count': int(values.size)
            })

    grid_patches.sort(key=lambda p: p['mean_intensity'], reverse=True)

    # 2) Bright connected patches (lesion-like clusters by intensity threshold).
    threshold_value = float(np.percentile(gray[fundus_mask], percentile))
    bright_mask = (gray >= threshold_value) & fundus_mask

    components = _connected_components(bright_mask)
    max_component_area = max(1, int(0.04 * np.sum(fundus_mask)))

    bright_patches = []
    for points in components:
        area = len(points)
        if area < min_component_area or area > max_component_area:
            continue

        ys = np.array([p[0] for p in points])
        xs = np.array([p[1] for p in points])
        vals = gray[ys, xs]

        bright_patches.append({
            'bbox': [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            'area': int(area),
            'mean_intensity': round(float(np.mean(vals)), 3),
            'max_intensity': round(float(np.max(vals)), 3)
        })

    bright_patches.sort(key=lambda p: p['mean_intensity'], reverse=True)

    grid_intensities = [p['mean_intensity'] for p in grid_patches]
    bright_intensities = [p['mean_intensity'] for p in bright_patches]

    return {
        'patch_size': patch_size,
        'threshold_percentile': percentile,
        'threshold_value': round(threshold_value, 3),
        # Limit payload size while preserving top-intensity regions.
        'grid_patches': grid_patches[:30],
        'bright_patches': bright_patches[:30],
        'summary': {
            'grid_patch_count': len(grid_patches),
            'bright_patch_count': len(bright_patches),
            'mean_grid_intensity': round(float(np.mean(grid_intensities)), 3) if grid_intensities else None,
            'max_grid_intensity': round(float(np.max(grid_intensities)), 3) if grid_intensities else None,
            'mean_bright_patch_intensity': round(float(np.mean(bright_intensities)), 3) if bright_intensities else None,
            'max_bright_patch_intensity': round(float(np.max(bright_intensities)), 3) if bright_intensities else None
        }
    }


def generate_patch_overlay_data_url(img_path, metrics, max_grid_boxes=12, max_bright_boxes=20):
    """Draw detected patch boxes on top of the retinal image and return as data URL."""
    overlay = Image.open(img_path).convert('RGB')
    drawer = ImageDraw.Draw(overlay)

    for patch in metrics.get('grid_patches', [])[:max_grid_boxes]:
        x1, y1, x2, y2 = patch['bbox']
        drawer.rectangle([x1, y1, x2, y2], outline=(0, 220, 255), width=2)

    for patch in metrics.get('bright_patches', [])[:max_bright_boxes]:
        x1, y1, x2, y2 = patch['bbox']
        drawer.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=2)

    return pil_image_to_data_url(overlay)


def build_patch_intensity_payload(img_path, patch_size=64, percentile=92):
    metrics = extract_patch_intensity(
        img_path,
        patch_size=patch_size,
        percentile=percentile
    )
    highlighted_image = generate_patch_overlay_data_url(img_path, metrics)
    return {
        **metrics,
        'highlighted_image': highlighted_image
    }

def model_predict(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    preds = model.predict(img_array)
    result = CLASS_NAMES[np.argmax(preds)]
    return result


def pil_image_to_data_url(pil_img, image_format='PNG'):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/{image_format.lower()};base64,{encoded}"


def generate_transformed_images(img_path):
    original = Image.open(img_path).convert('RGB')

    grayscale = original.convert('L')

    grayscale_arr = np.array(grayscale, dtype=np.float32) / 255.0
    thermal_arr = (cm.inferno(grayscale_arr)[:, :, :3] * 255).astype(np.uint8)
    thermal = Image.fromarray(thermal_arr)

    grayscale_data_url = pil_image_to_data_url(grayscale)
    thermal_data_url = pil_image_to_data_url(thermal)

    return grayscale_data_url, thermal_data_url

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    grayscale_image, thermal_image = generate_transformed_images(file_path)

    # Run prediction
    result = model_predict(file_path)

    # Include patch-intensity analysis in the same response.
    patch_intensity_payload = build_patch_intensity_payload(file_path)

    # Remove temporary file
    os.remove(file_path)

    return jsonify({
        'result': result,
        'grayscale_image': grayscale_image,
        'thermal_image': thermal_image,
        'patch_intensity': patch_intensity_payload
    })


@app.route('/patch-intensity', methods=['POST'])
def patch_intensity():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    patch_size = request.form.get('patch_size', default=64, type=int)
    percentile = request.form.get('percentile', default=92, type=float)

    if patch_size is None or patch_size <= 0:
        return jsonify({'error': 'patch_size must be a positive integer'}), 400
    if percentile is None or percentile <= 0 or percentile >= 100:
        return jsonify({'error': 'percentile must be between 0 and 100'}), 400

    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        payload = build_patch_intensity_payload(
            file_path,
            patch_size=patch_size,
            percentile=percentile
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(payload)

@app.route('/test', methods=['GET'])
def test():
    return "API is working!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
