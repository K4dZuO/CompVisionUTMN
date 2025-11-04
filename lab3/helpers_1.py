import numpy as np
from math import ceil, exp, sqrt
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO


# ========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ========================

def _separable_convolve(image: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    h, w = image.shape
    k = len(kernel_1d)
    pad = k // 2

    # Паддинг
    padded = np.pad(image, pad, mode='reflect') # для сохранения данных
    temp = np.zeros((h, w), dtype=np.float32)  # промежуточный без паддинга
    output = np.zeros((h, w), dtype=np.float32)

    # Горизонтальная свёртка → результат в temp (размер h x w)
    for i in range(h):
        for j in range(w):
            temp[i, j] = np.dot(padded[i + pad, j:j + k], kernel_1d)

    # Паддинг для temp по вертикали
    temp_padded = np.pad(temp, ((pad, pad), (0, 0)), mode='reflect')

    # Вертикальная свёртка
    for i in range(h):
        for j in range(w):
            output[i, j] = np.dot(temp_padded[i:i + k, j], kernel_1d)

    return output


def _manual_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Ручная 2D свёртка для несепарабельных ядер."""
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(patch * kernel)
    return output


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        size = 5
    if sigma <= 0:
        sigma = 1.0
    center = size // 2
    x = np.arange(size) - center
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


# Для совместимости с Shi-Tomasi (ранее использовал _gaussian_kernel)
def _gaussian_kernel(size: int, sigma: float = 5) -> np.ndarray:
    """2D гауссово ядро (для несепарабельной свёртки)."""
    if size <= 0 or size % 2 == 0:
        size = 5
    if sigma <= 0:
        sigma = 1.0
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    denom = 2 * sigma ** 2
    total = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            val = exp(-(x*x + y*y) / denom)
            kernel[i, j] = val
            total += val
    return kernel / total


# ========================
# ОСНОВНЫЕ ФУНКЦИИ
# ========================

def get_histogram(image_2d: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int32)
    flat = image_2d.ravel()
    for val in flat:
        if 0 <= val < 256:
            hist[int(val)] += 1
    return hist


def draw_histogram_image(hist: np.ndarray, color: str = 'gray') -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(hist)), hist, width=1, color=color.lower(), edgecolor='none')
    ax.set_title(color.upper())
    ax.set_xlim(0, 255)
    ax.set_ylim(0, hist.max() * 1.1)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)

    pil_img = Image.open(buf)
    return pil_img


def custom_colormap_pillow(image_np):
    img = Image.fromarray(image_np).convert("L")
    gray = np.array(img)
    norm = (gray - gray.min()) / (np.ptp(gray) + 1e-5) * 255
    norm = norm.astype(np.uint8)

    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = 255 - norm
    colored[..., 1] = np.abs(128 - norm) * 2
    colored[..., 2] = norm
    return colored


def custom_colormap_manual(image_np):
    if image_np.shape[2] == 3:
        gray = 0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2]
    else:
        gray = image_np.astype(np.float32)
    gray = gray.astype(np.float32)
    norm = (gray - gray.min()) / (np.ptp(gray) + 1e-5)

    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = (norm * 255).astype(np.uint8)
    colored[..., 1] = ((1 - np.abs(norm - 0.5) * 2) * 255).astype(np.uint8)
    colored[..., 2] = ((1 - norm) * 255).astype(np.uint8)
    return colored


def avarage_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if image is None:
        raise ValueError("Изображение не может быть None")
    if len(image.shape) != 3:
        raise ValueError("Изображение должно быть 3D (H, W, C)")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        kernel_size = 5

    kernel_1d = np.ones(kernel_size, dtype=np.float32) / kernel_size
    filtered = np.empty_like(image, dtype=np.float32)

    for c in range(image.shape[2]):
        filtered[:, :, c] = _separable_convolve(image[:, :, c].astype(np.float32), kernel_1d)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    if image is None:
        raise ValueError("Изображение не может быть None")
    if len(image.shape) != 3:
        raise ValueError("Изображение должно быть 3D (H, W, C)")
    if sigma <= 0:
        sigma = 1.0

    size = int(2 * ceil(3 * sigma)) + 1
    kernel_1d = _gaussian_kernel_1d(size, sigma)
    filtered = np.empty_like(image, dtype=np.float32)

    for c in range(image.shape[2]):
        filtered[:, :, c] = _separable_convolve(image[:, :, c].astype(np.float32), kernel_1d)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def get_hf_simple(original_image: np.ndarray, lf_image: np.ndarray, c: float) -> np.ndarray:
    high_pass_img = original_image.astype(np.float32) - c * lf_image.astype(np.float32)
    min_val = high_pass_img.min()
    max_val = high_pass_img.max()
    if max_val > min_val:
        high_pass_img = 255 * (high_pass_img - min_val) / (max_val - min_val)
    else:
        high_pass_img = np.zeros_like(high_pass_img)
    return np.clip(high_pass_img, 0, 255).astype(np.uint8)


def apply_convolution_filter(image: np.ndarray, kernel: np.ndarray, normalize: bool = True, add_128: bool = False) -> np.ndarray:
    if image is None or kernel is None:
        raise ValueError("Изображение и ядро не могут быть None")
    if len(kernel.shape) != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Ядро должно быть квадратной матрицей")
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")

    if normalize:
        s = np.sum(kernel)
        if abs(s) > 1e-10:
            kernel = kernel / s

    if len(image.shape) == 2:
        result = _manual_convolve2d(image.astype(np.float32), kernel)
    else:
        h, w, c = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        for ch in range(c):
            result[:, :, ch] = _manual_convolve2d(image[:, :, ch].astype(np.float32), kernel)

    if add_128:
        result += 128

    return np.clip(result, 0, 255).astype(np.uint8)


def parse_kernel_from_string(kernel_str: str, size: int) -> np.ndarray:
    elements = kernel_str.replace(',', ' ').split()
    try:
        values = [float(x) for x in elements]
    except ValueError:
        raise ValueError("Не удалось преобразовать элементы матрицы в числа")
    if len(values) != size * size:
        raise ValueError(f"Ожидается {size * size} элементов, получено {len(values)}")
    return np.array(values).reshape(size, size)


def get_standard_kernels():
    kernels = {
        "Размытие 3x3": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        "Размытие 5x5": np.ones((5, 5)) / 25,
        "Резкость": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Собель X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Собель Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "Лапласиан": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        "Лапласиан диагональный": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Превитт X": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "Превитт Y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    }
    return kernels


def _sobel_gradients(gray: np.ndarray):
    h, w = gray.shape
    padded = np.pad(gray, 1, mode='reflect')

    Ix = (
        -padded[1:-1, :-2] + padded[1:-1, 2:] +
        -2 * padded[:-2, :-2] + 2 * padded[:-2, 2:] +
        -padded[2:, :-2] + padded[2:, 2:]
    )

    Iy = (
        -padded[:-2, 1:-1] - 2 * padded[:-2, :-2] - padded[:-2, 2:] +
        padded[2:, 1:-1] + 2 * padded[2:, :-2] + padded[2:, 2:]
    )

    return Ix.astype(np.float32), Iy.astype(np.float32)


def harris_corner_detection(image: np.ndarray, k: float = 0.04, threshold: float = 0.01) -> np.ndarray:
    if image is None or len(image.shape) not in [2, 3]:
        raise ValueError("Неверный формат изображения")

    if len(image.shape) == 3:
        gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    else:
        gray = image.astype(np.float32)

    Ix, Iy = _sobel_gradients(gray)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

    kernel_1d = _gaussian_kernel_1d(5, 1.0)
    Ixx = _separable_convolve(Ixx, kernel_1d)
    Iyy = _separable_convolve(Iyy, kernel_1d)
    Ixy = _separable_convolve(Ixy, kernel_1d)

    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    R = det - k * (trace * trace)
    R = np.maximum(R, 0)
    if R.max() > 0:
        R /= R.max()

    local_max = np.zeros_like(R, dtype=bool)
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for dy, dx in shifts:
        shifted = np.roll(np.roll(R, dy, axis=0), dx, axis=1)
        local_max |= (R >= shifted)
    local_max[:1] = local_max[-1:] = False
    local_max[:, :1] = local_max[:, -1:] = False

    corners = (R > threshold) & local_max

    result = image.copy()
    if result.ndim == 2:
        result = np.stack([result, result, result], axis=-1)
    result[corners] = [255, 0, 0]
    return result


def _shi_tomasi_response(Ixx, Iyy, Ixy):
    T = Ixx + Iyy
    D = Ixx * Iyy - Ixy * Ixy
    discriminant = T * T - 4 * D
    discriminant = np.maximum(discriminant, 0)
    sqrt_disc = np.sqrt(discriminant)
    lambda1 = (T + sqrt_disc) / 2
    lambda2 = (T - sqrt_disc) / 2
    return np.minimum(lambda1, lambda2)


def shi_tomasi_corner_detection(image: np.ndarray, max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10) -> np.ndarray:
    if image is None:
        raise ValueError("Изображение не может быть None")
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")

    if max_corners <= 0: max_corners = 100
    if not (0 < quality_level < 1): quality_level = 0.01
    if min_distance <= 0: min_distance = 10

    if len(image.shape) == 3:
        gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        result = image.copy()
        if len(result.shape) == 2:
            result = np.stack([result, result, result], axis=2)
        return result

    gray = gray.astype(np.float32)
    Ix, Iy = _sobel_gradients(gray)

    # Используем 2D-свёртку, так как в Shi-Tomasi размытие не обязано быть сепарабельным
    Ixx = _manual_convolve2d(Ix * Ix, _gaussian_kernel(5, 1.0))
    Iyy = _manual_convolve2d(Iy * Iy, _gaussian_kernel(5, 1.0))
    Ixy = _manual_convolve2d(Ix * Iy, _gaussian_kernel(5, 1.0))

    response = _shi_tomasi_response(Ixx, Iyy, Ixy)
    response = np.maximum(response, 0)
    max_resp = response.max()
    if max_resp == 0:
        threshold = 0
    else:
        threshold = quality_level * max_resp

    corner_mask = response >= threshold
    coords = np.argwhere(corner_mask)
    if len(coords) == 0:
        result = image.copy()
        if len(result.shape) == 2:
            result = np.stack([result, result, result], axis=2)
        return result

    selected = []
    for y, x in coords:
        too_close = False
        for sy, sx in selected:
            if (y - sy) ** 2 + (x - sx) ** 2 < min_distance ** 2:
                too_close = True
                break
        if not too_close:
            selected.append((y, x))
        if len(selected) >= max_corners:
            break

    result = image.copy()
    if len(result.shape) == 2:
        result = np.stack([result, result, result], axis=2)

    for y, x in selected:
        cv2_style_circle(result, x, y, radius=3, color=(0, 255, 0))
    return result


def cv2_style_circle(img, cx, cy, radius=3, color=(255, 255, 255)):
    h, w = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                y, x = cy + dy, cx + dx
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = color


def sobel_edge_detection(image: np.ndarray, add_128: bool = True) -> np.ndarray:
    if image is None:
        raise ValueError("Изображение не может быть None")
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")

    if len(image.shape) == 3:
        gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)

    Gx, Gy = _sobel_gradients(gray)
    magnitude = np.hypot(Gx, Gy)

    max_mag = magnitude.max()
    if max_mag > 0:
        magnitude = magnitude / max_mag * 255

    if add_128:
        magnitude += 128

    magnitude = np.clip(magnitude, 0, 255)
    result = np.stack([magnitude, magnitude, magnitude], axis=2)
    return result.astype(np.uint8)


def non_max_suppression(magnitude, angle):
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = r = 0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            elif 112.5 <= a < 157.5:
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
    return suppressed


def double_threshold(img, low, high):
    strong, weak = 255, 75
    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, strong, weak


def hysteresis(img, strong, weak):
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == weak:
                neighbors = img[i-1:i+2, j-1:j+2]
                if strong in neighbors:
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    if image is None or len(image.shape) not in [2, 3]:
        raise ValueError("Неверный формат изображения")

    low_threshold = max(0, low_threshold)
    high_threshold = max(low_threshold + 1, high_threshold)

    if image.ndim == 3:
        gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    else:
        gray = image.astype(np.float32)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        result = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image.copy()
        return result

    blurred = _separable_convolve(gray, _gaussian_kernel_1d(5, 1.0))
    Gx, Gy = _sobel_gradients(blurred)
    magnitude = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx)

    suppressed = non_max_suppression(magnitude, angle)
    thresholded, strong, weak = double_threshold(suppressed, low_threshold, high_threshold)
    edges = hysteresis(thresholded, strong, weak)

    return np.stack([edges, edges, edges], axis=-1)
