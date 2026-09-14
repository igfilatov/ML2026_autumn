import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error

img_path = 'image.jpg'

img_pil = Image.open(img_path)
img_np = np.array(img_pil)
h, w, c = img_np.shape

original = img_np.astype(np.float64)

min_dim = min(h, w)
ranks = [1, 5, 10, 30, 100, min_dim]

U_list = []
s_list = []
Vt_list = []
for ch in range(3):
    U, s, Vt = np.linalg.svd(original[:, :, ch], full_matrices=False)
    U_list.append(U)
    s_list.append(s)
    Vt_list.append(Vt)

approximations = []

for r in ranks:
    channels_approx = []
    for ch in range(3):
        U = U_list[ch]
        s = s_list[ch]
        Vt = Vt_list[ch]
        approx_ch = (U[:, :r] * s[:r]) @ Vt[:r, :]
        channels_approx.append(approx_ch)
    img_approx = np.stack(channels_approx, axis=2)
    approximations.append(img_approx)


plt.figure(figsize=(6, 6))
plt.imshow(original.astype(np.uint8))
plt.title("Оригинальное изображение")
plt.axis('off')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

original_bytes = h * w * 3

for i, r in enumerate(ranks):
    img = approximations[i]
    img_display = np.clip(img, 0, 255).astype(np.uint8)


    compressed_numbers = r * (h + 1 + w) * 3
    compressed_bytes = compressed_numbers * 4
    ratio = original_bytes / compressed_bytes

    axes[i].imshow(img_display)
    axes[i].set_title(f"Ранг = {r}\nСжатие в {ratio:.2f} раз")
    axes[i].axis('off')

plt.show()
#
original_shape = (h, w, 3)
list_of_ranks = ranks
original_flat = original.flatten()

# ---------- БЛОК АВТОПРОВЕРКИ (НЕ РЕДАКТИРОВАТЬ) ----------
# Предполагается, что у вас есть переменные:
# original_shape = (h, w, c) или (h, w)
# list_of_ranks = [1, 5, 10, 30, 100, ...]
# approximations = список восстановленных массивов (numpy) для каждого ранга

# 1. Проверка размерностей
for i, r in enumerate(list_of_ranks):
    assert approximations[i].shape == original_shape, f"Ошибка: размерность для ранга {r} не совпадает с исходной"

# 2. Проверка, что значения не выходят за пределы 0-255 (для RGB/серого)
for i, r in enumerate(list_of_ranks):
    arr = approximations[i]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255) # Если float, то проверяем диапазон
    assert arr.max() <= 255.1, f"Значения превышают 255 для ранга {r}"
    assert arr.min() >= -0.1, f"Значения меньше 0 для ранга {r}"

# 3. Проверка, что с ростом ранга ошибка уменьшается (метрика MSE)
from sklearn.metrics import mean_squared_error
mse_list = []
for i in range(len(approximations)):
    # Если цветная, считаем MSE по всем каналам
    mse = mean_squared_error(original_flat, approximations[i].flatten())
    mse_list.append(mse)

# Проверяем монотонность (MSE должен падать, так как мы добавляем сингулярные числа)
for i in range(1, len(mse_list)):
    assert mse_list[i] <= mse_list[i-1] + 1e-6, f"MSE не уменьшается между рангами {list_of_ranks[i-1]} и {list_of_ranks[i]}"

print("✅ Все автоматические проверки пройдены. Задание выполнено корректно!")