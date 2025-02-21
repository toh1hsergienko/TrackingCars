from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO('C:\ML\yolo11\yolo11n.pt')

# Предсказание на изображении
results = model('cars.jpg')
results[0].show()  # Показать результат