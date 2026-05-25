model.train(
    data="your_dataset.yaml",
    epochs=50,
    imgsz=640,
    freeze=22,  # Freezes the first 22 layers depending on the dataset size
    batch=16,
    device=0
)
