from ultralytics import YOLO

if __name__ == '__main__':
    # Point directly to the last saved checkpoint in your train-3 folder
    model = YOLO(r"C:\Users\admin\Desktop\hieroglyph_app\runs\detect\train-3\weights\last.pt") 
    
    # Resume the training from where it crashed
    results = model.train(resume=True)