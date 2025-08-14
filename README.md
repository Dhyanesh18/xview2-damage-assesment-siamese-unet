# Siamese U-Net for Building Damage Assessment

This project implements a **Siamese U-Net** for pixel-wise building damage classification using the [xView2 dataset](https://xview2.org/).  
It supports **training**, **evaluation**, and a **FastAPI backend for inference**.

<img width="386" height="386" alt="Screenshot 2025-07-28 134335" src="https://github.com/user-attachments/assets/10fe09ed-185b-4d86-bfd2-e9925a5e2d2b" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img width="386" height="386" alt="predicted_mask" src="https://github.com/user-attachments/assets/286c2188-1eca-4b71-88fa-c9dc998dc317" />

---

## Features

- Siamese U-Net with dual encoders (shared weights)
- Pixel-wise segmentation (5 damage classes)
- Supports Dice + CrossEntropy loss
- Class balancing & metrics (mIoU, Dice, Pixel Acc)
- Test-time evaluation script with CSV report
-  FastAPI backend 
-  Docker-ready backend

---


## Project Structure

.  
├── app/  
│ ├── main.py # FastAPI app  
│ ├── model.py # Load model + predict helpers  
│ ├── utils.py # Color masks, overlay helpers  
├── model_architecture.py # SiameseUNet definition  
├── train_siamese_unet.py # Training script  
├── test_script.py # Test/eval script  
├── requirements.txt  
├── Dockerfile  
└── README.md  



---

  
## Requirements

```bash
# Create a virtual env (optional but recommended)
conda create -n damage-seg python=3.10
conda activate damage-seg

# Install dependencies
pip install -r requirements.txt
```



Training

Train your Siamese U-Net:
```
python train_siamese_unet.py
```
The model weights will be saved as .pth.





Evaluation

Run on your test dataset:
```
python test_script.py
```
Outputs:

    Predicted masks: ./test/predicted_masks/

    CSV report: ./test/metrics_report.csv

    Metrics: mIoU, Dice, per-class IoU, per-class Pixel Accuracy


## Example Evaluation Results

| Metric | Value |
|----------------|-----------|
| **Mean IoU**   | 0.6008 |
| **Mean Dice**  | 0.6298 |
| **Overall Pixel Accuracy** | 0.9551 |

**Per-class:**

| Class | IoU | Dice | Pixel Acc |
|-------|------|------|----------------|
| Class 0 | 0.9659 | 0.9813 | 0.9702 |
| Class 1 | 0.4664 | 0.5455 | 0.9666 |
| Class 2 | 0.3264 | 0.3456 | 0.9816 |
| Class 3 | 0.5744 | 0.5863 | 0.9941 |
| Class 4 | 0.6707 | 0.6902 | 0.9977 |

*These numbers are from the test set. They show strong segmentation of undamaged areas and reasonable damage class detection. There is scope to further improve damage class performance with more data, augmentations, and advanced training strategies.*

---



FastAPI Inference 

Run the backend:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test the /predict endpoint:

    Use Swagger UI: http://127.0.0.1:8000/docs

    Or Postman / curl with multipart/form-data for pre_disaster and post_disaster images.





License

MIT — free for research and personal projects.





Acknowledgements

    xView2 Dataset

    segmentation-models-pytorch
