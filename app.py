
import gradio as gr
import torch
import os

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names 
class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation
effnetb2, effnet_b2_transforms = create_effnetb2_model(num_classes=3)

# Load save weights 
effnetb2.load_state_dict(
    torch.load(
        f = 'effnetb2.pth',
        map_location = torch.device('cpu') # Load the model to CPU
    )
)

### 3. Predict function 
def predict(img) -> Tuple[Dict, float]:
  # Start a timer 
  start_time = timer()

  # Transform the input image for use with effnetb2
  img = effnetb2_transforms(img).unsqueeze(0) 

  # Put model into eval mode, make prediction
  effnetb2.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probabilities
    pred_probs = torch.softmax(effnetb2(img), dim = 1)

  # Create a prediction label and prediction probability dict
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

### 4. Gradio app 

# Create title, description and article
title = 'Pizza, Steak and Sushi'
description = 'An EfficientNetB2 Feature extractor'
article = 'Pytorch model deployment'

# Create example list 
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Create gradio demo
demo = gr.Interface(fn = predict,
                    inputs = gr.Image(type = 'pil'),
                    outputs = [gr.Label(num_top_classes = 3, label = 'Predictions'),
                               gr.Number(label = 'Prediction time (s)')],
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article)

demo.launch(debug = False,
            share = True)
