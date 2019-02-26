"""
The file is used to use a pre-trained neural nework loaded from a checkpoint file
and apply it on a single image for inference
"""

from model_def import load_model
from get_input import get_predict_args
from data_utils import predict
import json

# Import arguments from command line
predict_arg = get_predict_args()

# Determine device
if predict_arg.gpu:
    device = "cuda"
else:
    device = "cpu"

# Reload the model from the checkpoint
model = load_model(predict_arg.checkpoint)
model.to(device)

# Obtain top K classes and related probabilities

probs, classes = predict(predict_arg.path, model, device, predict_arg.topk)

# Convert classes to names
names = []
with open(predict_arg.category_names, 'r') as f:
    cat_to_names = json.load(f)
    
for cl in classes:
    names.append(cat_to_names[cl])

for i, j in zip(names, probs):
    print("The picture shows a {}, with {}% probability. \n".format(i,round(j*100,1)))
