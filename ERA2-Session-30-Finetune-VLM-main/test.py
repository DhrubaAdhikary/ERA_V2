import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import random_split, DataLoader
import pickle
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class llavadataset(Dataset):
  def __init__(self, coco_data, phi_model_name, clip_model_name,train_flag,tokenizer):

    self.tokenizer  = tokenizer
    self.processor  = AutoProcessor.from_pretrained(clip_model_name)
    self.caption_dataset = coco_data

    train_size = int(0.9 * len(self.caption_dataset))
    print(f"Train size {train_size} and validation size {len(self.caption_dataset) - train_size}")

    if train_flag == 'train':
      self.caption_dataset = self.caption_dataset[0:train_size]
    else:
      self.caption_dataset = self.caption_dataset[train_size:]
      self.caption_dataset.reset_index(drop=True, inplace=True)



  def __len__(self):
    return len(self.caption_dataset)

  def __getitem__(self, idx):

    # from image perspective

    img_url = self.caption_dataset.loc[idx]['image_url']
    caption = self.caption_dataset.loc[idx]['caption']
    # image load
    image_load = Image.open(requests.get(img_url,stream=True).raw)
    #image_load = Image.open(img_url)
    image_processed = self.processor(images=image_load, return_tensors="pt") ['pixel_values']
    image_processed = image_processed.squeeze(0)
    a = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
    return(image_processed , a['input_ids'].squeeze(0))


def collate_fn(batch):
    image_embeddings, captions = zip(*batch)
    image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    captions_padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=50256)
    return (image_embeddings_stacked, captions_padded)

phi_model_name  = "microsoft/phi-2"
clip_model_name = "openai/clip-vit-base-patch32"
with open("D:\\New folder\\ERA_V2\\ERA2-Session-30-Finetune-VLM-main\\captions.pickle", "rb") as fp:   # Unpickling
        coco_unpickle = pickle.load(fp)

train_batch_size = 1
val_batch_size   = 1
tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)

# model
# MModalGPT        = CLIPPhi2Model().to(device)
# The maximum number of training steps (iterations) is set to 20,000.
max_steps        = 20000
model_save_step  = 100
model_val_step   = 100
log_step         = 100
# Limits the maximum number of tokens (words, subwords) in the processed inputs, likely to filter out long captions or questions.
max_token_filter = 35

# train_dataloader: A PyTorch DataLoader for the training dataset.
# llavadataset: A custom dataset class that combines the loaded COCO dataset (coco_unpickle), the tokenizer, and other settings. It processes both image and text data.
# collate_fn: A custom function (collate_fn) to pad or batch the input data correctly, including images and tokenized text.
# val_dataloader: The DataLoader for the validation dataset, which is similar to the training DataLoader but with a smaller batch size (val_batch_size=2).

# data loaders
train_dataloader = DataLoader(llavadataset(coco_unpickle, phi_model_name,clip_model_name,'train',tokenizer),
                    collate_fn=collate_fn, batch_size=train_batch_size, num_workers = 0, shuffle=True, pin_memory=True)

val_dataloader   = DataLoader(llavadataset(coco_unpickle, phi_model_name,clip_model_name,'val',tokenizer),
                    collate_fn=collate_fn, batch_size=val_batch_size, num_workers = 0, shuffle=True, pin_memory=True)


batch = next(iter(train_dataloader))

# Print the values in the batch
print("Batch values:")

for batch_idx, (images, target_captions) in enumerate(val_dataloader):
  print(batch_idx)
