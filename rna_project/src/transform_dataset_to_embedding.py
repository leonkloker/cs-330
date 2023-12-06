import fm
import numpy as np
import os
import pandas as pd
import torch
import tqdm

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load model
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.to(device)
model.eval()

# Load RNA data to be transformed
print("Loading data...")

df = pd.read_csv("../../csv/train_data.csv")

N = 64

# Extract RNA sequences and labels
print("Extracting RNA sequences and labels...")
df1 = df[df["experiment_type"] == "2A3_MaP"]
df2 = df[df["experiment_type"] != "2A3_MaP"]
signal_to_noise_mask = (df1["SN_filter"] == 1).values
x = np.array(df1[signal_to_noise_mask].iloc[:, 1])
y_1 = np.array(df1[signal_to_noise_mask].iloc[:, 7:213])
y_2 = np.array(df2[signal_to_noise_mask].iloc[:, 7:213])

# Cut reactivity to [0,1]
y_1 = np.clip(y_1, 0, 1)
y_2 = np.clip(y_2, 0, 1)

# Cut to N samples
x = x[:N]
y_1 = y_1[:N]
y_2 = y_2[:N]

# Transform RNA sequences to embeddings
print("Transforming RNA sequences to embeddings...")
data = []
for i in range(len(x)):
    data.append((str(i), x[i]))
labels, strs, tokens = batch_converter(data)

# Extract embeddings
print("Extracting embeddings...")
batch_size = 1024
embeddings = []
for idx in range(int(np.ceil(N/batch_size))):
    batch = tokens[idx*batch_size:(idx+1)*batch_size,:]
    batch = batch.to(device)
    with torch.no_grad():
        results = model(batch, repr_layers=[12])
    batch_embeddings = results["representations"][12]
    embeddings.append(batch_embeddings.cpu().numpy())
embeddings = np.concatenate(embeddings, axis=0)

# Save embeddings
print("Saving embeddings...")
if not os.path.exists("../data/fm_embeddings"):
    os.makedirs("../data/fm_embeddings")

for i in tqdm.tqdm(range(N)):
    np.savez("../data/fm_embeddings/{}.npz".format(i), x=embeddings[i,:,:], y1=y_1[i,:], y2=y_2[i,:])
