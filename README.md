# SAR Wind Field Estimation
### Ocean Surface Wind Retrieval from Sentinel-1 SAR Imagery
**Tamil Nadu / Gujarat Coastal Waters, India**

---

This project trains a deep learning model to estimate ocean surface wind speed and direction from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. It pulls satellite data and ERA-5 reanalysis labels through Google Earth Engine, trains a ResNet-34 on the paired data, and serves the predictions through a FastAPI endpoint running inside Google Colab. The final output is a dense wind field map over the Tamil Nadu coast that looks like a proper meteorological product.

Everything runs on a free Colab T4 GPU. You don't need any paid cloud resources, but you do need a Google account and a few one-time setup steps before the first run.

---

## Before You Start

You need three things set up before any code will run:

1. A Google Earth Engine account
2. A Google Cloud project with the Earth Engine API enabled
3. A Google Drive folder for storing the dataset and model checkpoints

The next three sections walk through each one.

---

## Step 1 — Create a Google Earth Engine Account

Go to [https://earthengine.google.com](https://earthengine.google.com) and click **Get Started**. Sign in with your Google account.

On the registration page you'll be asked what you're using GEE for — select **Research / Academia** and fill in your institution. Approval is usually instant for students and researchers, but occasionally takes a day or two.

Once approved, you'll land on the GEE Code Editor at [https://code.earthengine.google.com](https://code.earthengine.google.com). You don't need to do anything in the Code Editor for this project — it just confirms your account is active.

---

## Step 2 — Create a Google Cloud Project

GEE now requires every API call to be associated with a Google Cloud project. This sounds more involved than it is — the project is just a billing/quota container and the free tier is more than enough for this work.

**2a. Open Google Cloud Console**

Go to [https://console.cloud.google.com](https://console.cloud.google.com). Sign in with the same Google account you used for GEE.

**2b. Create a new project**

Click the project dropdown at the top of the page (it might say "My First Project" or show the name of an existing project). In the dropdown that appears, click **New Project** in the top right corner.

Give the project a name — something like `sar-wind-project` works fine. The Project ID is generated automatically but you can edit it. Write down the Project ID because you'll need it in the code. Click **Create**.

Wait a few seconds for the project to be created, then make sure it's selected in the dropdown before continuing.

**2c. Enable the Earth Engine API**

With your new project selected, go to [https://console.cloud.google.com/apis/library/earthengine.googleapis.com](https://console.cloud.google.com/apis/library/earthengine.googleapis.com).

You should see a page for the "Google Earth Engine API". Click **Enable**. It takes about 30 seconds.

If you get a prompt asking you to set up billing, you can do so by adding a credit card — you won't be charged for Earth Engine access under the non-commercial research quota, but Google requires a payment method on file for any Cloud project. Alternatively, you can activate the free trial which gives $300 in credits, more than enough.

**2d. Register your project with Earth Engine**

Go back to [https://code.earthengine.google.com](https://code.earthengine.google.com), click on your account icon in the top right, and select **Register a new Cloud Project**. Enter the Project ID you created above and confirm. This links your GEE account to the Cloud project so the Python API can authenticate.

---

## Step 3 — Set Up Google Drive

The project saves everything (raw patches, tensors, model checkpoints, figures) to your Drive at `/content/drive/MyDrive/sar_wind_data`. This folder is created automatically the first time the data pipeline runs. You don't need to create it yourself.

The only thing to verify is that you have at least **2-3 GB free** in your Drive — the full dataset of SAR patches and saved tensors can get close to that depending on how many samples you collect.

---

## Running the Project

The project is split into four notebook cells. You need to run them in order, but after the data is fetched (Cell 1) you can re-run Cells 2, 3, and 4 independently without re-fetching from GEE — they just reload from the Drive checkpoint.

Open a new Google Colab notebook at [https://colab.research.google.com](https://colab.research.google.com), make sure you're using a **T4 GPU runtime** (Runtime → Change runtime type → GPU → T4), and paste each cell in sequence.

---

### Cell 1 — Data Pipeline (`cell1_data_pipeline.py`)

This cell does all the data collection. It loops over a grid of points along the Tamil Nadu coast, fetches a Sentinel-1 SAR patch for each one, pulls a collocated ERA-5 wind label, and saves everything to Drive.

**Before running**, change the project ID in this line to match what you created in Step 2:

```python
ee.Initialize(project='sar-wind-project')   # ← put your project id here
```

When the cell runs it will call `ee.Authenticate()` first. This opens a browser tab where you log in and get a verification code — paste it back into the Colab input box when prompted. You only need to do this once per session.

The fetch loop takes a while. On a fresh run with the default 20×15 grid and 5 training windows you're looking at roughly 1500 GEE requests, which takes 30-60 minutes depending on quota. The loop checkpoints to Drive every 30 requests so if the session times out you can just re-run the cell and it picks up where it left off.

When it finishes you'll see something like:

```
done — train: 847  test: 312
```

If the training count is below 500, expand the grid or add more date windows before training.

---

### Cell 2 — Model Training (`cell2_model_training.py`)

Trains a ResNet-34 adapted for two-channel SAR input (VV and VH polarisations). The model uses a split-head architecture — one head for wind speed and one for direction — with a weighted loss that applies extra pressure on the direction head to compensate for it being harder to learn from SAR texture patterns.

Training runs for 80 epochs: the first 15 are warmup (only the two task heads train, backbone is frozen), then the backbone unfreezes and cosine annealing kicks in.

On a T4 with ~800 training samples this takes about 25-35 minutes. The best checkpoint is saved to Drive automatically. At the end you'll see the test metrics on the 2023 held-out data — a direction MAE under 40° and speed MAE under 1.5 m/s is reasonable for this dataset size.

---

### Cell 3 — Inference and Visualization (`cell3_inference_vis.py`)

This cell does three things: runs the trained model on a dense 40×40 grid over the Tamil Nadu coast, generates the wind field figure, and builds a validation table comparing predictions to ERA-5.

The wind field is plotted as a smooth contourf background (75 levels, `plasma` colourmap) with quiver arrows overlaid. With 1600 vectors the figure looks much closer to a real SAR wind product than the sparse 8×10 version.

The cell also starts a FastAPI server on port 8000 that you can query with any HTTP client:

```python
import requests
r = requests.get('http://localhost:8000/wind_field', params={
    'min_lon'    : 79.7,
    'min_lat'    : 10.5,
    'max_lon'    : 80.5,
    'max_lat'    : 13.0,
    'date_start' : '2024-08-01',
    'date_end'   : '2024-08-28',
})
print(r.json())
```

Use `localhost:8000` from inside the notebook. The public Colab proxy URL printed in the output is browser-accessible but dies when the runtime closes.

---

### Cell 4 — Report Assets (`cell4_report_assets.py`)

Generates and downloads all the files you need for the report:

| File | Contents |
|---|---|
| `sar_patch.png` | 4 sample SAR patches (VV and VH rows) |
| `training_curves.png` | Loss and error curves over 80 epochs |
| `wind_field_output.png` | The final wind field quiver map |
| `validation_results.csv` | Per-sample pred vs ERA-5 comparison |
| `metrics_summary.txt` | Final test MAE / RMSE numbers |

After running, all five files are automatically downloaded to your browser's default downloads folder.

**Figure caption for the report:**

> *Figure X. Predicted high-resolution wind field over the Tamil Nadu coastal region derived from Sentinel-1 SAR imagery. Arrow vectors indicate wind direction while colour shading represents wind speed magnitude (m/s).*

---

## Common Problems

**"Earth Engine authentication failed" or quota error**
Make sure you ran `ee.Initialize(project='your-project-id')` with the exact project ID from Cloud Console. If you see a quota exceeded message, wait a few hours — the free tier resets daily.

**Fetch loop returns 0 samples**
Usually means the date windows you chose had poor Sentinel-1 coverage for that region. The IW (Interferometric Wide) mode doesn't cover every point every day. Try widening the date window from 28 days to 45 days, or add extra windows from different seasons.

**CUDA out of memory during training**
Reduce `BATCH_SIZE` from 16 to 8 in Cell 2. This slows training slightly but fits on a 15 GB T4.

**`SARWindDataset` not defined when running Cell 2**
`SARWindDataset` is defined in Cell 1. If you started a new session and jumped straight to Cell 2, re-run Cell 1 first (or at least paste the class definition into the cell).

**Drive ran out of space**
The raw patches pickle can get large. Delete `raw_patches.pkl` from Drive if you're sure you won't need to re-fetch — `dataset_tensors.pt` is all that's needed for training and inference.

---

## Project Structure

```
sar_wind_data/               ← created on Drive automatically
├── raw_patches.pkl          ← raw numpy arrays + done-keys (resumable checkpoint)
├── dataset_tensors.pt       ← processed PyTorch tensors (train + test)
├── best_wind_model.pth      ← best checkpoint from Cell 2
├── sar_patch.png            ← report figure 1
├── training_curves.png      ← report figure 2
├── wind_field_output.png    ← report figure 3
├── validation_results.csv   ← report table 1
└── metrics_summary.txt      ← final test metrics
```

---

## Requirements

Everything installs automatically in Colab. For reference the key packages are:

```
earthengine-api
albumentations
torch + torchvision
scipy
fastapi
uvicorn
nest_asyncio
scikit-learn
```

No local installation is needed — just a Google account, a GEE registration, and a Colab session with a T4 GPU.
