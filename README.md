 **SpeechAct: Towards Generating Whole-body Motion from Speech** 


1.Setup environment

Create conda environment:

```
conda create -n speechact python=3.7
conda activate speechact
pip install -r requirements.txt
```

2.Prepare models

- Go to "WavLM" folder and download "WavLM-large.pt".

- Download [**smplx model**](https://drive.google.com/file/d/1Ly_hQNLQcZ89KG0Nj4jYZwccQiimSUVn/view?usp=share_link) (Please register in the official [**SMPLX webpage**](https://smpl-x.is.tue.mpg.de) before you use it.) and place it in ``path-to-SpeechAct/visualise/smplx``.

- Download [pretrained models](https://drive.google.com/file/d/1FPalJ3NK5EY_kzmBa6LChz2vN48ZNbSZ/view?usp=drive_link), extract to the folder ``path-to-SpeechAct/checkpoints``.

2.Test

```
python demo.py --infer --audio_file test.wav --speaker_names scott
```
