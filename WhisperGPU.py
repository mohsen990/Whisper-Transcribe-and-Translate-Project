# install torch
# python -m pip install torch
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


import whisper
import torch
import time
import os
from whisper.utils import get_writer

File1 = "english_padcast.mp3"  # AudioFilename
File2 = "sampleVideo.mp4"  # VideoFilename
File3 ="erdbeben-auf-haiti.mp3" # GermanyFile
File4 = "largefile.mp4" # VideoLargeFile
fileName = File4
current_dir = ''
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = current_dir +  "\\Data\\"
pathFile = current_dir + fileName


# It will print out the GPU that you are using.
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

start_time = time.time()
print("Process is strating...")
model = whisper.load_model("tiny", device="cuda")
result = model.transcribe(audio= pathFile , fp16=False)


#print(result["text"])
# write the result the tsv format with start and end time 
writer = get_writer( 'tsv', './output/')
writer(result, f'transcribe_'+ fileName +'.{format}')
# write the result the tsv format with start and end time 
writer = get_writer( 'txt', './output/')
writer(result, f'transcribe_'+ fileName +'.{format}')

print("Precess finished!")
print("--- %s seconds (USE GPU)---" % (time.time() - start_time))