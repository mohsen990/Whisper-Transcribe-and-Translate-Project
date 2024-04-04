
import whisper
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

start_time = time.time()
model = whisper.load_model("tiny" , device="cpu")
result = model.transcribe(audio= pathFile , fp16=False)

#print(result["text"])
# write the result the tsv format with start and end time 
writer = get_writer( 'tsv', './output/')
writer(result, f'transcribe_'+ fileName +'.{format}')
# write the result the tsv format with start and end time 
writer = get_writer( 'txt', './output/')
writer(result, f'transcribe_'+ fileName +'.{format}')


print("Precess finished!")
print("--- %s seconds (USE CPU)---" % (time.time() - start_time))
 