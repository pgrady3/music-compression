# functions to help read and write audio files

import os
import glob
import pydub
import numpy as np
import torch


def read_mp3(file_name, normalized=False):
  '''
  Reads the mp3 file from the disk
  '''
  try:
    a = pydub.AudioSegment.from_mp3(file_name)
  except:
    print("Failed converting file: ", filename)
  y = np.array(a.get_array_of_samples())
  if a.channels == 2:
    y = preprocess_audio(y.reshape((-1, 2)))
  if normalized:
    return a.frame_rate, np.float32(y) / 2**15
  else:
    return a.frame_rate, y


def preprocess_audio(input_vec):
  '''
  Convert to single channel data
  '''
  if len(list(input_vec.shape)) == 1:
    return input_vec
  return np.mean(input_vec, axis=1)


def write_mp3(f, sr, x, normalized=False):
  """numpy array to MP3"""
  channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
  if normalized:  # normalized array - each item should be a float in [-1, 1)
    y = np.int16(x * 2 ** 15)
  else:
    y = np.int16(x)
  song = pydub.AudioSegment(y.tobytes(), frame_rate=sr,
                            sample_width=2, channels=channels)
  song.export(f, format="mp3", bitrate="441k")


def convert_audio_to_tensors(file_dir):
  '''
  Converts all the MP3 files into tensors of fixed length
  '''

  files = glob.glob(
      os.path.join(file_dir, '*.mp3')
  )

  for f in files:
    _, audio = read_mp3(f)

    output_name = os.path.join(
        file_dir, os.path.basename(f).split('.')[0] + '.pt'
    )

    tt = torch.tensor(audio, dtype=torch.float32)

    torch.save(tt, output_name)


if __name__ == '__main__':
  # framerate, audio = read_mp3('data/individual_samples/sample1.mp3')

  # print(np.nonzero(audio))

  convert_audio_to_tensors('data/sample_data/')
