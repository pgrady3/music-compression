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
  a = pydub.AudioSegment.from_mp3(file_name)
  y = np.array(a.get_array_of_samples())
  if a.channels == 2:
    y = y.reshape((-1, 2))
  if normalized:
    return a.frame_rate, np.float32(y) / 2**15
  else:
    return a.frame_rate, preprocess_audio(y)


def preprocess_audio(input):
  '''
  Convert to single channel data
  '''
  return np.mean(input, axis=1)


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

  snippet_size = 65536
  block_size = 10000

  for f in files:
    framerate, audio = read_mp3(f)

    print('Framerate = {}'.format(framerate))
    output_name = os.path.join(
        file_dir, os.path.basename(f).split('.')[0] + '_{}.pt')

    counter = 0
    for idx in range(0, audio.shape[0]-snippet_size, block_size):
      torch.save(torch.tensor(audio[idx:idx+snippet_size], dtype=torch.float32),
                 output_name.format(counter))
      counter += 1


if __name__ == '__main__':
  # framerate, audio = read_mp3('data/individual_samples/sample1.mp3')

  # print(np.nonzero(audio))

  convert_audio_to_tensors('data/sample/061/train/')
