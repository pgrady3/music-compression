import os
from runner.train_frequency_autoencoder import TrainerFrequencyAutoencoder
import argparse


def parse_args():
  parser = argparse.ArgumentParser(
      description='Parameters for training frequency autoencoder')

  # Dataset setup
  parser.add_argument('--data_dir', type=str, default='data/fma_xs/')
  parser.add_argument('--model_dir', type=str, default='model_logs/')
  parser.add_argument('--num_classes', type=int, default=8)
  parser.add_argument('--snippet_len', type=int, default=512*128)

  # Training setup
  parser.add_argument('--cuda', type=bool, default=True)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--weight_decay', type=float, default=1e-5)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--load_encoder', action='store_true')
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--latent_ch', type=int, default=8)

  # MLP classifier
  parser.add_argument('--num_class_epochs', type=int, default=50)

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parse_args()
  print(args)

  trainer = TrainerFrequencyAutoencoder(args)

  if not args.load_encoder:
    trainer.train(num_epochs=args.num_epochs)

  trainer.train_classifier(num_epochs=args.num_class_epochs)
