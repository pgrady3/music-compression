import os
import torch

from dataset_loader.music_genre_loader import MusicGenreLoader

from torch.autograd import Variable


def latent_space_analysis(compression_model,
                          data_dir,
                          is_cuda
                          ):

  # get the test set dataloader
  dataloader_args = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
  test_dataset = MusicGenreLoader(
      data_dir, split='test', snippet_size=65536)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=50, shuffle=True,
                                            **dataloader_args
                                            )
  if is_cuda:
    accumulated_latent_space = [torch.zeros(32, 255).cuda()]*8
  else:
    accumulated_latent_space = [torch.zeros(32, 255)]*8
  num_items = [0]*8

  sample_latent_space = []

  sparsity_ratio = 0
  for batch_idx, batch in enumerate(test_loader):
    # generate latent space of all the entries in the first batch

    if is_cuda:
      input_data, target_data = Variable(
          batch[0]).cuda(), Variable(batch[1]).cuda()
    else:
      input_data, target_data = Variable(batch[0]), Variable(batch[1])
    print('Processing {} items'.format(input_data.shape[0]))
    latent_data = compression_model.forward_encoder(input_data)

    if batch_idx == 0:
      for elem_idx in range(latent_data.shape[0]):
        sample_latent_space.append(
            latent_data[elem_idx].cpu().detach().numpy())

    sparsity_ratio += torch.sum(latent_data < 1e-5).cpu().detach().item() / \
        (latent_data.shape[1]*latent_data.shape[2])

    for label_idx in range(8):
      target_match = (target_data == label_idx).detach()
      latent_subdata = torch.sum(latent_data[target_match, :, :], axis=0)
      num_samples = torch.sum(target_match).item()

      # print('{} samples for label {}'.format(num_samples, label_idx))

      accumulated_latent_space[label_idx] += latent_subdata
      num_items[label_idx] += num_samples

  overall_latent_space = accumulated_latent_space[0]
  total_items = num_items[0]

  accumulated_latent_space[0] = accumulated_latent_space[0].cpu().detach(
  ).numpy()/num_items[0]
  for label_idx in range(1, 8):
    overall_latent_space += accumulated_latent_space[label_idx]
    total_items += num_items[label_idx]

    accumulated_latent_space[label_idx] = accumulated_latent_space[label_idx].cpu().detach(
    ).numpy()/num_items[label_idx]

  overall_latent_space /= total_items
  overall_latent_space = overall_latent_space.cpu().detach().numpy()

  sparsity_ratio = sparsity_ratio/total_items

  print('Sparsity ratio = {}'.format(sparsity_ratio))
  print('Num items analysed = {}'.format(total_items))

  return overall_latent_space, accumulated_latent_space, sample_latent_space
