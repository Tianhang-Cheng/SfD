import os
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def prune_old_checkpoints(directory, keep_last=3):
    ''' Keep only the `keep_last` most recent numbered checkpoints (by iter number) in `directory`, deleting older ones. "latest.pth" is left untouched. '''
    ckpts = []
    for fname in os.listdir(directory):
        if fname == "latest.pth" or not fname.endswith(".pth"):
            continue
        ckpts.append((int(fname[:-len(".pth")]), fname))

    ckpts.sort(key=lambda x: x[0])
    for _, fname in ckpts[:-keep_last]:
        os.remove(os.path.join(directory, fname))


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1).reshape(batch_size * total_pixels, -1)

    return model_outputs