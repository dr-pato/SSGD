import torch
from asteroid.losses import singlesrc_neg_sisdr, PITLossWrapper


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size, step,
        use_last_samples=True):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step  > 0:
            yield (i + 1) * step, data_length


def css_lookahead(mixture, model, n_src, window_size=2, stride=1, lookbehind=None,
                  lookahead=None, fs=16000, window_type="hamming",
                  reorder_func=singlesrc_neg_sisdr, pad_end=True):

    assert mixture.ndim == 3
    orig_length = mixture.shape[-1]  # original length
    window_size = int(window_size*fs)
    stride = int(stride*fs)

    if window_type is not None:
        if window_type.startswith("hamming"):
            window = torch.hamming_window(window_size).to(mixture)
        elif window_type.startswith("hanning"):
            window = torch.hann_window(window_size).to(mixture)
        else:
            raise NotImplementedError
        if window_type.endswith("squared"):
            window = window ** 2
    else:
        window = torch.ones((window_size)).to(mixture)

    if lookbehind is not None:
        lookbehind = int(lookbehind*fs)
    else:
        lookbehind = 0
    if lookahead is not None:
        lookahead = int(lookahead*fs)
    else:
        lookahead = 0

    if pad_end:
        n, r = divmod(orig_length, stride)
        if r != 0:
            n = n + 1
            pad_right = stride * n - (orig_length)

            noise = torch.randn((mixture.shape[0], mixture.shape[1], pad_right)).to(mixture)*1e-8
            mixture = torch.cat((mixture, noise), dim=-1)
    else:
        raise NotImplementedError

    n_frames = _count_frames(mixture.shape[-1], window_size, stride)
    outputs = []
    for f_indx, (f_start, f_stop) in enumerate(_gen_frame_indices(mixture.shape[-1], window_size, stride)):

        c_start = max(0, f_start - lookbehind)
        c_stop = min(f_stop + lookahead, mixture.shape[-1])
        c_frame = model(mixture[..., c_start:c_stop])
        c_frame = c_frame[..., (f_start - c_start):(c_frame.shape[-1]-lookahead)]
        if f_start == 0:
            pass
            # first segment append without anything
        else:
            c_frame = _reorder_sources(c_frame, outputs[-1], n_src, window_size, stride,
                                       reorder_func=reorder_func)
        outputs.append(c_frame)

    overlap = window_size - stride
    # second pass we apply a window
    first = outputs[0].clone()
    window = window.reshape(*[1 for x in range(first.ndim - 1)], -1)
    #outputs[0] = first

    result = [first[..., :stride]]  # will be discarded
    buff = first[..., stride:]*window[..., stride:]

    for i in range(1, len(outputs)-1):
        temp = outputs[i].clone()
        temp *= window
        result.append(temp[..., :stride] + buff[..., :stride])
        buff = temp[..., stride:] #torch.cat((temp[..., stride:-stride] + buff[..., stride:], temp[..., -stride:]), -1)

    last = outputs[-1]
    last[..., :stride] = last[..., :stride]*window[..., :stride] + buff[..., :stride]
    result.append(last)
    result = torch.cat(result, -1)

    return result[..., :orig_length]


def _reorder_sources(
    current: torch.FloatTensor,
    previous: torch.FloatTensor,
    n_src: int,
    window_size: int,
    hop_size: int,
    reorder_func=singlesrc_neg_sisdr
):
    """
     Reorder sources in current chunk to maximize correlation with previous chunk.
     Used for Continuous Source Separation. Standard dsp correlation is used
     for reordering.
    Args:
        current (:class:`torch.Tensor`): current chunk, tensor
                                        of shape (batch, n_src, window_size)
        previous (:class:`torch.Tensor`): previous chunk, tensor
                                        of shape (batch, n_src, window_size)
        n_src (:class:`int`): number of sources.
        window_size (:class:`int`): window_size, equal to last dimension of
                                    both current and previous.
        hop_size (:class:`int`): hop_size between current and previous tensors.
    """
    #batch, frames = current.size()
    #current = current.reshape(-1, n_src, frames)
    #previous = previous.reshape(-1, n_src, frames)

    bsz, n_src, frames = current.shape
    overlap_f = window_size - hop_size

    # We maximize correlation-like between previous and current.
    pw_losses = PITLossWrapper.get_pw_losses(reorder_func, current[..., :overlap_f], previous[..., -overlap_f:])
    best_perm = PITLossWrapper.find_best_perm(pw_losses)[-1]

    return PITLossWrapper.reorder_source(current, best_perm)