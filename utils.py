import torch.nn as nn
import torch
import traceback
from functools import wraps

def load_ckpt(model, checkpoint_path):
    fake_task = nn.Module()
    fake_task.model = model
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    fake_task.load_state_dict(state_dict, strict=True)
    print(f"| load ckpt from {checkpoint_path}.")


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
                if (
                        value is not None and
                        not isinstance(value, list) and
                        fn.__name__ in ['test_dataloader', 'val_dataloader']
                ):
                    value = [value]
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + \
                    str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader
