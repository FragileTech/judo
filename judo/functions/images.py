import judo
from judo.typing import Tensor


AVAILABLE_FUNCTIONS = {"resize_image"}


def resize_image(frame: Tensor, width: int, height: int, mode: str = "RGB") -> Tensor:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    from PIL import Image

    frame = judo.to_numpy(frame)
    with judo.Backend.use_backend(name="numpy"):
        frame = Image.fromarray(frame)
        frame = judo.to_numpy(frame.convert(mode).resize(size=(width, height)))
    return judo.tensor(frame)
