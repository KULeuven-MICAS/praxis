from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Returns all available passes."""

    def get_linalg_to_stream():
        from naxirzag.transforms import linalg_to_stream

        return linalg_to_stream.LinalgToStream

    return {
        "linalg-to-stream": get_linalg_to_stream,
    }
