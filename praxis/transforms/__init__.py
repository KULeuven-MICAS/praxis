from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Returns all available passes."""

    def get_apply_zigzag_schedule():
        from praxis.transforms import apply_zigzag_schedule

        return apply_zigzag_schedule.ApplyZigzagSchedule

    return {
        "apply-zigzag-schedule": get_apply_zigzag_schedule,
    }
