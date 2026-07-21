"""DexHand device integrations."""

from prism.devices.hand.socket_client import (
	GESTURE_ID_TO_NAME,
	GESTURE_ID_TO_POSE,
	GESTURE_TABLE,
	MechHandClient,
)

__all__ = [
	'MechHandClient',
	'GESTURE_TABLE',
	'GESTURE_ID_TO_POSE',
	'GESTURE_ID_TO_NAME',
]
