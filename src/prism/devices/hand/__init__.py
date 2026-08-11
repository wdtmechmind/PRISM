"""DexHand device integrations."""

from prism.devices.hand.socket_client import (
	GESTURE_ID_TO_NAME,
	GESTURE_ID_TO_POSE,
	GESTURE_TABLE,
	MechHandClient,
)
from prism.devices.hand.v3_daemon_client import (
	MechHandV3CommandError,
	MechHandV3DaemonClient,
	MechHandV3ProtocolError,
	degrees_to_tenths,
)

__all__ = [
	'MechHandClient',
	'GESTURE_TABLE',
	'GESTURE_ID_TO_POSE',
	'GESTURE_ID_TO_NAME',
	'MechHandV3DaemonClient',
	'MechHandV3ProtocolError',
	'MechHandV3CommandError',
	'degrees_to_tenths',
]
