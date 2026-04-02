from dataclasses import dataclass, field
from typing import Any

@dataclass
class RequiredPPE:
    helmet: bool = False
    vest: bool = False
    harness: bool = False
    gloves: bool = False
    boots: bool = False
    goggles: bool = False

@dataclass
class ZoneData:
    id: str
    site_id: str
    name: str
    type: str
    polygon_geojson: dict

class ZoneManager:
    def add_zones_direct(self, site_id: str, zones: list[ZoneData]) -> None:
        pass

    def get_required_ppe(self, zone: ZoneData) -> RequiredPPE:
        return RequiredPPE(helmet=True, vest=True, boots=True)
