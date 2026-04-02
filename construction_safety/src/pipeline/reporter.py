from dataclasses import dataclass, field
from typing import Any

@dataclass
class ViolationReport:
    overall_status: str
    worker_count: int
    violation_count: int
    summary_text: str
    workers: list[Any] = field(default_factory=list)

    def to_dict(self):
        return {
            "overall_status": self.overall_status,
            "worker_count": self.worker_count,
            "violation_count": self.violation_count,
            "summary_text": self.summary_text
        }

def generate_report(result: Any, all_violations: list[Any]) -> ViolationReport:
    worker_count = len(result.workers)
    violation_count = len(all_violations)
    
    if violation_count == 0:
        overall_status = "SAFE"
    elif violation_count < 3:
        overall_status = "WARNING"
    else:
        overall_status = "UNSAFE"
        
    return ViolationReport(
        overall_status=overall_status,
        worker_count=worker_count,
        violation_count=violation_count,
        summary_text=f"Found {violation_count} violations among {worker_count} workers.",
        workers=result.workers
    )