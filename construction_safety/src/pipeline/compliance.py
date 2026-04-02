from dataclasses import dataclass, field
from typing import Any

@dataclass
class Violation:
    severity: str
    description: str
    ppe_item: str = ''
    confidence: float = 1.0

@dataclass
class ComplianceResult:
    violations: list[Violation] = field(default_factory=list)

class ComplianceEngine:
    def evaluate(self, worker: Any, zone: Any, required_ppe: Any, site_id: str, frame_id: str) -> ComplianceResult:
        violations = []
        
        # Determine what's required (default defaults if no zone rules)
        req_helmet = required_ppe.helmet if required_ppe else True
        req_vest = required_ppe.vest if required_ppe else True
        req_boots = required_ppe.boots if required_ppe else True
        req_gloves = required_ppe.gloves if required_ppe else False
        req_harness = required_ppe.harness if required_ppe else False
        req_goggles = required_ppe.goggles if required_ppe else False

        if req_helmet and worker.helmet and worker.helmet.status == 'absent' and worker.helmet.confidence > 0.5:
            v = Violation(severity="high", description="Missing Helmet", confidence=worker.helmet.confidence, ppe_item='helmet')
            violations.append(v)
            worker.violations.append(v)
        if req_vest and worker.vest and worker.vest.status == 'absent' and worker.vest.confidence > 0.5:
            v = Violation(severity="high", description="Missing Vest", confidence=worker.vest.confidence, ppe_item='vest')
            violations.append(v)
            worker.violations.append(v)
        if req_boots and worker.boots and worker.boots.status == 'absent' and worker.boots.confidence > 0.5:      
            v = Violation(severity="medium", description="Missing Boots", confidence=worker.boots.confidence, ppe_item='boots')
            violations.append(v)
            worker.violations.append(v)
        if req_gloves and worker.gloves and worker.gloves.status == 'absent' and worker.gloves.confidence > 0.5:   
            v = Violation(severity="low", description="Missing Gloves", confidence=worker.gloves.confidence, ppe_item='gloves')
            violations.append(v)
            worker.violations.append(v)
            
        return ComplianceResult(violations=violations)
