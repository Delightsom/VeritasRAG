from pydantic import BaseModel, Field
from typing import List, Literal

class ClaimVerdict(BaseModel):
    claim: str = Field(description="The atomic claim extracted from the summary.")
    citation: str = Field(description="The source chunk ID cited for this claim.")
    verdict: Literal["Grounded", "Unsupported", "Contradicted"] = Field(description="Evaluation verdict.")
    reasoning: str = Field(description="Brief justification for the verdict.")

class AuditReport(BaseModel):
    claims: List[ClaimVerdict]
    overall_faithfulness_score: float = Field(description="Score between 0.0 and 1.0 representing overall faithfulness.")

class Auditor:
    def evaluate_claims(self, claims: List[dict], retrieved_chunks: dict) -> AuditReport:
        # Guardrail logic to ensure grounded citations match retrieved chunk IDs
        valid_ids = set(retrieved_chunks.get('ids', [[]])[0])
        
        evaluated_claims = []
        scores = []

        for item in claims:
            verdict = item.get("verdict")
            citation = item.get("citation")

            # Guardrail check: Reject Grounded status if citation is not in retrieved labels
            if verdict == "Grounded" and citation not in valid_ids:
                verdict = "Unsupported"

            if verdict == "Grounded":
                score = 1.0
            elif verdict == "Unsupported":
                score = 0.5
            else:
                score = 0.0

            scores.append(score)
            evaluated_claims.append(ClaimVerdict(
                claim=item.get("claim"),
                citation=citation,
                verdict=verdict,
                reasoning=item.get("reasoning", "")
            ))

        overall_score = sum(scores) / len(scores) if scores else 0.0

        return AuditReport(
            claims=evaluated_claims,
            overall_faithfulness_score=round(overall_score, 2)
        )