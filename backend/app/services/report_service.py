"""Blood-report upload, OCR processing and AI explanation."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.base import build_agent
from app.ai.llm.factory import get_llm_provider
from app.ai.multimodal.ocr import SUPPORTED_TYPES, process_report
from app.ai.rag.retriever import get_retriever
from app.core.logging import get_logger
from app.models.enums import AgentName, ReportStatus
from app.models.report import BloodReport
from app.repositories.report import BiomarkerRepository, BloodReportRepository
from app.schemas.common import MEDICAL_DISCLAIMER
from app.services.storage_service import get_storage

logger = get_logger(__name__)


class ReportService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.reports = BloodReportRepository(session)
        self.biomarkers = BiomarkerRepository(session)
        self.storage = get_storage()

    async def upload_and_process(
        self, user_id: uuid.UUID, data: bytes, filename: str, content_type: str
    ) -> BloodReport:
        """Store the file, extract biomarkers and persist the result.

        Processing runs inline rather than on a queue. At the observed cost — a
        digital PDF parses in well under a second — a background job would add
        infrastructure and a polling UI for no user-visible benefit. If OCR of
        large scanned documents becomes common, this is the natural seam to move
        behind a worker.
        """
        self.storage.validate(data, content_type, SUPPORTED_TYPES)

        report = await self.reports.create(
            user_id=user_id,
            filename=filename[:255],
            content_type=content_type,
            status=ReportStatus.PROCESSING.value,
        )
        await self.session.flush()

        try:
            stored = await self.storage.upload(data, content_type, "reports", user_id)
            report.file_url = stored.url
        except Exception as exc:
            # A storage failure must not lose the parsed values — the extracted
            # biomarkers are the thing the user actually came for.
            logger.warning("report file storage failed", extra={"error": str(exc)})

        try:
            result = process_report(data, content_type)
        except Exception as exc:
            report.status = ReportStatus.FAILED.value
            report.error_message = str(exc)[:500]
            await self.session.commit()
            logger.error("report processing failed", extra={"error": str(exc)})
            return report

        report.extracted_text = result.text[:50_000] if result.text else None
        report.ocr_confidence = result.confidence

        for parsed in result.biomarkers:
            await self.biomarkers.create(
                report_id=report.id,
                name=parsed.key,
                display_name=parsed.display_name,
                value=parsed.value,
                unit=parsed.unit,
                reference_low=parsed.reference_low,
                reference_high=parsed.reference_high,
                flag=parsed.flag.value,
                interpretation=parsed.interpretation,
            )

        report.summary = self._summarise(result)
        report.status = (
            ReportStatus.COMPLETED.value
            if result.biomarkers or result.text
            else ReportStatus.FAILED.value
        )
        if not result.biomarkers and not result.text:
            report.error_message = "No readable text was found in this file."

        await self.session.commit()
        logger.info(
            "report processed",
            extra={"report_id": str(report.id), "biomarkers": len(result.biomarkers)},
        )
        return await self.reports.get_detail(report.id, user_id)

    @staticmethod
    def _summarise(result) -> str:  # type: ignore[no-untyped-def]
        """Deterministic overview written from the parsed values only."""
        if not result.biomarkers:
            return (
                "No recognised biomarkers were found in this report. You can "
                "enter values manually, or upload a clearer copy."
            )

        out_of_range = [b for b in result.biomarkers if b.flag.value in {"high", "low"}]
        lines = [f"Read {len(result.biomarkers)} biomarker(s) from this report."]
        if out_of_range:
            names = ", ".join(f"{b.display_name} ({b.flag.value})" for b in out_of_range)
            lines.append(
                f"{len(out_of_range)} value(s) fall outside the reference range "
                f"printed on the report: {names}."
            )
        else:
            lines.append(
                "All recognised values fall within the reference ranges printed "
                "on the report."
            )
        lines.extend(result.warnings)
        lines.append(
            "Take this report to your doctor — they interpret these values "
            "alongside your symptoms and history, which is something no app can do."
        )
        return " ".join(lines)

    async def explain(self, user_id: uuid.UUID, report_id: uuid.UUID) -> dict:
        """Run the Blood Report Analyzer over a processed report."""
        report = await self.reports.get_detail(report_id, user_id)
        if not report.biomarkers:
            return {
                "ai_explanation": (
                    "There are no parsed biomarkers to explain for this report."
                ),
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        table = "\n".join(
            f"- {b.display_name}: {b.value} {b.unit} "
            f"(reference {b.reference_low}–{b.reference_high}, flagged {b.flag})"
            for b in report.biomarkers
        )
        query = (
            "Explain these lab results in plain language. For each marker say "
            "what it measures and why a clinician would have ordered it, and "
            "note which ones relate to PCOS assessment. Do not tell me whether "
            "I have any condition. Finish with three specific questions I "
            "should ask my doctor.\n\n"
            f"My results:\n{table}"
        )

        agent = build_agent(
            AgentName.BLOOD_REPORT_ANALYZER, get_llm_provider(), get_retriever()
        )
        response = await agent.answer(query)
        return {
            "ai_explanation": response.content,
            "sources": [c.to_citation() for c in response.sources],
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    async def list_reports(self, user_id: uuid.UUID, limit: int = 20):
        return await self.reports.list_for_user_with_biomarkers(user_id, limit=limit)

    async def get_report(self, user_id: uuid.UUID, report_id: uuid.UUID) -> BloodReport:
        return await self.reports.get_detail(report_id, user_id)

    async def delete_report(self, user_id: uuid.UUID, report_id: uuid.UUID) -> None:
        await self.reports.delete_for_user(report_id, user_id)
        await self.session.commit()

    async def biomarker_trend(self, user_id: uuid.UUID, name: str) -> list[dict]:
        """Track one marker across every report the user has uploaded."""
        rows = await self.biomarkers.latest_values(user_id, name)
        return [
            {
                "value": row.value,
                "unit": row.unit,
                "flag": row.flag,
                "recorded_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
