"""Blood report and biomarker repositories."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError
from app.models.report import Biomarker, BloodReport
from app.repositories.base import BaseRepository


class BloodReportRepository(BaseRepository[BloodReport]):
    model = BloodReport

    async def list_for_user_with_biomarkers(
        self, user_id: uuid.UUID, *, limit: int = 20
    ) -> list[BloodReport]:
        stmt = (
            select(BloodReport)
            .where(BloodReport.user_id == user_id)
            .options(selectinload(BloodReport.biomarkers))
            .order_by(BloodReport.created_at.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_detail(
        self, report_id: uuid.UUID, user_id: uuid.UUID
    ) -> BloodReport:
        stmt = (
            select(BloodReport)
            .where(BloodReport.id == report_id, BloodReport.user_id == user_id)
            .options(selectinload(BloodReport.biomarkers))
        )
        report = (await self.session.execute(stmt)).scalar_one_or_none()
        if report is None:
            raise NotFoundError(f"Report {report_id} was not found.")
        return report


class BiomarkerRepository(BaseRepository[Biomarker]):
    model = Biomarker

    async def latest_values(
        self, user_id: uuid.UUID, name: str, *, limit: int = 10
    ) -> list[Biomarker]:
        """Trend a single biomarker across every report the user uploaded."""
        stmt = (
            select(Biomarker)
            .join(BloodReport, BloodReport.id == Biomarker.report_id)
            .where(BloodReport.user_id == user_id, Biomarker.name == name)
            .order_by(BloodReport.created_at.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())
