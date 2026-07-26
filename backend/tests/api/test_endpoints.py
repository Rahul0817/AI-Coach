"""API tests driving real HTTP requests through the assembled application."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

pytestmark = pytest.mark.api


class TestSystem:
    def test_liveness_touches_no_dependency(self, client) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_root_carries_the_medical_notice(self, client) -> None:
        body = client.get("/").json()
        assert "does not diagnose" in body["notice"]

    def test_openapi_schema_generates(self, client) -> None:
        """A schema that fails to generate means broken response models —
        worth catching in CI rather than when a client integrates."""
        spec = client.get("/openapi.json").json()
        assert len(spec["paths"]) > 50

    def test_security_headers_present(self, client) -> None:
        headers = client.get("/health/live").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "X-Request-ID" in headers


class TestAuthentication:
    def test_register_returns_tokens(self, client, user_payload) -> None:
        response = client.post("/api/v1/auth/register", json=user_payload)
        assert response.status_code == 201
        body = response.json()
        assert body["access_token"] and body["refresh_token"]
        assert body["token_type"] == "bearer"

    def test_weak_password_rejected(self, client) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "a@b.com", "password": "short", "full_name": "A B"},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_common_password_rejected(self, client) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "a@b.com", "password": "password123", "full_name": "A B"},
        )
        assert response.status_code == 422

    def test_duplicate_email_conflicts(self, client, user_payload) -> None:
        client.post("/api/v1/auth/register", json=user_payload)
        response = client.post("/api/v1/auth/register", json=user_payload)
        assert response.status_code == 409
        assert response.json()["error"] == "conflict"

    def test_login_succeeds(self, client, user_payload) -> None:
        client.post("/api/v1/auth/register", json=user_payload)
        response = client.post(
            "/api/v1/auth/login",
            json={"email": user_payload["email"], "password": user_payload["password"]},
        )
        assert response.status_code == 200

    def test_wrong_password_and_unknown_email_are_indistinguishable(
        self, client, user_payload
    ) -> None:
        """User enumeration guard: both paths must return the same error."""
        client.post("/api/v1/auth/register", json=user_payload)

        wrong_password = client.post(
            "/api/v1/auth/login",
            json={"email": user_payload["email"], "password": "Wrong!Password99"},
        )
        unknown_email = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "Wrong!Password99"},
        )

        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.json()["message"] == unknown_email.json()["message"]

    def test_protected_route_requires_a_token(self, client) -> None:
        response = client.get("/api/v1/users/profile")
        assert response.status_code == 401
        assert response.json()["error"] == "authentication_failed"

    def test_garbage_token_rejected(self, client) -> None:
        response = client.get(
            "/api/v1/users/profile", headers={"Authorization": "Bearer nonsense"}
        )
        assert response.status_code == 401

    def test_refresh_rotates_the_token(self, client, user_payload) -> None:
        """A refresh token must be single-use, so a stolen one is usable once."""
        registered = client.post("/api/v1/auth/register", json=user_payload).json()
        refresh = registered["refresh_token"]

        first = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert first.status_code == 200

        replay = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert replay.status_code == 401

    def test_access_token_rejected_at_the_refresh_endpoint(
        self, client, user_payload
    ) -> None:
        registered = client.post("/api/v1/auth/register", json=user_payload).json()
        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": registered["access_token"]}
        )
        assert response.status_code == 401

    def test_logout_revokes_the_token(self, client, auth_headers) -> None:
        assert client.post("/api/v1/auth/logout", headers=auth_headers).status_code == 200
        assert client.get("/api/v1/auth/me", headers=auth_headers).status_code == 401

    def test_account_deletion_requires_exact_confirmation(
        self, client, auth_headers, user_payload
    ) -> None:
        response = client.post(
            "/api/v1/auth/delete-account",
            headers=auth_headers,
            json={"password": user_payload["password"], "confirmation": "yes"},
        )
        assert response.status_code == 422


class TestProfile:
    def test_profile_returns_computed_values(self, client, auth_headers) -> None:
        response = client.patch(
            "/api/v1/users/profile",
            headers=auth_headers,
            json={"date_of_birth": "2002-05-14", "height_cm": 165, "weight_kg": 72.5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["bmi"] == 26.6
        assert body["bmi_category"] == "overweight"
        assert body["age"] is not None

    def test_out_of_range_values_rejected(self, client, auth_headers) -> None:
        response = client.patch(
            "/api/v1/users/profile", headers=auth_headers, json={"height_cm": 400}
        )
        assert response.status_code == 422

    def test_implausible_birth_date_rejected(self, client, auth_headers) -> None:
        response = client.patch(
            "/api/v1/users/profile",
            headers=auth_headers,
            json={"date_of_birth": "1850-01-01"},
        )
        assert response.status_code == 422

    def test_partial_update_preserves_other_fields(self, client, auth_headers) -> None:
        """PATCH semantics: setting height must not clear stored allergies."""
        client.patch(
            "/api/v1/users/profile",
            headers=auth_headers,
            json={"allergies": ["peanuts"], "height_cm": 165},
        )
        response = client.patch(
            "/api/v1/users/profile", headers=auth_headers, json={"weight_kg": 70}
        )
        assert response.json()["allergies"] == ["peanuts"]
        assert response.json()["height_cm"] == 165

    def test_export_is_a_downloadable_archive(self, client, auth_headers) -> None:
        response = client.get("/api/v1/users/export", headers=auth_headers)
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        payload = json.loads(response.content)
        for section in ("user", "profile", "cycles", "meals", "conversations"):
            assert section in payload


class TestChat:
    def test_agent_directory_is_public(self, client) -> None:
        response = client.get("/api/v1/chat/agents")
        assert response.status_code == 200
        assert len(response.json()) == 8

    def test_chat_routes_and_grounds(self, client, auth_headers) -> None:
        response = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "What should I eat for breakfast?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["routing"]["agent"] == "nutrition_coach"
        assert body["used_rag"] is True
        assert body["message"]["sources"]

    def test_every_response_carries_the_disclaimer(self, client, auth_headers) -> None:
        """Enforced in code, so it cannot be lost to prompt drift."""
        response = client.post(
            "/api/v1/chat", headers=auth_headers, json={"message": "Is PCOS curable?"}
        )
        content = response.json()["message"]["content"].lower()
        assert "consult a qualified healthcare professional" in content

    def test_crisis_message_bypasses_the_model(self, client, auth_headers) -> None:
        """The guardrail must fire before generation, not depend on the model
        choosing to behave."""
        response = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "I want to kill myself"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["message"]["model"] == "safety-guardrail"
        assert "988" in body["message"]["content"]

    def test_conversation_persists_and_is_titled(self, client, auth_headers) -> None:
        first = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "How much protein should I eat daily?"},
        ).json()
        conversation_id = first["conversation_id"]

        client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "And for dinner?", "conversation_id": conversation_id},
        )

        detail = client.get(
            f"/api/v1/chat/conversations/{conversation_id}", headers=auth_headers
        ).json()
        assert detail["message_count"] == 4  # two user + two assistant
        assert detail["title"] != "New conversation"

    def test_memory_persists_across_messages(self, client, auth_headers) -> None:
        """The headline requirement: a stated fact must not need repeating."""
        learned = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "I am 22 years old and allergic to peanuts"},
        ).json()
        assert "allergies" in learned["memory_updates"]

        profile = client.get("/api/v1/users/profile", headers=auth_headers).json()
        assert profile["allergies"] == ["peanuts"]

    def test_forced_agent_overrides_routing(self, client, auth_headers) -> None:
        response = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "What should I eat?", "agent": "fitness_coach"},
        )
        assert response.json()["routing"]["agent"] == "fitness_coach"

    def test_cannot_read_another_users_conversation(self, client, user_payload) -> None:
        owner = client.post("/api/v1/auth/register", json=user_payload).json()
        owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
        conversation_id = client.post(
            "/api/v1/chat", headers=owner_headers, json={"message": "hello"}
        ).json()["conversation_id"]

        intruder = client.post(
            "/api/v1/auth/register",
            json={
                "email": "intruder@example.com",
                "password": "Oviora!Secure24",
                "full_name": "In Truder",
            },
        ).json()
        intruder_headers = {"Authorization": f"Bearer {intruder['access_token']}"}

        response = client.get(
            f"/api/v1/chat/conversations/{conversation_id}", headers=intruder_headers
        )
        assert response.status_code == 404

    def test_empty_message_rejected(self, client, auth_headers) -> None:
        response = client.post(
            "/api/v1/chat", headers=auth_headers, json={"message": "   "}
        )
        assert response.status_code == 422


class TestPrediction:
    def test_high_and_low_risk_are_discriminated(
        self, client, auth_headers, sample_risk_payload, low_risk_payload
    ) -> None:
        high = client.post(
            "/api/v1/predict", headers=auth_headers, json=sample_risk_payload
        ).json()
        low = client.post(
            "/api/v1/predict", headers=auth_headers, json=low_risk_payload
        ).json()

        assert high["risk_score"] > low["risk_score"]
        assert high["risk_band"] == "high"
        assert low["risk_band"] == "low"

    def test_score_never_reaches_certainty(
        self, client, auth_headers, sample_risk_payload
    ) -> None:
        """A screening questionnaire reporting 100% would be indefensible."""
        body = client.post(
            "/api/v1/predict", headers=auth_headers, json=sample_risk_payload
        ).json()
        assert 0.02 <= body["risk_score"] <= 0.97

    def test_explanation_accompanies_every_score(
        self, client, auth_headers, sample_risk_payload
    ) -> None:
        body = client.post(
            "/api/v1/predict", headers=auth_headers, json=sample_risk_payload
        ).json()
        assert len(body["top_factors"]) == 5
        for factor in body["top_factors"]:
            assert factor["direction"] in {"increases", "decreases"}
            assert factor["explanation"]
        assert body["recommendations"]
        assert "healthcare professional" in body["disclaimer"]

    def test_out_of_range_input_rejected(
        self, client, auth_headers, sample_risk_payload
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            headers=auth_headers,
            json={**sample_risk_payload, "bmi": 200},
        )
        assert response.status_code == 422

    def test_model_provenance_is_exposed(self, client, auth_headers) -> None:
        """A prediction a user cannot interrogate is one they cannot trust."""
        body = client.get("/api/v1/predict/model-info", headers=auth_headers).json()
        assert body["metrics"]["roc_auc"] > 0.8
        assert len(body["features"]) == 19
        assert len(body["all_model_scores"]) == 3

    def test_history_records_assessments(
        self, client, auth_headers, sample_risk_payload
    ) -> None:
        client.post("/api/v1/predict", headers=auth_headers, json=sample_risk_payload)
        history = client.get("/api/v1/predict/history", headers=auth_headers).json()
        assert len(history) == 1


class TestTracking:
    def test_water_upsert_is_idempotent(self, client, auth_headers) -> None:
        client.put("/api/v1/water", headers=auth_headers, json={"millilitres": 1000})
        response = client.put(
            "/api/v1/water", headers=auth_headers, json={"millilitres": 1800}
        )
        assert response.json()["millilitres"] == 1800
        assert len(client.get("/api/v1/water", headers=auth_headers).json()) == 1

    def test_future_dates_rejected(self, client, auth_headers) -> None:
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        response = client.put(
            "/api/v1/water",
            headers=auth_headers,
            json={"millilitres": 500, "logged_on": tomorrow},
        )
        assert response.status_code == 422

    def test_workout_calories_estimated_when_absent(self, client, auth_headers) -> None:
        response = client.post(
            "/api/v1/workouts",
            headers=auth_headers,
            json={
                "title": "Strength",
                "workout_type": "strength",
                "duration_minutes": 45,
            },
        )
        assert response.json()["calories_burned"] > 0

    def test_habit_streak_tracks_check_ins(self, client, auth_headers) -> None:
        habit_id = client.post(
            "/api/v1/habits", headers=auth_headers, json={"name": "Morning walk"}
        ).json()["id"]

        for offset in range(3):
            day = (date.today() - timedelta(days=offset)).isoformat()
            client.post(
                f"/api/v1/habits/{habit_id}/check-in",
                headers=auth_headers,
                json={"logged_on": day, "completed_count": 1},
            )

        habits = client.get("/api/v1/habits", headers=auth_headers).json()
        assert habits[0]["current_streak"] == 3
        assert habits[0]["completed_today"] is True

    def test_duplicate_habit_name_conflicts(self, client, auth_headers) -> None:
        client.post("/api/v1/habits", headers=auth_headers, json={"name": "Walk"})
        response = client.post(
            "/api/v1/habits", headers=auth_headers, json={"name": "Walk"}
        )
        assert response.status_code == 409

    def test_nutrition_summary_aggregates(self, client, auth_headers) -> None:
        for calories, protein in [(300, 20), (500, 30)]:
            client.post(
                "/api/v1/meals",
                headers=auth_headers,
                json={
                    "name": "Meal",
                    "meal_type": "lunch",
                    "calories": calories,
                    "protein_g": protein,
                    "carbs_g": 40,
                    "fat_g": 10,
                    "fibre_g": 5,
                },
            )
        summary = client.get("/api/v1/meals/summary", headers=auth_headers).json()
        assert summary["total_calories"] == 800
        assert summary["total_protein_g"] == 50
        assert summary["meal_count"] == 2

    def test_cycle_logging_backfills_previous_length(self, client, auth_headers) -> None:
        """Cycle length is only knowable once the next cycle starts."""
        client.post(
            "/api/v1/cycles",
            headers=auth_headers,
            json={"start_date": "2026-01-01", "end_date": "2026-01-05"},
        )
        client.post(
            "/api/v1/cycles",
            headers=auth_headers,
            json={"start_date": "2026-02-10", "end_date": "2026-02-14"},
        )
        cycles = client.get("/api/v1/cycles", headers=auth_headers).json()
        earlier = next(c for c in cycles if c["start_date"] == "2026-01-01")
        assert earlier["cycle_length_days"] == 40
        assert earlier["is_irregular"] is True

    def test_duplicate_cycle_start_conflicts(self, client, auth_headers) -> None:
        payload = {"start_date": "2026-01-01"}
        client.post("/api/v1/cycles", headers=auth_headers, json=payload)
        response = client.post("/api/v1/cycles", headers=auth_headers, json=payload)
        assert response.status_code == 409


class TestPlanGeneration:
    def test_diet_plan_respects_allergies(self, client, auth_headers) -> None:
        """Allergy exclusion must hold with certainty, which is the whole
        reason this is generated in code rather than by a language model."""
        response = client.post(
            "/api/v1/plans/diet",
            headers=auth_headers,
            json={"days": 3, "allergies": ["peanut", "almond"], "target_calories": 1800},
        )
        assert response.status_code == 200
        body = response.json()
        text = json.dumps(body).lower()
        assert "peanut" not in text
        assert "almond" not in text

    def test_vegan_plan_excludes_all_animal_products(self, client, auth_headers) -> None:
        body = client.post(
            "/api/v1/plans/diet",
            headers=auth_headers,
            json={"days": 2, "dietary_preference": "vegan"},
        ).json()
        text = json.dumps(body).lower()
        for animal in ("chicken", "fish", "salmon", "egg", "paneer", "curd", "yoghurt"):
            assert animal not in text, f"vegan plan contained {animal}"

    def test_diet_plan_lands_near_the_calorie_target(self, client, auth_headers) -> None:
        """A plan whose numbers do not add up is worse than no plan."""
        body = client.post(
            "/api/v1/plans/diet",
            headers=auth_headers,
            json={"days": 1, "target_calories": 1800},
        ).json()
        total = body["days"][0]["total_calories"]
        assert 1400 <= total <= 2300, f"plan totalled {total} kcal against 1800"

    def test_plan_generation_is_deterministic(self, client, auth_headers) -> None:
        request = {"days": 2, "target_calories": 1800}
        first = client.post(
            "/api/v1/plans/diet", headers=auth_headers, json=request
        ).json()
        second = client.post(
            "/api/v1/plans/diet", headers=auth_headers, json=request
        ).json()
        assert first == second

    def test_workout_plan_prioritises_strength(self, client, auth_headers) -> None:
        body = client.post(
            "/api/v1/plans/workout", headers=auth_headers, json={"days_per_week": 4}
        ).json()
        assert len(body["weekly_schedule"]) == 4
        strength = [s for s in body["weekly_schedule"] if s["workout_type"] == "strength"]
        assert len(strength) >= 2


class TestDashboard:
    def test_dashboard_serves_everything_in_one_request(
        self, client, auth_headers
    ) -> None:
        client.put("/api/v1/weight", headers=auth_headers, json={"weight_kg": 70})
        client.put("/api/v1/sleep", headers=auth_headers, json={"hours": 7.5})

        body = client.get("/api/v1/analytics/dashboard", headers=auth_headers).json()
        for key in (
            "metrics",
            "weight_trend",
            "sleep_trend",
            "calorie_trend",
            "water_trend",
            "mood_trend",
            "workout_minutes_trend",
            "macro_split",
            "habit_completion",
            "cycle_summary",
            "insights",
        ):
            assert key in body
        assert len(body["metrics"]) == 8
        assert body["insights"]

    def test_untracked_days_are_null_not_zero(self, client, auth_headers) -> None:
        """A day with no weigh-in is unknown, not 0 kg. Zero-filling would draw
        a cliff on the chart and poison the average."""
        client.put("/api/v1/weight", headers=auth_headers, json={"weight_kg": 70})
        body = client.get(
            "/api/v1/analytics/dashboard?days=14", headers=auth_headers
        ).json()
        points = body["weight_trend"]["points"]
        assert len(points) == 14
        assert any(p["value"] is None for p in points)

    def test_new_account_gets_an_onboarding_insight(self, client, auth_headers) -> None:
        body = client.get("/api/v1/analytics/dashboard", headers=auth_headers).json()
        assert body["insights"][0]["category"] == "onboarding"

    def test_weekly_and_monthly_rollups(self, client, auth_headers) -> None:
        assert (
            client.get("/api/v1/analytics/weekly", headers=auth_headers).status_code
            == 200
        )
        monthly = client.get("/api/v1/analytics/monthly", headers=auth_headers).json()
        assert 0 <= monthly["consistency_score"] <= 100


class TestNotifications:
    def test_welcome_notification_created_on_registration(
        self, client, auth_headers
    ) -> None:
        rows = client.get("/api/v1/notifications", headers=auth_headers).json()
        assert any("Welcome to Oviora" in row["title"] for row in rows)

    def test_counts_and_mark_all_read(self, client, auth_headers) -> None:
        before = client.get("/api/v1/notifications/counts", headers=auth_headers).json()
        assert before["unread"] >= 1

        client.post("/api/v1/notifications/read-all", headers=auth_headers)
        after = client.get("/api/v1/notifications/counts", headers=auth_headers).json()
        assert after["unread"] == 0


class TestErrorContract:
    def test_every_error_has_the_same_shape(self, client) -> None:
        """One error shape across the API is what lets a client write a single
        handler instead of one per endpoint."""
        response = client.get("/api/v1/users/profile")
        body = response.json()
        assert set(body) >= {"error", "message"}
        assert "request_id" in body

    def test_unknown_route_returns_the_standard_shape(self, client) -> None:
        body = client.get("/api/v1/does-not-exist").json()
        assert body["error"] == "http_404"

    def test_validation_errors_name_the_field(self, client) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "x", "full_name": "A"},
        )
        assert response.status_code == 422
        assert "email" in response.json()["details"]
