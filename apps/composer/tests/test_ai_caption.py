"""Tests for the composer's AI caption endpoint (Claude call mocked)."""

import json
from unittest import mock

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import User
from apps.members.models import OrgMembership, WorkspaceMembership
from apps.organizations.models import Organization
from apps.social_accounts.models import SocialAccount
from apps.workspaces.models import Workspace


@override_settings(ANTHROPIC_API_KEY="test-key")
class AICaptionViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com",
            password="testpass123",
            tos_accepted_at=timezone.now(),
        )
        self.org = Organization.objects.create(name="Test Org")
        self.workspace = Workspace.objects.create(organization=self.org, name="Test Workspace")
        OrgMembership.objects.create(user=self.user, organization=self.org, org_role=OrgMembership.OrgRole.OWNER)
        WorkspaceMembership.objects.create(
            user=self.user,
            workspace=self.workspace,
            workspace_role=WorkspaceMembership.WorkspaceRole.OWNER,
        )
        self.client.force_login(self.user)
        self.youtube = SocialAccount.objects.create(
            workspace=self.workspace,
            platform="youtube",
            account_platform_id="yt-1",
            account_name="YT Channel",
            connection_status=SocialAccount.ConnectionStatus.CONNECTED,
        )
        self.url = reverse("composer:ai_caption", kwargs={"workspace_id": self.workspace.id})

    def _post(self, payload):
        return self.client.post(self.url, data=json.dumps(payload), content_type="application/json")

    def test_returns_generated_caption_and_passes_platforms(self):
        with mock.patch(
            "apps.composer.ai_caption.generate_caption",
            return_value={"title": "Diwali Sale", "caption": "20% off all plans! #diwali"},
        ) as gen:
            resp = self._post({"brief": "Diwali offer", "tone": "bold", "account_ids": [str(self.youtube.id)]})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"title": "Diwali Sale", "caption": "20% off all plans! #diwali"})
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["brief"], "Diwali offer")
        self.assertEqual(kwargs["tone"], "bold")
        self.assertEqual(kwargs["platforms"][0]["platform"], "youtube")

    def test_requires_brief_or_caption(self):
        resp = self._post({"brief": "  ", "caption": ""})
        self.assertEqual(resp.status_code, 400)

    def test_error_message_is_returned(self):
        from apps.composer import ai_caption

        with mock.patch(
            "apps.composer.ai_caption.generate_caption",
            side_effect=ai_caption.AICaptionError("Claude is busy right now. Try again in a minute."),
        ):
            resp = self._post({"brief": "Diwali offer"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("busy", resp.json()["error"])

    def test_compose_page_shows_button(self):
        resp = self.client.get(reverse("composer:compose", kwargs={"workspace_id": self.workspace.id}))
        self.assertContains(resp, "Write with AI")

    @override_settings(ANTHROPIC_API_KEY="")
    def test_compose_page_hides_button_without_key(self):
        resp = self.client.get(reverse("composer:compose", kwargs={"workspace_id": self.workspace.id}))
        self.assertNotContains(resp, "Write with AI")

    def test_ignores_accounts_from_other_workspaces(self):
        other_ws = Workspace.objects.create(organization=self.org, name="Other")
        other = SocialAccount.objects.create(
            workspace=other_ws,
            platform="tiktok",
            account_platform_id="tt-9",
            account_name="other",
            connection_status=SocialAccount.ConnectionStatus.CONNECTED,
        )
        with mock.patch(
            "apps.composer.ai_caption.generate_caption", return_value={"title": "", "caption": "x"}
        ) as gen:
            self._post({"brief": "hi", "account_ids": [str(other.id)]})
        self.assertEqual(gen.call_args.kwargs["platforms"], [])


@override_settings(GEMINI_API_KEY="gem-key", GEMINI_MODEL="gemini-test", ANTHROPIC_API_KEY="")
class GeminiCaptionTests(TestCase):
    def _call(self):
        from apps.composer import ai_caption

        return ai_caption.generate_caption(
            brief="Halloween offer",
            current_caption="",
            title="",
            platforms=[{"platform": "youtube", "limit": 5000}],
            tone="playful",
        )

    def _response(self, status, payload):
        resp = mock.Mock(status_code=status, text=json.dumps(payload))
        resp.json.return_value = payload
        return resp

    def test_gemini_reply_is_parsed(self):
        reply = {"candidates": [{"content": {"parts": [{"text": '{"title": "Boo!", "caption": "Spooky deals #halloween"}'}]}}]}
        with mock.patch("apps.composer.ai_caption.httpx.post", return_value=self._response(200, reply)) as post:
            result = self._call()

        self.assertEqual(result, {"title": "Boo!", "caption": "Spooky deals #halloween"})
        args, kwargs = post.call_args
        self.assertIn("models/gemini-test:generateContent", args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "gem-key")
        self.assertIn("Halloween offer", kwargs["json"]["contents"][0]["parts"][0]["text"])

    def test_code_fenced_json_is_accepted(self):
        reply = {"candidates": [{"content": {"parts": [{"text": '```json\n{"title": "", "caption": "Hi"}\n```'}]}}]}
        with mock.patch("apps.composer.ai_caption.httpx.post", return_value=self._response(200, reply)):
            self.assertEqual(self._call()["caption"], "Hi")

    def test_bad_key_gives_clear_error(self):
        from apps.composer import ai_caption

        err = {"error": {"message": "API key not valid. Please pass a valid API key."}}
        with mock.patch("apps.composer.ai_caption.httpx.post", return_value=self._response(400, err)):
            with self.assertRaisesMessage(ai_caption.AICaptionError, "GEMINI_API_KEY"):
                self._call()

    def test_blocked_prompt_gives_clear_error(self):
        from apps.composer import ai_caption

        with mock.patch(
            "apps.composer.ai_caption.httpx.post",
            return_value=self._response(200, {"promptFeedback": {"blockReason": "SAFETY"}}),
        ):
            with self.assertRaisesMessage(ai_caption.AICaptionError, "declined"):
                self._call()
