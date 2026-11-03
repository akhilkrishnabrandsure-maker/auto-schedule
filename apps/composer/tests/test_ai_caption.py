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
