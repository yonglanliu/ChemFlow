"""Sign in with ChatGPT for the local ChemFlow desktop application."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests


ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
KEYRING_SERVICE = "ChemFlow Sign in with ChatGPT"


@dataclass(frozen=True)
class ChatGPTSession:
    email: str
    access_token: str
    models: tuple[tuple[str, str], ...] = ()


class _KeyringStore:
    def get(self, name: str) -> str | None:
        try:
            import keyring
        except ImportError as error:
            raise RuntimeError(
                "Secure account storage is unavailable. Reinstall ChemFlow with "
                "the desktop extra: pip install -e '.[desktop]'"
            ) from error
        return keyring.get_password(KEYRING_SERVICE, name)

    def set(self, name: str, value: str) -> None:
        import keyring

        keyring.set_password(KEYRING_SERVICE, name, value)


class ChatGPTAuth:
    """Manage one locally authorized ChatGPT account in the OS keychain."""

    def __init__(self, *, store: Any | None = None, http: Any = requests) -> None:
        self.store = store or _KeyringStore()
        self.http = http
        self._refresh_lock = threading.Lock()

    def profile(self) -> dict[str, Any]:
        raw = self.store.get("profile")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def is_connected(self) -> bool:
        profile = self.profile()
        return bool(profile.get("access_token") and profile.get("refresh_token"))

    def account_label(self) -> str:
        profile = self.profile()
        return str(profile.get("email") or "ChatGPT account")

    def cached_models(self) -> tuple[tuple[str, str], ...]:
        models = self.profile().get("models", [])
        return tuple(
            (str(item[0]), str(item[1]))
            for item in models
            if isinstance(item, list) and len(item) == 2
        )

    def sign_in(
        self,
        *,
        timeout: float = 300.0,
        cancel_event: threading.Event | None = None,
    ) -> ChatGPTSession:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("ChatGPT sign-in was cancelled.")
        existing = self.profile()
        host_id = self._host_id()
        requested_client_id = str(existing.get("client_id") or "dynamic_agent_client")
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = self._pkce_challenge(verifier)
        callback: dict[str, str] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(handler) -> None:  # noqa: N802
                parsed = urlparse(handler.path)
                if parsed.path != "/auth/callback":
                    handler.send_error(404)
                    return
                values = parse_qs(parsed.query)
                callback.update({key: items[0] for key, items in values.items() if items})
                body = (
                    "<html><body><h2>Authorization received.</h2>"
                    "<p>Return to ChemFlow while it verifies the connection.</p>"
                    "</body></html>"
                ).encode("utf-8")
                handler.send_response(200)
                handler.send_header("Content-Type", "text/html; charset=utf-8")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), CallbackHandler)
        server.timeout = 0.25
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        parameters = {
            "client_id": requested_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "ext_agent_host_id": host_id,
        }
        if requested_client_id == "dynamic_agent_client":
            parameters["agent_name_hint"] = "ChemFlow"
        else:
            if existing.get("id_token"):
                parameters["id_token_hint"] = str(existing["id_token"])
            if existing.get("email"):
                parameters["login_hint"] = str(existing["email"])

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("ChatGPT sign-in was cancelled.")
            if not webbrowser.open(f"{AUTHORIZE_URL}?{urlencode(parameters)}"):
                raise RuntimeError("Could not open the system browser for ChatGPT sign-in.")
            deadline = time.monotonic() + timeout
            while not callback and time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("ChatGPT sign-in was cancelled.")
                server.handle_request()
        finally:
            server.server_close()

        if not callback:
            raise TimeoutError("ChatGPT sign-in timed out.")
        if not secrets.compare_digest(callback.get("state", ""), state):
            raise RuntimeError("ChatGPT sign-in returned an invalid state value.")
        if callback.get("error"):
            raise RuntimeError(
                f"ChatGPT sign-in was not completed: {callback['error']}."
            )
        code = callback.get("code", "")
        if not code:
            raise RuntimeError("ChatGPT sign-in returned no authorization code.")
        issued_client_id = callback.get("client_id", requested_client_id)
        if requested_client_id == "dynamic_agent_client":
            if not issued_client_id or issued_client_id == "dynamic_agent_client":
                raise RuntimeError("ChatGPT did not issue a client registration.")
        elif issued_client_id != requested_client_id:
            raise RuntimeError("ChatGPT returned a different client registration.")

        tokens = self._token_request(
            {
                "grant_type": "authorization_code",
                "client_id": issued_client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            }
        )
        claims = self._validate_id_token(
            str(tokens.get("id_token", "")), issued_client_id, nonce
        )
        if existing.get("subject") and existing.get("client_id") == issued_client_id:
            if claims["sub"] != existing["subject"]:
                raise RuntimeError("The returned ChatGPT account did not match this profile.")
        scopes = set(str(tokens.get("scope", "")).split())
        if "chatgpt.tokens.use.direct" not in scopes:
            raise RuntimeError("ChatGPT plan usage was not authorized for ChemFlow.")
        profile = {
            "email": str(claims.get("email") or claims.get("name") or "ChatGPT account"),
            "issuer": ISSUER,
            "subject": claims["sub"],
            "client_id": issued_client_id,
            "ext_agent_host_id": host_id,
            "id_token": tokens["id_token"],
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": tokens.get("token_type", "Bearer"),
            "expires_in": int(tokens.get("expires_in", 3600)),
            "scopes": sorted(scopes),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_profile(profile)
        try:
            models = self.list_models(profile["access_token"])
        except (requests.RequestException, RuntimeError, ValueError):
            models = ()
        profile["models"] = [list(model) for model in models]
        self._save_profile(profile)
        return ChatGPTSession(profile["email"], profile["access_token"], models)

    def access_token(self) -> str:
        with self._refresh_lock:
            profile = self.profile()
            if not profile.get("access_token") or not profile.get("refresh_token"):
                raise RuntimeError("Connect a ChatGPT account before running the agent.")
            saved_at = datetime.fromisoformat(str(profile["saved_at"]))
            expires_at = saved_at + timedelta(seconds=int(profile.get("expires_in", 3600)))
            if datetime.now(timezone.utc) < expires_at - timedelta(minutes=5):
                return str(profile["access_token"])
            tokens = self._token_request(
                {
                    "grant_type": "refresh_token",
                    "client_id": profile["client_id"],
                    "refresh_token": profile["refresh_token"],
                    "resource": RESOURCE,
                }
            )
            profile.update(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_in=int(tokens.get("expires_in", 3600)),
                saved_at=datetime.now(timezone.utc).isoformat(),
            )
            if tokens.get("id_token"):
                profile["id_token"] = tokens["id_token"]
            if tokens.get("scope"):
                profile["scopes"] = sorted(str(tokens["scope"]).split())
            self._save_profile(profile)
            return str(profile["access_token"])

    def list_models(self, access_token: str | None = None) -> tuple[tuple[str, str], ...]:
        token = access_token or self.access_token()
        response = self.http.get(
            f"{RESOURCE}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Could not load ChatGPT models (HTTP {response.status_code}).")
        rows = response.json().get("models", [])
        return tuple(
            (str(row["slug"]), str(row.get("display_name") or row["slug"]))
            for row in rows
            if row.get("visibility") == "list" and row.get("slug")
        )

    def disconnect(self) -> None:
        profile = self.profile()
        refresh_token = str(profile.get("refresh_token") or "")
        if refresh_token and profile.get("client_id"):
            try:
                discovery = self.http.get(
                    f"{ISSUER}/.well-known/openid-configuration", timeout=30
                )
                revocation_endpoint = discovery.json().get("revocation_endpoint", "")
                if not revocation_endpoint:
                    raise RuntimeError("OpenAI did not publish a revocation endpoint.")
                self.http.post(
                    revocation_endpoint,
                    data={
                        "token": refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": profile["client_id"],
                    },
                    timeout=30,
                )
            except (requests.RequestException, RuntimeError, ValueError):
                pass
        for key in ("access_token", "refresh_token", "id_token"):
            profile.pop(key, None)
        if profile:
            self._save_profile(profile)

    def _host_id(self) -> str:
        value = self.store.get("host_id")
        if value:
            return value
        value = f"urn:uuid:{uuid.uuid4()}"
        self.store.set("host_id", value)
        return value

    def _save_profile(self, profile: dict[str, Any]) -> None:
        self.store.set("profile", json.dumps(profile, separators=(",", ":")))

    def _token_request(self, data: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(TOKEN_URL, data=data, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"ChatGPT token request failed (HTTP {response.status_code}).")
        payload = response.json()
        if not payload.get("access_token") or not payload.get("refresh_token"):
            raise RuntimeError("ChatGPT returned incomplete authorization credentials.")
        return payload

    @staticmethod
    def _validate_id_token(token: str, client_id: str, nonce: str) -> dict[str, Any]:
        if not token:
            raise RuntimeError("ChatGPT returned no identity token.")
        try:
            import jwt
        except ImportError as error:
            raise RuntimeError(
                "JWT verification support is unavailable. Reinstall ChemFlow with "
                "the desktop extra: pip install -e '.[desktop]'"
            ) from error
        signing_key = jwt.PyJWKClient(JWKS_URL).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            leeway=5,
            options={"require": ["sub", "exp", "iat"]},
        )
        if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise RuntimeError("ChatGPT identity verification failed (nonce mismatch).")
        return claims

    @staticmethod
    def _pkce_challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
