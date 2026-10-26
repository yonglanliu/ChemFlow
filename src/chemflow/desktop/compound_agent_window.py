"""Qt window for agent-assisted compound selection."""

from __future__ import annotations

import os
import threading

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from chemflow.desktop.adme_page import export_prediction_csv
from chemflow.desktop.chatgpt_auth import ChatGPTAuth, ChatGPTSession
from chemflow.desktop.compound_agent import CompoundSelection, CompoundSelectionAgent
from chemflow.desktop.widgets.forms import action_button, field


class _AgentWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        frame: pd.DataFrame,
        request: str,
        model: str,
        auth_mode: str,
        api_key: str,
    ) -> None:
        super().__init__()
        self.frame = frame
        self.request = request
        self.model = model
        self.auth_mode = auth_mode
        self.api_key = api_key

    @Slot()
    def run(self) -> None:
        try:
            token = ChatGPTAuth().access_token() if self.auth_mode == "chatgpt" else None
            selection = CompoundSelectionAgent(
                api_key=self.api_key or None, oauth_access_token=token
            ).run(
                self.frame, self.request, model=self.model
            )
            self.completed.emit(selection)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class _AuthWorker(QObject):
    completed = Signal(str, object)
    failed = Signal(str)

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            auth = ChatGPTAuth()
            result = (
                auth.sign_in(cancel_event=self.cancel_event)
                if self.action == "sign_in"
                else auth.disconnect()
            )
            self.completed.emit(self.action, result)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class CompoundAgentDialog(QDialog):
    """Non-blocking assistant operating on a snapshot of prediction results."""

    selection_applied = Signal(object)

    def __init__(self, frame: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compound Selection Agent · ChemFlow")
        self.resize(900, 720)
        self.setMinimumSize(720, 560)
        self.frame = frame.reset_index(drop=True).copy()
        self.selection: CompoundSelection | None = None
        self._thread: QThread | None = None
        self._worker: _AgentWorker | None = None
        self._auth_thread: QThread | None = None
        self._auth_worker: _AuthWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        title = QLabel("Compound Selection Agent")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)
        detail = QLabel(
            f"The agent can inspect and rank {len(self.frame):,} current prediction "
            "rows, then apply an explicit selection. Queried rows are sent to OpenAI. "
            "Predictions require experimental confirmation."
        )
        detail.setWordWrap(True)
        detail.setObjectName("Muted")
        layout.addWidget(detail)

        settings = QGridLayout()
        self.auth_mode = QComboBox()
        self.auth_mode.addItem("API key", "api_key")
        self.auth_mode.addItem("ChatGPT account", "chatgpt")
        self.model = QLineEdit(os.environ.get("CHEMFLOW_AGENT_MODEL", "gpt-5.5"))
        self.model.setPlaceholderText("OpenAI model")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText(
            "Using OPENAI_API_KEY"
            if os.environ.get("OPENAI_API_KEY", "").strip()
            else "Paste API key (kept for this window only)"
        )
        settings.addWidget(field("Authentication", self.auth_mode), 0, 0)
        settings.addWidget(field("Model", self.model), 0, 1)
        settings.addWidget(field("API key", self.api_key), 1, 0, 1, 2)
        layout.addLayout(settings)

        account_row = QHBoxLayout()
        self.account_status = QLabel("ChatGPT account not connected")
        self.account_status.setObjectName("Muted")
        self.connect_button = QPushButton("Continue with ChatGPT")
        self.connect_button.clicked.connect(self._sign_in)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self._disconnect)
        account_row.addWidget(self.account_status)
        account_row.addStretch()
        account_row.addWidget(self.disconnect_button)
        account_row.addWidget(self.connect_button)
        layout.addLayout(account_row)
        self._refresh_account_ui()

        eligibility_note = QLabel(
            "ChatGPT sign-in requires an eligible Plus/Pro account and workspace. "
            "Managed Work or Enterprise workspace policy may block third-party plan "
            "usage; select API key above when that occurs."
        )
        eligibility_note.setWordWrap(True)
        eligibility_note.setObjectName("Muted")
        layout.addWidget(eligibility_note)

        key_note = QLabel(
            "Leave the API-key field empty to use OPENAI_API_KEY. A pasted key is "
            "kept in memory only and is discarded when this window closes."
        )
        key_note.setWordWrap(True)
        key_note.setObjectName("Muted")
        layout.addWidget(key_note)

        self.transcript = QTextBrowser()
        self.transcript.setMinimumHeight(250)
        self.transcript.setPlaceholderText(
            "Ask the agent to balance LogD, kinetic or thermodynamic solubility, "
            "MC uncertainty, similarity, and OOD risk."
        )
        layout.addWidget(self.transcript, 1)

        self.request = QPlainTextEdit()
        self.request.setMaximumHeight(120)
        self.request.setPlaceholderText(
            "Example: Select up to 10 in-domain compounds with kinetic solubility "
            "above 50 µM, low MC uncertainty, and the lowest LogD available."
        )
        layout.addWidget(self.request)

        controls = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("Muted")
        self.export_button = QPushButton("Export selected CSV…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_selection)
        self.run_button = action_button("Ask agent and select")
        self.run_button.clicked.connect(self._start)
        controls.addWidget(self.status)
        controls.addStretch()
        controls.addWidget(self.export_button)
        controls.addWidget(self.run_button)
        layout.addLayout(controls)

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @Slot()
    def _start(self) -> None:
        if self._thread is not None:
            return
        prompt = self.request.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(
                self, "Selection objective required", "Describe which compounds to select."
            )
            return
        auth_mode = str(self.auth_mode.currentData())
        api_key = ""
        if auth_mode == "api_key":
            api_key = self.api_key.text().strip() or os.environ.get(
                "OPENAI_API_KEY", ""
            ).strip()
        if auth_mode == "api_key" and not api_key:
            QMessageBox.warning(
                self,
                "OpenAI API key required",
                "Paste an API key into this window or set OPENAI_API_KEY.",
            )
            self.api_key.setFocus()
            return
        if auth_mode == "chatgpt" and not self._chatgpt_connected():
            QMessageBox.warning(
                self,
                "ChatGPT account required",
                "Select Continue with ChatGPT and finish sign-in first.",
            )
            return
        self.transcript.append(f"<b>You</b><br>{prompt}")
        self.status.setText("Agent is inspecting candidates…")
        self.run_button.setEnabled(False)
        self._thread = QThread(self)
        self._worker = _AgentWorker(
            self.frame.copy(), prompt, self.model.text(), auth_mode, api_key
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.completed.connect(self._complete)
        self._worker.failed.connect(self._failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.completed.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    @Slot()
    def _sign_in(self) -> None:
        if (
            self._auth_thread is not None
            and self._auth_worker is not None
            and self._auth_worker.action == "sign_in"
        ):
            self.account_status.setText("Cancelling ChatGPT sign-in…")
            self._auth_worker.cancel()
            self.connect_button.setEnabled(False)
            return
        self._start_auth_action("sign_in")

    @Slot()
    def _disconnect(self) -> None:
        self._start_auth_action("disconnect")

    def _start_auth_action(self, action: str) -> None:
        if self._auth_thread is not None:
            return
        self.account_status.setText(
            "Waiting for browser sign-in…" if action == "sign_in" else "Disconnecting…"
        )
        self.connect_button.setText(
            "Cancel sign-in" if action == "sign_in" else "Continue with ChatGPT"
        )
        self.connect_button.setEnabled(action == "sign_in")
        self.disconnect_button.setEnabled(False)
        self._auth_thread = QThread(self)
        self._auth_worker = _AuthWorker(action)
        self._auth_worker.moveToThread(self._auth_thread)
        self._auth_thread.started.connect(self._auth_worker.run)
        self._auth_worker.completed.connect(self._auth_complete)
        self._auth_worker.failed.connect(self._auth_failed)
        self._auth_worker.completed.connect(self._auth_thread.quit)
        self._auth_worker.failed.connect(self._auth_thread.quit)
        self._auth_worker.completed.connect(self._auth_worker.deleteLater)
        self._auth_worker.failed.connect(self._auth_worker.deleteLater)
        self._auth_thread.finished.connect(self._auth_thread_finished)
        self._auth_thread.start()

    @Slot(str, object)
    def _auth_complete(self, action: str, result: object) -> None:
        if action == "sign_in" and isinstance(result, ChatGPTSession):
            self.account_status.setText(f"Connected: {result.email}")
            self.auth_mode.setCurrentIndex(self.auth_mode.findData("chatgpt"))
            if result.models:
                self.model.setText(result.models[0][0])
                self.model.setToolTip(
                    "Available: "
                    + ", ".join(display for _slug, display in result.models)
                )
        else:
            self.account_status.setText("ChatGPT account disconnected")
            self.auth_mode.setCurrentIndex(self.auth_mode.findData("api_key"))

    @Slot(str)
    def _auth_failed(self, message: str) -> None:
        self._refresh_account_ui()
        if "cancelled" not in message.lower():
            QMessageBox.critical(
                self,
                "ChatGPT account connection failed",
                message
                + "\n\nIf the browser says this app is unavailable for your "
                "workspace, use an eligible personal Plus/Pro workspace or select "
                "API key in ChemFlow. Workspace policy cannot be bypassed by the app.",
            )

    @Slot()
    def _auth_thread_finished(self) -> None:
        if self._auth_thread is not None:
            self._auth_thread.deleteLater()
        self._auth_thread = None
        self._auth_worker = None
        self.connect_button.setText("Continue with ChatGPT")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(self._chatgpt_connected())

    def _chatgpt_connected(self) -> bool:
        try:
            return ChatGPTAuth().is_connected()
        except Exception:
            return False

    def _refresh_account_ui(self) -> None:
        try:
            auth = ChatGPTAuth()
            connected = auth.is_connected()
            label = auth.account_label() if connected else ""
            models = auth.cached_models() if connected else ()
        except Exception:
            connected = False
            label = ""
            models = ()
        self.account_status.setText(
            f"Connected: {label}" if connected else "ChatGPT account not connected"
        )
        self.disconnect_button.setEnabled(connected)
        if connected:
            self.auth_mode.setCurrentIndex(self.auth_mode.findData("chatgpt"))
            slugs = {slug for slug, _display in models}
            if models and self.model.text().strip() not in slugs:
                self.model.setText(models[0][0])
                self.model.setToolTip(
                    "Available: "
                    + ", ".join(display for _slug, display in models)
                )

    @Slot(object)
    def _complete(self, selection: CompoundSelection) -> None:
        self.selection = selection
        self.transcript.append(
            f"<br><b>Agent</b><br>{selection.response_text}<br><br>"
            f"<b>Selected rows:</b> {', '.join(map(str, selection.row_ids))}"
        )
        self.status.setText(f"Selected {len(selection.row_ids):,} compound(s)")
        self.export_button.setEnabled(True)
        self.selection_applied.emit(selection)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.status.setText("Agent request failed")
        QMessageBox.critical(self, "Compound Selection Agent failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        self.run_button.setEnabled(True)
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    @Slot()
    def _export_selection(self) -> None:
        if self.selection is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export agent-selected compounds",
            "agent_selected_compounds.csv",
            "CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not selected:
            return
        frame = self.frame.iloc[list(self.selection.row_ids)].copy()
        frame.insert(0, "AI_selection_rationale", self.selection.rationale)
        try:
            output = export_prediction_csv(frame, selected)
        except Exception as error:
            QMessageBox.critical(self, "Cannot export selection", str(error))
            return
        self.status.setText(f"Saved {len(frame):,} selected compounds to {output.name}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None or self._auth_thread is not None:
            QMessageBox.information(
                self,
                "ChemFlow is working",
                "Wait for the current agent or account request to finish.",
            )
            event.ignore()
            return
        self.api_key.clear()
        super().closeEvent(event)
