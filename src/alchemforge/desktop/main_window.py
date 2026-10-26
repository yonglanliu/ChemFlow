import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QPointF, QProcess, QSize, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QWindow,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alchemforge.desktop.dialogs.abfe_gromacs_dialog import ABFEGromacsDialog
from alchemforge.desktop.dialogs.gromacs_ligand_prepare_dialog import (
    GromacsLigandPrepareDialog,
)
from alchemforge.desktop.dialogs.gromacs_system_prepare_dialog import (
    GromacsSystemPrepareDialog,
)
from alchemforge.desktop.dialogs.ligand_network_planner_dialog import (
    LigandNetworkPlannerDialog,
)
from alchemforge.desktop.dialogs.ligand_rbfe_gromacs_dialog import (
    LigandRBFEGromacsDialog,
)
from alchemforge.desktop.dialogs.minimize_dialog import (
    GromacsMinimizeDialog,
    OpenMMMinimizeDialog,
)
from alchemforge.desktop.dialogs.openfe_charge_dialog import (
    OpenFEChargeMoleculesDialog,
)
from alchemforge.desktop.dialogs.openfe_execute_dialog import OpenFEExecuteDialog
from alchemforge.desktop.dialogs.openfe_workflow_dialog import (
    OpenFEABFEDialog,
    OpenFERBFEDialog,
    OpenFESepTopDialog,
)
from alchemforge.desktop.dialogs.pdbfixer_dialog import PDBFixerDialog
from alchemforge.desktop.dialogs.ResAlchemFEP_GROMACS_Analysis_Dialog import (
    ResFEPGromacsAnalysisDialog,
)
from alchemforge.desktop.dialogs.ResAlchemFEP_GROMACS_Dialog import (
    ResAlchemFEPDialog,
)
from alchemforge.desktop.dialogs.septop_gromacs_dialog import SepTopGromacsDialog
from alchemforge.desktop.dialogs.structure_viewer_dialog import StructureViewerDialog
from alchemforge.desktop.modeless import show_modeless
from alchemforge.desktop.styles.style import STYLE


class MolecularSidebar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(450)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor("#f8fcff"))
        g.setColorAt(0.55, QColor("#edf6ff"))
        g.setColorAt(1.0, QColor("#dceefe"))
        p.fillRect(self.rect(), g)

        wave = QPainterPath()
        wave.moveTo(0, self.height() * 0.66)
        wave.cubicTo(
            self.width() * 0.25,
            self.height() * 0.59,
            self.width() * 0.58,
            self.height() * 0.84,
            self.width(),
            self.height() * 0.68,
        )
        wave.lineTo(self.width(), self.height())
        wave.lineTo(0, self.height())
        wave.closeSubpath()
        p.fillPath(wave, QColor(176, 215, 249, 80))

        atoms = [
            (0.13, 0.79, 20),
            (0.31, 0.74, 14),
            (0.48, 0.80, 22),
            (0.66, 0.74, 14),
            (0.82, 0.80, 17),
            (0.31, 0.87, 14),
            (0.53, 0.90, 18),
        ]
        pts = [QPointF(self.width() * x, self.height() * y) for x, y, _ in atoms]

        p.setPen(QPen(QColor(107, 166, 220, 125), 3))
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (2, 6), (5, 6)]:
            p.drawLine(pts[a], pts[b])

        for i, (x, y, r) in enumerate(atoms):
            c = pts[i]
            rg = QLinearGradient(c.x() - r, c.y() - r, c.x() + r, c.y() + r)
            rg.setColorAt(0, QColor(235, 247, 255, 235))
            rg.setColorAt(1, QColor(63, 132, 208, 205))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(rg))
            p.drawEllipse(c, r, r)

        p.setPen(QPen(QColor(95, 157, 216, 45), 2))
        upper = [
            QPointF(16, 120),
            QPointF(45, 104),
            QPointF(74, 120),
            QPointF(103, 104),
        ]
        for a, b in zip(upper[:-1], upper[1:]):
            p.drawLine(a, b)
        for c in upper:
            p.drawEllipse(c, 5, 5)

        super().paintEvent(event)


class HeroBanner(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(335)
        self.setMaximumHeight(385)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()

        g = QLinearGradient(0, 0, r.width(), r.height())
        g.setColorAt(0, QColor("#f7fcff"))
        g.setColorAt(0.52, QColor("#d8ecff"))
        g.setColorAt(1, QColor("#83bced"))
        p.fillRect(r, g)

        # ribbon motif
        p.setPen(QPen(QColor(255, 255, 255, 110), 3))
        for offset in range(0, 40, 8):
            path = QPainterPath()
            path.moveTo(0, r.height() - 55 + offset)
            path.cubicTo(
                r.width() * 0.28,
                r.height() - 135 + offset,
                r.width() * 0.50,
                r.height() + 10 + offset,
                r.width() * 0.78,
                r.height() - 100 + offset,
            )
            p.drawPath(path)

        # faint chemistry motifs
        p.setPen(QPen(QColor(78, 150, 218, 48), 2))
        for cx, cy in [(70, 45), (140, 78), (215, 42), (300, 82), (380, 52)]:
            hp = QPainterPath()
            for i in range(7):
                a = math.radians(60 * i - 30)
                x = cx + 17 * math.cos(a)
                y = cy + 17 * math.sin(a)
                if i == 0:
                    hp.moveTo(x, y)
                else:
                    hp.lineTo(x, y)
            p.drawPath(hp)

        # abstract protein pocket
        colors = [
            QColor(248, 252, 255, 240),
            QColor(191, 222, 248, 230),
            QColor(111, 174, 228, 225),
            QColor(45, 118, 190, 220),
        ]
        blobs = [
            (0.80, 0.18, 55, 0),
            (0.87, 0.17, 64, 1),
            (0.94, 0.22, 62, 0),
            (0.79, 0.34, 70, 2),
            (0.88, 0.34, 75, 3),
            (0.97, 0.37, 66, 1),
            (0.80, 0.54, 76, 2),
            (0.89, 0.53, 84, 3),
            (0.98, 0.56, 72, 1),
            (0.83, 0.74, 70, 1),
            (0.94, 0.73, 78, 0),
        ]
        p.setPen(Qt.NoPen)
        for xr, yr, rad, ci in blobs:
            p.setBrush(colors[ci])
            p.drawEllipse(QPointF(r.width() * xr, r.height() * yr), rad, rad)

        pocket = QPointF(r.width() * 0.88, r.height() * 0.48)
        p.setBrush(QColor(21, 73, 133, 205))
        p.drawEllipse(pocket, 68, 56)

        ligand = [
            QPointF(pocket.x() - 34, pocket.y() + 13),
            QPointF(pocket.x() - 10, pocket.y() - 10),
            QPointF(pocket.x() + 15, pocket.y() + 3),
            QPointF(pocket.x() + 40, pocket.y() - 20),
            QPointF(pocket.x() + 50, pocket.y() + 14),
        ]
        p.setPen(QPen(QColor("#ffffff"), 7, Qt.SolidLine, Qt.RoundCap))
        for a, b in zip(ligand[:-1], ligand[1:]):
            p.drawLine(a, b)
        for i, c in enumerate(ligand):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#f0443e") if i in (3, 4) else QColor("#2465c4"))
            p.drawEllipse(c, 7, 7)

        # hero copy
        p.setPen(QColor("#103f7c"))
        p.setFont(QFont("Arial", 34, QFont.Bold))
        p.drawText(52, 96, "AlchemForge")

        p.setPen(QColor("#57779a"))
        p.setFont(QFont("Arial", 18))
        p.drawText(54, 133, "Molecular Alchemy for Real Discovery")

        p.setPen(QColor("#365d84"))
        p.setFont(QFont("Arial", 11))
        p.drawText(
            54, 173, "An integrated platform for alchemical free-energy calculations,"
        )
        p.drawText(54, 195, "molecular simulation, and drug discovery.")

        p.setPen(QColor("#245f9b"))
        p.setFont(QFont("Arial", 10))
        p.drawText(55, 238, "⚡  Streamline")
        p.drawText(55, 258, "    complex workflows")
        p.drawText(230, 238, "▥  From structure")
        p.drawText(230, 258, "    to insight")
        p.drawText(415, 238, "◎  Accelerate")
        p.drawText(415, 258, "    real discovery")

        p.setPen(QColor("#2f7ed8"))
        p.setFont(QFont("Arial", 11, QFont.StyleItalic))
        p.drawText(55, 302, "Turning molecular ideas into real impact.")
        p.setPen(QPen(QColor("#76adea"), 2))
        p.drawLine(55, 310, 300, 310)

        super().paintEvent(event)


class ActionCard(QFrame):
    def __init__(self, title, subtitle, icon, callback=None):
        super().__init__()
        self.setObjectName("Card")
        self.callback = callback
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(120)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(6)

        ico = QLabel()
        ico.setAlignment(Qt.AlignCenter)
        ico.setPixmap(icon.pixmap(36, 36))

        ttl = QLabel(title)
        ttl.setAlignment(Qt.AlignCenter)
        ttl.setStyleSheet("font-size:12pt; font-weight:700;")

        sub = QLabel(subtitle)
        sub.setAlignment(Qt.AlignCenter)
        sub.setWordWrap(True)
        sub.setObjectName("Muted")

        lay.addWidget(ico)
        lay.addWidget(ttl)
        lay.addWidget(sub)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.callback:
            self.callback()
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_path = None
        self.working_directory = os.getcwd()
        self.job_records = []
        self.job_processes = {}
        self.viewer_process = None
        self.viewer_window = None
        self.viewer_output = b""
        self.viewer_control_path = None
        self.dssp_process = None
        self.viewer_structure_path = None
        self.viewer_dssp_path = None

        self.setWindowTitle("AlchemForge")
        self.resize(1500, 930)
        self.setMinimumSize(1120, 720)

        self.build_menu_bar()
        self.build_toolbar()
        self.build_ui()
        self.build_status_bar()

    def build_menu_bar(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")

        a = QAction("New Project", self)
        a.setShortcut("Ctrl+N")
        a.triggered.connect(self.new_project)
        fm.addAction(a)

        a = QAction("Open Project...", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_project)
        fm.addAction(a)

        a = QAction("Open C++ Viewer...", self)
        a.setShortcut("Ctrl+V")
        a.triggered.connect(self.open_cpp_viewer)
        fm.addAction(a)

        a = QAction("Open 3D Structure Viewer...", self)
        a.setShortcut("Ctrl+Shift+V")
        a.triggered.connect(self.open_structure_viewer)
        fm.addAction(a)

        self.recent_menu = fm.addMenu("Open Recent Project")
        fm.addSeparator()

        a = QAction("Close Project", self)
        a.triggered.connect(self.close_project)
        fm.addAction(a)

        fm.addSeparator()

        a = QAction("Change Working Directory...", self)
        a.setShortcut("Ctrl+D")
        a.triggered.connect(self.change_working_directory)
        fm.addAction(a)

        fm.addSeparator()
        fm.addAction("Exit", self.close, "Ctrl+Q")

        em = mb.addMenu("Edit")
        em.addAction("Undo")
        em.addAction("Redo")
        em.addSeparator()
        em.addAction("Preferences...")

        wm = mb.addMenu("Workspace")
        wm.addAction("Project Settings...")
        wm.addAction("Task Manager")
        wm.addSeparator()
        wm.addAction("Refresh Workspace")

        sm = mb.addMenu("Scripts")
        sm.addAction("Run Script...")
        sm.addAction("Open Script Editor")
        sm.addAction("Recent Scripts")

        vm = mb.addMenu("View")
        vm.addAction("Home")
        vm.addAction("Show Sidebar")
        vm.addAction("Show Project Panel")
        vm.addSeparator()
        vm.addAction("Reset Layout")

        winm = mb.addMenu("Window")
        winm.addAction("Minimize", self.showMinimized)
        winm.addAction("Maximize", self.showMaximized)
        winm.addAction("Restore", self.showNormal)

        hm = mb.addMenu("Help")
        hm.addAction("Documentation")
        hm.addAction("About AlchemForge", self.show_about)

    def build_toolbar(self):
        tb = QToolBar("Main")
        tb.setIconSize(QSize(28, 28))
        tb.setMovable(False)
        self.addToolBar(tb)
        s = self.style()

        for icon, text, cb in [
            (QStyle.SP_FileIcon, "New", self.new_project),
            (QStyle.SP_DirOpenIcon, "Open", self.open_project),
            (QStyle.SP_DialogSaveButton, "Save", None),
        ]:
            a = QAction(s.standardIcon(icon), text, self)
            if cb:
                a.triggered.connect(cb)
            tb.addAction(a)

        tb.addSeparator()

        for icon, text, cb in [
            (QStyle.SP_MediaPlay, "Run", self.run_workflow),
            (QStyle.SP_MediaStop, "Stop", None),
        ]:
            a = QAction(s.standardIcon(icon), text, self)
            if cb:
                a.triggered.connect(cb)
            tb.addAction(a)

        tb.addSeparator()
        a = QAction(s.standardIcon(QStyle.SP_FileDialogDetailedView), "Settings", self)
        tb.addAction(a)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search (Ctrl+K)")
        self.search_edit.setMaximumWidth(220)
        tb.addWidget(self.search_edit)

        brand = QLabel("   AlchemForge")
        brand.setStyleSheet(
            "font-size:19pt; font-weight:700; color:#123f76; padding:0 18px;"
        )
        tb.addWidget(brand)

    def build_ui(self):
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)

        split.addWidget(self.build_sidebar())
        self.center_stack = QStackedWidget()
        self.home_page = self.build_center()
        self.center_stack.addWidget(self.home_page)
        self.jobs_page = self.build_jobs_page()
        self.center_stack.addWidget(self.jobs_page)
        self.viewer_page = QWidget()
        viewer_page_layout = QHBoxLayout(self.viewer_page)
        viewer_page_layout.setContentsMargins(0, 0, 0, 0)

        viewer_canvas = QFrame()
        viewer_canvas.setObjectName("Card")
        self.viewer_layout = QVBoxLayout(viewer_canvas)
        self.viewer_layout.setContentsMargins(10, 10, 10, 10)
        self.viewer_layout.addWidget(QLabel("AlchemViewer"))
        self.viewer_layout.addWidget(QLabel("Open a PDB file to start AlchemViewer."))
        viewer_page_layout.addWidget(viewer_canvas, 1)

        viewer_controls = QFrame()
        viewer_controls.setObjectName("RightCard")
        viewer_controls.setMinimumWidth(230)
        viewer_controls.setMaximumWidth(280)
        controls_layout = QVBoxLayout(viewer_controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.addWidget(QLabel("Objects"))
        self.viewer_object_label = QLabel("All")
        self.viewer_object_label.setObjectName("Muted")
        controls_layout.addWidget(self.viewer_object_label)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Chain"))
        self.viewer_chain_combo = QComboBox()
        self.viewer_chain_combo.addItem("All chains")
        controls_layout.addWidget(self.viewer_chain_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Background"))
        self.viewer_background_combo = QComboBox()
        self.viewer_background_combo.addItems(
            ["Black", "White", "Dark gray", "Light gray"]
        )
        controls_layout.addWidget(self.viewer_background_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Show As"))

        self.viewer_style_combo = QComboBox()
        self.viewer_style_combo.addItems(
            [
                "Lines",
                "Sticks",
                "Spheres",
                "Dots",
                "Surface",
                "Mesh",
                "Cartoon",
                "Ribbon",
                "Labels",
                "Loop",
                "Ball and stick",
                "Spacefill",
            ]
        )
        self.viewer_style_combo.setCurrentText("Cartoon")
        controls_layout.addWidget(self.viewer_style_combo)
        for style_name in (
            "Cartoon",
            "Ribbon",
            "Loop",
            "Lines",
            "Sticks",
            "Spheres",
            "Dots",
            "Surface",
            "Mesh",
            "Labels",
            "Ball and stick",
            "Spacefill",
        ):
            style_button = QPushButton(style_name)
            style_button.setCheckable(True)
            style_button.setChecked(style_name == "Cartoon")
            style_button.clicked.connect(
                lambda checked, value=style_name: self.select_viewer_style(value)
            )
            controls_layout.addWidget(style_button)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Color By"))
        self.viewer_color_combo = QComboBox()
        self.viewer_color_combo.addItems(
            ["Element", "Spectrum", "Secondary structure", "Chain"]
        )
        self.viewer_color_combo.setCurrentText("Secondary structure")
        controls_layout.addWidget(self.viewer_color_combo)
        self.viewer_color_combo.setToolTip(
            "Choose element, spectrum, secondary-structure, or chain coloring"
        )
        controls_layout.addSpacing(18)
        controls_layout.addWidget(QLabel("Controls"))
        controls_layout.addWidget(
            QLabel("Left drag: rotate\nWheel: zoom\nEsc: close viewer")
        )
        controls_layout.addStretch()
        viewer_page_layout.addWidget(viewer_controls)
        self.center_stack.addWidget(self.viewer_page)
        split.addWidget(self.center_stack)
        split.addWidget(self.build_right())
        split.setSizes([220, 930, 320])
        self.setCentralWidget(split)

    def build_jobs_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Jobs")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        controls = QHBoxLayout()
        terminate_job = QPushButton("Terminate Selected Job")
        terminate_job.setToolTip(
            "Stop the selected local process or cancel its submitted SLURM jobs"
        )
        terminate_job.clicked.connect(self.terminate_selected_job)
        controls.addWidget(terminate_job)
        refresh_jobs = QPushButton("Refresh")
        refresh_jobs.clicked.connect(self.poll_jobs)
        controls.addWidget(refresh_jobs)
        controls.addStretch()
        layout.addLayout(controls)

        self.jobs_table = QTableWidget(0, 5)
        self.jobs_table.setHorizontalHeaderLabels(
            ["Job", "Task", "Status", "SLURM IDs", "Error"]
        )
        self.jobs_table.verticalHeader().setVisible(False)
        self.jobs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.jobs_table.setSelectionMode(QTableWidget.SingleSelection)
        self.jobs_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.jobs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.jobs_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.jobs_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.jobs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.jobs_table)
        self.job_timer = QTimer(self)
        self.job_timer.timeout.connect(self.poll_jobs)
        self.job_timer.start(5000)
        return page

    def start_backend_job(self, task_directory, script_path, config_path):
        process = QProcess(self)
        process.setWorkingDirectory(task_directory)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "alchemforge.desktop.job_runner",
                script_path,
                config_path,
            ]
        )
        record = {
            "name": Path(task_directory).name,
            "task": task_directory,
            "status": "RUNNING",
            "ids": [],
            "error": "",
            "output": "",
            "process": process,
            "termination_requested": False,
        }
        self.job_records.append(record)
        self.job_processes[id(process)] = record
        process.readyReadStandardOutput.connect(lambda: self.read_job_output(process))
        process.readyReadStandardError.connect(lambda: self.read_job_error(process))
        process.finished.connect(
            lambda code, status: self.backend_job_finished(process, code, status)
        )
        process.start()
        self.center_stack.setCurrentWidget(self.jobs_page)
        self.refresh_jobs_table()

    def read_job_output(self, process):
        record = self.job_processes.get(id(process))
        if record is not None:
            record["output"] += bytes(process.readAllStandardOutput()).decode(
                errors="replace"
            )

    def read_job_error(self, process):
        record = self.job_processes.get(id(process))
        if record is not None:
            record["error"] += bytes(process.readAllStandardError()).decode(
                errors="replace"
            )

    def backend_job_finished(self, process, exit_code, _exit_status):
        self.read_job_output(process)
        self.read_job_error(process)
        record = self.job_processes.get(id(process))
        if record is None:
            return

        if record.get("termination_requested"):
            record["status"] = "TERMINATED"
            record["error"] = "Terminated by user."
        elif exit_code != 0:
            record["status"] = "FAILED"
            record["error"] = (
                record["error"].strip()
                or record["output"].strip()
                or f"Submission exited with code {exit_code}."
            )
        else:
            record["ids"] = re.findall(
                r"(?:job:|Submitted batch job)\s*([0-9]+)",
                record["output"],
                flags=re.IGNORECASE,
            )
            record["status"] = "SUBMITTED" if record["ids"] else "COMPLETED"
        self.refresh_jobs_table()

    @staticmethod
    def _process_descendants(parent_pid):
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,ppid="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        children = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            pid, ppid = (int(value) for value in fields)
            children.setdefault(ppid, []).append(pid)
        descendants = []
        pending = list(children.get(parent_pid, ()))
        while pending:
            pid = pending.pop()
            descendants.append(pid)
            pending.extend(children.get(pid, ()))
        return descendants

    def _signal_process_tree(self, process, requested_signal):
        pid = int(process.processId())
        if pid <= 0:
            return
        if os.name == "posix":
            try:
                os.killpg(pid, requested_signal)
                return
            except OSError:
                pass
        for child_pid in reversed(self._process_descendants(pid)):
            try:
                os.kill(child_pid, requested_signal)
            except OSError:
                pass
        try:
            os.kill(pid, requested_signal)
        except OSError:
            pass

    def _force_terminate_job(self, process):
        record = self.job_processes.get(id(process))
        if record is None or not record.get("termination_requested"):
            return
        if process.state() != QProcess.NotRunning:
            self._signal_process_tree(process, signal.SIGKILL)
            process.kill()

    def terminate_selected_job(self):
        row = self.jobs_table.currentRow()
        if row < 0 or row >= len(self.job_records):
            QMessageBox.information(
                self, "Terminate Job", "Select a job in the table first."
            )
            return
        record = self.job_records[row]
        if record["status"] not in {"RUNNING", "SUBMITTED", "TERMINATING"}:
            QMessageBox.information(
                self,
                "Terminate Job",
                f"Job '{record['name']}' is not running or submitted.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Terminate Job",
            f"Terminate job '{record['name']}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        if record["status"] == "SUBMITTED" and record["ids"]:
            try:
                result = subprocess.run(
                    ["scancel", *record["ids"]],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                QMessageBox.critical(self, "Cannot Cancel Job", str(error))
                return
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip()
                QMessageBox.critical(
                    self, "Cannot Cancel Job", message or "scancel failed"
                )
                return
            record["status"] = "TERMINATED"
            record["error"] = "SLURM job cancelled by user."
            self.refresh_jobs_table()
            return

        process = record.get("process")
        if process is None or process.state() == QProcess.NotRunning:
            record["status"] = "TERMINATED"
            record["error"] = "Terminated by user."
            self.refresh_jobs_table()
            return
        record["termination_requested"] = True
        record["status"] = "TERMINATING"
        self._signal_process_tree(process, signal.SIGTERM)
        process.terminate()
        QTimer.singleShot(3000, lambda: self._force_terminate_job(process))
        self.refresh_jobs_table()

    def poll_jobs(self):
        for record in self.job_records:
            if record["status"] != "SUBMITTED" or not record["ids"]:
                continue
            try:
                result = subprocess.run(
                    [
                        "sacct",
                        "-X",
                        "-j",
                        ",".join(record["ids"]),
                        "--format=State",
                        "--noheader",
                        "--parsable2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode != 0:
                continue
            states = {
                line.strip().split("+")[0]
                for line in result.stdout.splitlines()
                if line.strip()
            }
            if not states:
                continue
            failed_states = {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
            }
            if states & failed_states:
                record["status"] = "FAILED"
                record["error"] = "SLURM state: " + ", ".join(sorted(states))
            elif states and states <= {"COMPLETED"}:
                record["status"] = "COMPLETED"
        self.refresh_jobs_table()

    def refresh_jobs_table(self):
        if not hasattr(self, "jobs_table"):
            return
        self.jobs_table.setRowCount(0)
        for record in self.job_records:
            row = self.jobs_table.rowCount()
            self.jobs_table.insertRow(row)
            values = (
                record["name"],
                record["task"],
                record["status"],
                ", ".join(record["ids"]),
                record["error"].strip(),
            )
            for column, value in enumerate(values):
                self.jobs_table.setItem(row, column, QTableWidgetItem(value))

    def build_sidebar(self):
        side = MolecularSidebar()

        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 8, 0, 12)

        # ============================================================
        # Navigation tree
        # ============================================================

        self.nav = QTreeWidget()
        self.nav.setObjectName("Navigation")
        self.nav.setStyleSheet("""
            QTreeWidget {
                background: transparent;
                border: none;
                outline: none;
                color: #23486f;
                font-size: 11pt;
            }

            QTreeWidget::item {
                height: 38px;
                padding-left: 8px;
                border-radius: 6px;
            }

            QTreeWidget::item:hover {
                background: rgba(72, 145, 215, 35);
            }

            QTreeWidget::item:selected {
                background: rgba(72, 145, 215, 55);
                color: #155b9e;
                font-weight: 600;
            }

            QTreeWidget::branch {
                background: transparent;
            }
            """)

        # Hide normal tree header
        self.nav.setHeaderHidden(True)

        # Remove root decoration lines
        self.nav.setRootIsDecorated(True)

        # Indentation for child items
        self.nav.setIndentation(22)

        # ============================================================
        # Home
        # ============================================================

        home = QTreeWidgetItem(self.nav, ["⌂   Home"])

        home.setData(0, Qt.UserRole, "home")

        # ============================================================
        # Binding Free Energy Workflows
        # ============================================================

        binding = QTreeWidgetItem(self.nav, ["▥   Binding Free Energy Workflow"])

        binding.setData(0, Qt.UserRole, "binding_free_energy")

        gromacs_workflows = QTreeWidgetItem(binding, ["GROMACS"])
        gromacs_workflows.setData(0, Qt.UserRole, "gromacs_workflows")

        gromacs_ligand_fep = QTreeWidgetItem(
            gromacs_workflows, ["LigandFEP-RBFE"]
        )
        gromacs_ligand_fep.setData(0, Qt.UserRole, "lig_rbfe_gromacs")

        gromacs_septop = QTreeWidgetItem(
            gromacs_workflows, ["LigandSepTop-RBFE"]
        )
        gromacs_septop.setData(0, Qt.UserRole, "septop_rbfe_gromacs")

        gromacs_abfe = QTreeWidgetItem(gromacs_workflows, ["LigandFEP-ABFE"])
        gromacs_abfe.setData(0, Qt.UserRole, "abfe_gromacs")

        gromacs_resfep = QTreeWidgetItem(gromacs_workflows, ["ResFEP-RBFE"])
        gromacs_resfep.setData(0, Qt.UserRole, "res_alchem_fep")

        openfe_workflows = QTreeWidgetItem(binding, ["OpenFE"])
        openfe_workflows.setData(0, Qt.UserRole, "openfe_workflows")

        openfe_ligand_fep = QTreeWidgetItem(openfe_workflows, ["LigandFEP-RBFE"])
        openfe_ligand_fep.setData(0, Qt.UserRole, "lig_alchem_fep")

        openfe_septop = QTreeWidgetItem(
            openfe_workflows, ["LigandSepTop-RBFE"]
        )
        openfe_septop.setData(0, Qt.UserRole, "septop_rbfe_openfe")

        openfe_abfe = QTreeWidgetItem(openfe_workflows, ["LigandFEP-ABFE"])
        openfe_abfe.setData(0, Qt.UserRole, "abfe_openfe")

        # ============================================================
        # Reusable Preparation Tools
        # ============================================================

        preparation = QTreeWidgetItem(self.nav, ["⚒   Preparation & Tools"])
        preparation.setData(0, Qt.UserRole, "preparation_tools")

        fix_pdb = QTreeWidgetItem(preparation, ["PDBFixer"])
        fix_pdb.setToolTip(
            0,
            "Repair protein, membrane, solvent, and cofactor PDB files, then "
            "optionally assemble them into one structure.",
        )
        fix_pdb.setData(0, Qt.UserRole, "fix_pdb_pdbfixer")

        ligand_preparation = QTreeWidgetItem(preparation, ["Ligand Preparation"])
        ligand_preparation.setData(0, Qt.UserRole, "ligand_preparation")
        openfe_ligand_preparation = QTreeWidgetItem(ligand_preparation, ["OpenFE"])
        openfe_ligand_preparation.setData(0, Qt.UserRole, "openfe_ligand_preparation")
        charge_openfe = QTreeWidgetItem(openfe_ligand_preparation, ["Charge Molecules"])
        charge_openfe.setData(0, Qt.UserRole, "charge_sdf_openfe")

        gromacs_ligand_preparation = QTreeWidgetItem(ligand_preparation, ["GROMACS"])
        gromacs_ligand_preparation.setData(0, Qt.UserRole, "gromacs_ligand_preparation")
        prepare_gromacs_ligands = QTreeWidgetItem(
            gromacs_ligand_preparation, ["Prepare Ligands"]
        )
        prepare_gromacs_ligands.setData(0, Qt.UserRole, "prepare_gromacs_ligands")

        network_planning = QTreeWidgetItem(preparation, ["Ligand Network Planning"])
        network_planning.setData(0, Qt.UserRole, "ligand_network_planning")

        system_preparation = QTreeWidgetItem(preparation, ["System Preparation"])
        system_preparation.setData(0, Qt.UserRole, "system_preparation")
        prepare_gromacs_system = QTreeWidgetItem(
            system_preparation, ["GROMACS System Builder"]
        )
        prepare_gromacs_system.setData(
            0, Qt.UserRole, "prepare_gromacs_system"
        )
        parameterize_pdb_sdf = QTreeWidgetItem(
            system_preparation, ["PDB + SDF Parameterization"]
        )
        parameterize_pdb_sdf.setToolTip(
            0,
            "Parameterize an assembled system with Amber protein/lipid force "
            "fields and OpenFF/GAFF small molecules, then export GROMACS files.",
        )
        parameterize_pdb_sdf.setData(
            0, Qt.UserRole, "parameterize_pdb_sdf"
        )
        structure_viewer = QTreeWidgetItem(system_preparation, ["3D Structure Viewer"])
        structure_viewer.setData(0, Qt.UserRole, "structure_viewer")
        minimization = QTreeWidgetItem(system_preparation, ["Energy Minimization"])
        minimization.setData(0, Qt.UserRole, "energy_minimization")
        minimize_openmm = QTreeWidgetItem(minimization, ["OpenFE / OpenMM"])
        minimize_openmm.setData(0, Qt.UserRole, "minimize_openmm")
        minimize_gromacs = QTreeWidgetItem(minimization, ["GROMACS"])
        minimize_gromacs.setData(0, Qt.UserRole, "minimize_gromacs")

        # ============================================================
        # Analysis
        # ============================================================

        analysis = QTreeWidgetItem(self.nav, ["↗   Analysis"])

        analysis.setData(0, Qt.UserRole, "analysis")

        res_rbfe_gromacs_analysis = QTreeWidgetItem(
            analysis, ["ResRBFE_GROMACS_Analysis"]
        )
        res_rbfe_gromacs_analysis.setData(0, Qt.UserRole, "res_rbfe_gromacs_analysis")
        # ============================================================
        # Jobs
        # ============================================================

        jobs = QTreeWidgetItem(self.nav, ["☷   Jobs"])

        jobs.setData(0, Qt.UserRole, "jobs")

        # ============================================================
        # Results
        # ============================================================

        results = QTreeWidgetItem(self.nav, ["▰   Results"])

        results.setData(0, Qt.UserRole, "results")

        # ============================================================
        # Initial state
        # ============================================================

        # Initially collapse workflow and tool groups
        binding.setExpanded(False)
        preparation.setExpanded(False)

        # Initially collapse Analysis
        analysis.setExpanded(False)

        # Select Home
        self.nav.setCurrentItem(home)

        # ============================================================
        # Signals
        # ============================================================

        self.nav.itemClicked.connect(self.on_navigation_clicked)

        lay.addWidget(self.nav, 1)

        # ============================================================
        # Bottom tagline
        # ============================================================

        tag = QLabel("Explore\nSimulate\nDesign\nDiscover")

        tag.setAlignment(Qt.AlignCenter)

        tag.setStyleSheet(
            """
            color: #4f90d7;
            font-size: 12pt;
            font-style: italic;
            padding-bottom: 18px;
            """
        )

        lay.addWidget(tag)

        return side

    def build_center(self):
        c = QWidget()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        tab = QFrame()
        tab.setObjectName("Card")
        t = QHBoxLayout(tab)
        t.setContentsMargins(10, 6, 10, 6)
        lab = QLabel("⌂  Home")
        lab.setStyleSheet("font-weight:700;")
        t.addWidget(lab)
        t.addStretch()
        x = QLabel("×")
        x.setStyleSheet("font-size:16pt; color:#44688e;")
        t.addWidget(x)
        lay.addWidget(tab)

        hero = HeroBanner()
        hero.setObjectName("Card")
        lay.addWidget(hero)

        sec = QLabel("⚡  Quick Start")
        sec.setObjectName("SectionTitle")
        lay.addWidget(sec)

        cards = QGridLayout()
        cards.setSpacing(10)
        s = self.style()
        cards.addWidget(
            ActionCard(
                "New Project",
                "Start a new workflow",
                s.standardIcon(QStyle.SP_FileIcon),
                self.new_project,
            ),
            0,
            0,
        )
        cards.addWidget(
            ActionCard(
                "Open Project",
                "Open an existing project",
                s.standardIcon(QStyle.SP_DirOpenIcon),
                self.open_project,
            ),
            0,
            1,
        )
        cards.addWidget(
            ActionCard(
                "Set Working Directory",
                "Choose working directory",
                s.standardIcon(QStyle.SP_DirIcon),
                self.change_working_directory,
            ),
            0,
            2,
        )
        lay.addLayout(cards)

        rh = QHBoxLayout()
        ttl = QLabel("◷  Recent Projects")
        ttl.setObjectName("SectionTitle")
        rh.addWidget(ttl)
        rh.addStretch()
        b = QPushButton("View All →")
        b.setStyleSheet(
            "border:none; color:#2f7ed8; font-weight:700; background:transparent;"
        )
        rh.addWidget(b)
        lay.addLayout(rh)

        self.recent_table = QTableWidget(0, 3)
        self.recent_table.setHorizontalHeaderLabels(["Name", "Path", "Last Opened"])
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.recent_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.recent_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        lay.addWidget(self.recent_table, 1)
        return c

    def build_right(self):
        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(350)
        lay = QVBoxLayout(right)
        lay.setContentsMargins(8, 10, 10, 10)
        lay.setSpacing(12)

        project = QFrame()
        project.setObjectName("RightCard")
        p = QVBoxLayout(project)
        p.setContentsMargins(14, 12, 14, 12)
        title = QLabel("▰  Project")
        title.setObjectName("SideTitle")
        p.addWidget(title)
        self.project_label = QLabel("No project open")
        self.project_label.setObjectName("Muted")
        p.addWidget(self.project_label)
        p.addSpacing(8)
        p.addWidget(QLabel("Working Directory:"))
        self.workdir_label = QLabel(self.working_directory)
        self.workdir_label.setWordWrap(True)
        p.addWidget(self.workdir_label)
        ch = QPushButton("Change Directory...")
        ch.clicked.connect(self.change_working_directory)
        p.addWidget(ch)
        lay.addWidget(project)

        quick = QFrame()
        quick.setObjectName("RightCard")
        q = QVBoxLayout(quick)
        q.setContentsMargins(14, 12, 14, 12)
        title = QLabel("⌄  Quick Actions")
        title.setObjectName("SideTitle")
        q.addWidget(title)
        for label, cb in (
            ("New Project", self.new_project),
            ("Open Project...", self.open_project),
            ("Run Workflow...", self.run_workflow),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            q.addWidget(btn)
        lay.addWidget(quick)

        tips = QFrame()
        tips.setObjectName("RightCard")
        t = QVBoxLayout(tips)
        t.setContentsMargins(14, 12, 14, 12)
        title = QLabel("ⓘ  Tips")
        title.setObjectName("SideTitle")
        t.addWidget(title)
        body = QLabel(
            "Create or open a project to get started. Use the sidebar to navigate through the AlchemForge workflow."
        )
        body.setObjectName("Muted")
        body.setWordWrap(True)
        t.addWidget(body)
        mol = QLabel("⌬      ⌬\n   ╲  ╱\n    ⌬")
        mol.setAlignment(Qt.AlignCenter)
        mol.setStyleSheet("font-size:24pt; color:#c4ddf5; padding:20px 0;")
        t.addWidget(mol)
        slogan = QLabel("Small Molecules\nBig Possibilities")
        slogan.setAlignment(Qt.AlignCenter)
        slogan.setStyleSheet("color:#2f7ed8; font-size:12pt; font-style:italic;")
        t.addWidget(slogan)
        lay.addWidget(tips, 1)

        return right

    def build_status_bar(self):
        st = QStatusBar()
        self.setStatusBar(st)
        ready = QLabel("●  Ready")
        ready.setStyleSheet("color:#169c4b;")
        st.addWidget(ready)
        st.addPermanentWidget(
            QLabel(
                f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
        )
        self.status_project = QLabel("No project")
        st.addPermanentWidget(self.status_project)
        self.status_workdir = QLabel(self.working_directory)
        st.addPermanentWidget(self.status_workdir)

    def new_project(self):
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Create AlchemForge Project",
            self.working_directory,
            "AlchemForge Project (*.afp)",
        )
        if not p:
            return
        project = Path(p)
        if project.suffix.lower() != ".afp":
            project = project.with_suffix(".afp")
        project.touch(exist_ok=True)
        self.project_path = project
        self.project_label.setText(project.stem)
        self.status_project.setText(project.stem)
        self.setWindowTitle(f"AlchemForge — {project.stem}")
        self.add_recent(project)

    def open_project(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Open AlchemForge Project",
            self.working_directory,
            "AlchemForge Project (*.afp);;All Files (*)",
        )
        if not p:
            return
        project = Path(p)
        self.project_path = project
        self.project_label.setText(project.stem)
        self.status_project.setText(project.stem)
        self.setWindowTitle(f"AlchemForge — {project.stem}")
        self.add_recent(project)

    def open_structure_viewer(self):
        dialog = StructureViewerDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        show_modeless(self, dialog)

    def open_cpp_viewer(self):
        structure, _ = QFileDialog.getOpenFileName(
            self,
            "Open Structure in AlchemViewer",
            self.working_directory,
            "PDB Structures (*.pdb *.ent);;All Files (*)",
        )
        if not structure:
            return

        viewer = (
            Path(__file__).resolve().parents[3]
            / "alchemforge_viewer"
            / "build"
            / "alchemforge_viewer"
        )
        if not viewer.is_file():
            QMessageBox.warning(
                self,
                "Viewer Not Built",
                f"Build the C++ viewer first:\n\n{viewer}",
            )
            return

        control = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="alchemviewer_",
            suffix=".control",
            delete=False,
        )
        control.write("style cartoon\ncolor secondary\nchain all\nbackground black\n")
        control.close()
        self.viewer_control_path = control.name
        self.viewer_style_combo.currentTextChanged.connect(self.write_viewer_control)
        self.viewer_color_combo.currentTextChanged.connect(self.write_viewer_control)
        self.viewer_chain_combo.currentTextChanged.connect(self.write_viewer_control)
        self.viewer_background_combo.currentTextChanged.connect(
            self.write_viewer_control
        )
        self.center_stack.setCurrentWidget(self.viewer_page)
        dssp_output = tempfile.NamedTemporaryFile(
            prefix="alchemviewer_",
            suffix=".dssp",
            delete=False,
        )
        dssp_output.close()
        self.viewer_structure_path = structure
        self.viewer_object_label.setText(Path(structure).stem)
        self.viewer_chain_combo.clear()
        self.viewer_chain_combo.addItem("All chains")
        chains = set()
        for line in Path(structure).read_text(errors="replace").splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                chain = line[21].strip()
                if chain:
                    chains.add(chain)
        self.viewer_chain_combo.addItems(sorted(chains))
        self.viewer_dssp_path = dssp_output.name
        self.dssp_process = QProcess(self)
        self.dssp_process.setProgram("bash")
        self.dssp_process.setArguments(
            [
                "-lc",
                'module load DSSP/2.3.0 && mkdssp -i "$1" -o "$2"',
                "alchemforge-dssp",
                structure,
                self.viewer_dssp_path,
            ]
        )
        self.dssp_process.finished.connect(
            lambda code, _status: self.start_cpp_viewer(
                viewer,
                code == 0,
            )
        )
        self.dssp_process.start()

    def select_viewer_style(self, style):
        self.viewer_style_combo.setCurrentText(style)

    def start_cpp_viewer(self, viewer, dssp_succeeded):
        self.viewer_process = QProcess(self)
        self.viewer_process.setProgram(str(viewer))
        arguments = [self.viewer_structure_path, self.viewer_control_path]
        if dssp_succeeded:
            arguments.append(self.viewer_dssp_path)
        self.viewer_process.setArguments(arguments)
        self.viewer_process.readyReadStandardOutput.connect(self.embed_viewer_output)
        self.viewer_process.start()

    def write_viewer_control(self):
        if not self.viewer_control_path:
            return
        style = {
            "Lines": "lines",
            "Sticks": "sticks",
            "Spheres": "spheres",
            "Dots": "dots",
            "Surface": "surface",
            "Mesh": "mesh",
            "Spacefill": "spacefill",
            "Ball and stick": "ball_and_stick",
            "Cartoon": "cartoon",
            "Ribbon": "ribbon",
            "Labels": "labels",
            "Loop": "loop",
        }[self.viewer_style_combo.currentText()]
        color = {
            "Element": "element",
            "Spectrum": "spectrum",
            "Secondary structure": "secondary",
            "Chain": "chain",
        }[self.viewer_color_combo.currentText()]
        chain = self.viewer_chain_combo.currentText()
        chain = "all" if chain == "All chains" else chain
        background = {
            "Black": "black",
            "White": "white",
            "Dark gray": "dark_gray",
            "Light gray": "light_gray",
        }[self.viewer_background_combo.currentText()]
        Path(self.viewer_control_path).write_text(
            f"style {style}\ncolor {color}\nchain {chain}\nbackground {background}\n",
            encoding="utf-8",
        )

    def embed_viewer_output(self):
        self.viewer_output += bytes(self.viewer_process.readAllStandardOutput())
        marker = b"ALCHEMVIEWER_X11_WINDOW="
        if marker not in self.viewer_output or self.viewer_window is not None:
            return

        line = self.viewer_output.split(marker, 1)[1].splitlines()[0]
        try:
            native_id = int(line.decode().strip())
        except ValueError:
            return

        self.viewer_window = QWindow.fromWinId(native_id)
        container = QWidget.createWindowContainer(
            self.viewer_window,
            self.viewer_page,
        )
        self.viewer_layout.addWidget(container, 1)

    def close_project(self):
        self.project_path = None
        self.project_label.setText("No project open")
        self.status_project.setText("No project")
        self.setWindowTitle("AlchemForge")

    def change_working_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Change Working Directory", self.working_directory
        )
        if d:
            self.working_directory = d
            self.workdir_label.setText(d)
            self.status_workdir.setText(d)

    def run_workflow(self):
        QMessageBox.information(
            self,
            "Run Workflow",
            "Workflow execution will be connected to your task system.",
        )

    def add_recent(self, project):
        row = self.recent_table.rowCount()
        self.recent_table.insertRow(row)
        self.recent_table.setItem(row, 0, QTableWidgetItem(project.name))
        self.recent_table.setItem(row, 1, QTableWidgetItem(str(project.parent)))
        self.recent_table.setItem(row, 2, QTableWidgetItem("Now"))

    def show_about(self):
        QMessageBox.about(
            self,
            "About AlchemForge",
            "<b>AlchemForge</b><br><br>A graphical workbench for alchemical free-energy calculations, molecular simulation, and HPC workflows.",
        )

    def on_navigation_clicked(self, item, column):
        page = item.data(0, Qt.UserRole)

        # ------------------------------------
        # Parent menus
        # ------------------------------------

        if page == "binding_free_energy":
            item.setExpanded(not item.isExpanded())

            return

        if page in {"gromacs_workflows", "openfe_workflows"}:
            item.setExpanded(not item.isExpanded())
            return

        if page == "preparation_tools":
            item.setExpanded(not item.isExpanded())
            return

        if page in {
            "ligand_preparation",
            "openfe_ligand_preparation",
            "gromacs_ligand_preparation",
        }:
            item.setExpanded(not item.isExpanded())
            return

        if page in {"system_preparation", "energy_minimization"}:
            item.setExpanded(not item.isExpanded())
            return

        if page == "analysis":
            item.setExpanded(not item.isExpanded())
            return

        # ------------------------------------
        # Workflow pages
        # ------------------------------------

        if page == "home":
            self.center_stack.setCurrentWidget(self.home_page)

        elif page == "res_alchem_fep":
            dialog = ResAlchemFEPDialog(
                working_directory=self.working_directory, parent=self
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "charge_sdf_openfe":
            dialog = OpenFEChargeMoleculesDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "prepare_gromacs_ligands":
            dialog = GromacsLigandPrepareDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "execute_openfe_transformations":
            dialog = OpenFEExecuteDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "ligand_network_planning":
            dialog = LigandNetworkPlannerDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            show_modeless(self, dialog)

        elif page == "fix_pdb_pdbfixer":
            dialog = PDBFixerDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "structure_viewer":
            self.open_structure_viewer()

        elif page == "prepare_gromacs_system":
            dialog = GromacsSystemPrepareDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "parameterize_pdb_sdf":
            dialog = OpenMMMinimizeDialog(
                working_directory=self.working_directory,
                parent=self,
                parameterization_mode=True,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "minimize_openmm":
            dialog = OpenMMMinimizeDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "minimize_gromacs":
            dialog = GromacsMinimizeDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "abfe_gromacs":
            dialog = ABFEGromacsDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "abfe_openfe":
            dialog = OpenFEABFEDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "lig_alchem_fep":
            dialog = OpenFERBFEDialog(
                working_directory=self.working_directory, parent=self
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "septop_rbfe_openfe":
            dialog = OpenFESepTopDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "lig_rbfe_gromacs":
            dialog = LigandRBFEGromacsDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "septop_rbfe_gromacs":
            dialog = SepTopGromacsDialog(
                working_directory=self.working_directory,
                parent=self,
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "res_rbfe_gromacs_analysis":
            dialog = ResFEPGromacsAnalysisDialog(
                working_directory=self.working_directory, parent=self
            )
            dialog.job_requested.connect(self.start_backend_job)
            show_modeless(self, dialog)

        elif page == "jobs":
            self.center_stack.setCurrentWidget(self.jobs_page)
            self.refresh_jobs_table()

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("AlchemForge")
    app.setFont(QFont("Arial", 10))
    app.setStyleSheet(STYLE)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
