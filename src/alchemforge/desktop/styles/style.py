STYLE = """
QMainWindow { background:#f5f8fb; }
QWidget { font-family:Arial; font-size:11pt; color:#17304f; }

QMenuBar {
    background:#ffffff; border-bottom:1px solid #d7e0ea; padding:2px 6px;
}
QMenuBar::item { padding:6px 10px; border-radius:4px; }
QMenuBar::item:selected { background:#eaf3fd; }

QMenu {
    background:#ffffff; border:1px solid #cbd7e3; padding:5px;
}
QMenu::item { padding:7px 32px 7px 26px; border-radius:4px; }
QMenu::item:selected { background:#e8f2fd; color:#0d5eb8; }
QMenu::separator { height:1px; background:#dfe6ee; margin:5px 8px; }

QToolBar {
    background:#ffffff; border-bottom:1px solid #d7e0ea;
    spacing:8px; padding:5px 10px;
}
QToolButton {
    background:transparent; border:none; border-radius:6px;
    padding:7px 11px; color:#17304f;
}
QToolButton:hover { background:#edf5fd; }

QListWidget#Navigation {
    background:transparent; border:none; outline:none;
}
QListWidget#Navigation::item {
    margin:2px 8px; padding:10px 12px; border-radius:7px;
}
QListWidget#Navigation::item:hover { background:rgba(232,243,253,220); }
QListWidget#Navigation::item:selected { background:#2f7ed8; color:white; }

QFrame#Card, QFrame#RightCard {
    background:#ffffff; border:1px solid #d7e2ec; border-radius:8px;
}
QLabel#SectionTitle { font-size:13pt; font-weight:700; color:#17304f; }
QLabel#SideTitle { font-size:12pt; font-weight:700; color:#17304f; }
QLabel#Muted { color:#6f85a0; }

QPushButton {
    background:#ffffff; border:1px solid #c9d7e5; border-radius:6px;
    padding:7px 12px; color:#17304f;
}
QPushButton:hover { background:#edf5fd; border-color:#8eb6de; }

QLineEdit {
    background:white; border:1px solid #cad7e5; border-radius:6px; padding:6px 9px;
}
QLineEdit:focus { border:1px solid #6ba0d4; }

QTableWidget {
    background:white; border:1px solid #d7e1eb; border-radius:6px;
    gridline-color:#e8edf3;
}
QHeaderView::section {
    background:#f2f6fa; border:none; border-bottom:1px solid #d8e1eb;
    padding:7px; color:#496581;
}

QStatusBar {
    background:#f7f9fb; border-top:1px solid #d7e0ea; color:#52687f;
}
"""
