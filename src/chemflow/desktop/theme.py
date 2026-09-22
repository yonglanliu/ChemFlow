"""Visual system for the ChemFlow desktop."""

COLORS = {
    "canvas": "#07111f",
    "surface": "#0d1a2b",
    "surface_alt": "#122238",
    "border": "#203652",
    "text": "#edf6ff",
    "muted": "#8ea5bd",
    "accent": "#43e0c2",
    "accent_blue": "#6da8ff",
    "danger": "#ff6b7a",
}


def stylesheet() -> str:
    return f"""
    * {{
        font-family: Inter, SF Pro Display, Segoe UI, sans-serif;
        font-size: 13px;
        color: {COLORS['text']};
    }}
    QMainWindow, QWidget#Root {{ background: {COLORS['canvas']}; }}
    QFrame#Sidebar {{
        background: #091625;
        border-right: 1px solid {COLORS['border']};
    }}
    QLabel#Brand {{ font-size: 21px; font-weight: 700; letter-spacing: 1px; }}
    QLabel#BrandMark {{
        background: {COLORS['accent']}; color: #06141c; border-radius: 12px;
        font-size: 17px; font-weight: 800; min-width: 42px; min-height: 42px;
    }}
    QPushButton#HelpButton, QToolButton#HelpButton {{
        background: #10283a; color: {COLORS['accent']};
        border: 1px solid #24566a; border-radius: 17px;
        padding: 0; font-size: 17px; font-weight: 800;
    }}
    QPushButton#HelpButton:hover {{
        background: #17384c; border-color: {COLORS['accent']};
    }}
    QDialog#HelpDialog {{ background: {COLORS['canvas']}; }}
    QTextBrowser#HelpBrowser {{
        background: {COLORS['surface']}; color: {COLORS['text']};
        border: 1px solid {COLORS['border']}; border-radius: 10px;
        padding: 18px; selection-background-color: #245f6c;
    }}
    QLabel#Eyebrow {{
        color: {COLORS['accent']}; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px;
    }}
    QLabel#PageTitle {{ font-size: 27px; font-weight: 700; }}
    QLabel#PageSubtitle, QLabel#Muted {{ color: {COLORS['muted']}; }}
    QLabel#SelectionBanner {{
        background: #10283a; color: {COLORS['accent']};
        border: 1px solid #24566a; border-radius: 9px;
        padding: 10px 14px; font-size: 11px; font-weight: 800;
        letter-spacing: 1.2px;
    }}
    QPushButton#NavButton {{
        background: transparent; border: 0; border-radius: 10px;
        color: {COLORS['muted']}; text-align: left; padding: 11px 15px;
        font-weight: 600;
    }}
    QPushButton#NavButton:hover {{ background: #11243a; color: {COLORS['text']}; }}
    QPushButton#NavButton:checked {{
        background: #153149; color: {COLORS['accent']};
        border-left: 3px solid {COLORS['accent']};
    }}
    QFrame#PropertyCategory {{
        background: #0d1d30; border: 1px solid {COLORS['border']};
        border-radius: 12px;
    }}
    QLabel#PropertyCategoryTitle {{
        color: {COLORS['accent_blue']}; font-size: 11px; font-weight: 800;
        letter-spacing: 1.4px; border: 0; background: transparent;
    }}
    QFrame#PropertyCategory QLabel#Muted {{
        border: 0; background: transparent;
    }}
    QPushButton#SelectAllButton {{
        background: transparent; color: {COLORS['accent']};
        border: 1px solid #24566a; text-align: left;
    }}
    QPushButton#PropertyButton {{
        background: #081522; color: {COLORS['muted']};
        border: 1px solid {COLORS['border']}; border-radius: 8px;
        padding: 10px 11px; text-align: left; font-weight: 600;
    }}
    QPushButton#PropertyButton:hover {{
        border-color: {COLORS['accent_blue']}; color: {COLORS['text']};
    }}
    QPushButton#PropertyButton:checked {{
        background: #153149; color: {COLORS['text']};
        border-color: {COLORS['accent']};
    }}
    QFrame#Card {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
        border-radius: 14px;
    }}
    QFrame#AccentCard {{
        background: #10283a; border: 1px solid #24566a; border-radius: 14px;
    }}
    QFrame#MoleculePreviewPanel {{
        background: #f7fbff; border: 1px solid #42617c; border-radius: 10px;
    }}
    QLabel#MoleculePreviewTitle {{
        background: transparent; border: 0; color: #536a80;
        font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
    }}
    QSvgWidget#MoleculePreview {{
        background: #f7fbff; border: 0;
    }}
    QLabel#CardTitle {{ font-size: 15px; font-weight: 700; }}
    QLabel#Metric {{ font-size: 27px; font-weight: 700; color: {COLORS['accent']}; }}
    QPushButton {{
        background: {COLORS['surface_alt']}; border: 1px solid {COLORS['border']};
        border-radius: 9px; padding: 9px 14px; font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {COLORS['accent_blue']}; background: #182d47; }}
    QPushButton#Primary {{
        background: {COLORS['accent']}; border: 0; color: #06141c;
    }}
    QPushButton#Primary:hover {{ background: #67efd5; }}
    QPushButton#Danger {{ color: {COLORS['danger']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
        background: #081522; border: 1px solid {COLORS['border']};
        border-radius: 8px; padding: 8px; selection-background-color: #245f6c;
    }}
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border-color: {COLORS['accent']};
    }}
    QComboBox::drop-down {{ border: 0; width: 24px; }}
    QComboBox:on {{
        border-color: {COLORS['accent']};
        background: #0b1b2c;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['surface_alt']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 4px;
        outline: 0;
        selection-background-color: #245f6c;
        selection-color: #ffffff;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 4px 8px;
        background-color: {COLORS['surface_alt']};
        color: {COLORS['text']};
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: #1a4053;
        color: #ffffff;
    }}
    QComboBox QAbstractItemView::item:selected {{
        background-color: #245f6c;
        color: #ffffff;
    }}
    QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 10px; }}
    QTabBar::tab {{
        background: transparent; color: {COLORS['muted']}; padding: 10px 18px;
    }}
    QTabBar::tab:selected {{ color: {COLORS['accent']}; border-bottom: 2px solid {COLORS['accent']}; }}
    QTableWidget {{
        background: #081522; alternate-background-color: #0b1b2c;
        border: 1px solid {COLORS['border']}; border-radius: 8px;
        gridline-color: #1a3048; selection-background-color: #245f6c;
    }}
    QHeaderView::section {{
        background: {COLORS['surface_alt']}; color: {COLORS['text']};
        border: 0; border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']}; padding: 8px;
        font-weight: 700;
    }}
    QScrollBar:vertical {{ background: transparent; width: 8px; }}
    QScrollBar::handle:vertical {{ background: #29425d; border-radius: 4px; min-height: 30px; }}
    QStatusBar {{ background: #081522; color: {COLORS['muted']}; }}
    """
