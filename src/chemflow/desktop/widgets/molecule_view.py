"""Interactive PyOpenGL molecule-inspired dashboard scene."""

from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_SRC_ALPHA,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetAttribLocation,
    glGetShaderiv,
    glGetShaderInfoLog,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    GL_COMPILE_STATUS,
)
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget


VERTEX_SHADER = """
#version 330 core
in vec3 position;
in vec3 color;
uniform mat4 mvp;
uniform float point_scale;
out vec3 v_color;
void main() {
    vec4 clip = mvp * vec4(position, 1.0);
    gl_Position = clip;
    gl_PointSize = point_scale / max(0.75, clip.w);
    v_color = color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
uniform float atoms;
void main() {
    if (atoms > 0.5) {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float radius = dot(p, p);
        if (radius > 1.0) discard;
        float light = 0.58 + 0.42 * max(0.0, dot(normalize(vec3(p, sqrt(max(0.0, 1.0-radius)))), normalize(vec3(-0.5, -0.7, 1.0))));
        frag_color = vec4(v_color * light, 1.0);
    } else {
        frag_color = vec4(v_color, 0.48);
    }
}
"""


def _perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    scale = 1.0 / math.tan(math.radians(fov) / 2.0)
    return np.asarray(
        [[scale / aspect, 0, 0, 0], [0, scale, 0, 0], [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)], [0, 0, -1, 0]],
        dtype=np.float32,
    )


def _rotation(x_angle: float, y_angle: float) -> np.ndarray:
    x, y = math.radians(x_angle), math.radians(y_angle)
    rx = np.asarray([[1, 0, 0, 0], [0, math.cos(x), -math.sin(x), 0], [0, math.sin(x), math.cos(x), 0], [0, 0, 0, 1]], dtype=np.float32)
    ry = np.asarray([[math.cos(y), 0, math.sin(y), 0], [0, 1, 0, 0], [-math.sin(y), 0, math.cos(y), 0], [0, 0, 0, 1]], dtype=np.float32)
    return ry @ rx


class MoleculeView(QOpenGLWidget):
    """Animated molecular constellation with mouse orbit and wheel zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._x_angle = -15.0
        self._y_angle = 20.0
        self._zoom = 7.2
        self._last_mouse = QPoint()
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._bond_count = 0
        self._atom_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)

    def _scene_data(self) -> tuple[np.ndarray, int, int]:
        atoms = np.asarray([
            [-1.75, 0.15, 0.10], [-0.95, 0.75, -0.05], [-0.05, 0.32, 0.18],
            [0.80, 0.88, 0.02], [1.72, 0.40, -0.16], [1.58, -0.62, 0.12],
            [0.56, -0.92, -0.08], [-0.30, -0.42, 0.10], [-1.22, -0.72, -0.14],
            [0.12, 1.42, 0.28], [2.45, 0.88, 0.20], [-2.48, 0.58, -0.22],
        ], dtype=np.float32)
        colors = np.asarray([
            [0.26, .88, .76], [.42, .66, 1.0], [.92, .96, 1.0], [.42, .66, 1.0],
            [.26, .88, .76], [.92, .96, 1.0], [.42, .66, 1.0], [.92, .96, 1.0],
            [.26, .88, .76], [1.0, .43, .50], [1.0, .75, .34], [1.0, .43, .50],
        ], dtype=np.float32)
        bonds = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,2),(7,8),(8,0),(3,9),(4,10),(0,11)]
        bond_vertices = []
        for first, second in bonds:
            bond_vertices.extend([(atoms[first], np.asarray([.33,.48,.62])), (atoms[second], np.asarray([.33,.48,.62]))])
        bond_array = np.asarray([[*position, *color] for position, color in bond_vertices], dtype=np.float32)
        atom_array = np.hstack([atoms, colors]).astype(np.float32)
        return np.vstack([bond_array, atom_array]), len(bond_array), len(atom_array)

    @staticmethod
    def _compile_shader(source: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8"))
        return shader

    def initializeGL(self) -> None:
        glClearColor(0.035, 0.075, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        vertex = self._compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = self._compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self._program = glCreateProgram()
        glAttachShader(self._program, vertex)
        glAttachShader(self._program, fragment)
        glLinkProgram(self._program)
        glDeleteShader(vertex)
        glDeleteShader(fragment)

        vertices, self._bond_count, self._atom_count = self._scene_data()
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        stride = 6 * vertices.itemsize
        position = glGetAttribLocation(self._program, "position")
        color = glGetAttribLocation(self._program, "color")
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(color)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))
        glBindVertexArray(0)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self._program)
        aspect = max(1.0, self.width()) / max(1.0, self.height())
        model = _rotation(self._x_angle, self._y_angle)
        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -self._zoom
        mvp = _perspective(42.0, aspect, 0.1, 50.0) @ view @ model
        glUniformMatrix4fv(glGetUniformLocation(self._program, "mvp"), 1, GL_FALSE, mvp.T)
        glBindVertexArray(self._vao)
        glUniform1f(glGetUniformLocation(self._program, "atoms"), 0.0)
        glDrawArrays(GL_LINES, 0, self._bond_count)
        glUniform1f(glGetUniformLocation(self._program, "atoms"), 1.0)
        glUniform1f(glGetUniformLocation(self._program, "point_scale"), 150.0)
        glDrawArrays(GL_POINTS, self._bond_count, self._atom_count)
        glBindVertexArray(0)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, width, height)

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        point = event.position().toPoint()
        delta = point - self._last_mouse
        self._x_angle += delta.y() * 0.45
        self._y_angle += delta.x() * 0.45
        self._last_mouse = point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event) -> None:
        self._zoom = min(12.0, max(4.5, self._zoom - event.angleDelta().y() / 480.0))
        self.update()

    def _animate(self) -> None:
        if not self.underMouse():
            self._y_angle += 0.16
            self.update()

    def closeEvent(self, event) -> None:
        self.makeCurrent()
        if self._program:
            glDeleteProgram(self._program)
        self.doneCurrent()
        super().closeEvent(event)
