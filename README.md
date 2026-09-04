# Detección de caídas

Este proyecto detecta caídas usando OpenCV, Ultralytics YOLO (pose) y MediaPipe (detección de manos para señal de auxilio).

## Requisitos

- Python 3.12 o superior
- [uv](https://docs.astral.sh/uv/) instalado
- Cámara disponible en la computadora
- Git instalado

---

## 1. Instalar Git y clonar el repositorio

### Opción A — Por línea de comandos

**Windows (PowerShell):**
```powershell
winget install --id Git.Git -e
```
Cierra y abre PowerShell de nuevo, luego:
```powershell
git clone https://github.com/<tu-usuario>/Deteccion-de-Caidas.git
cd Deteccion-de-Caidas
```

**macOS:**
```bash
brew install git
```
(Si no tenés Homebrew, instalalo primero desde https://brew.sh)
```bash
git clone https://github.com/<tu-usuario>/Deteccion-de-Caidas.git
cd Deteccion-de-Caidas
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install git -y
git clone https://github.com/<tu-usuario>/Deteccion-de-Caidas.git
cd Deteccion-de-Caidas
```

### Opción B — Clonar con Visual Studio Code

1. Instalá [Visual Studio Code](https://code.visualstudio.com/).
2. Instalá la extensión **Git** (ya viene integrada por defecto).
3. Abrí VS Code → `Ctrl+Shift+P` (o `Cmd+Shift+P` en macOS) → escribí `Git: Clone`.
4. Pegá la URL del repositorio: `https://github.com/<tu-usuario>/Deteccion-de-Caidas.git`.
5. Elegí una carpeta local donde guardarlo.
6. Cuando termine, hacé clic en **Open** para abrir el proyecto.
7. Abrí una terminal integrada con `` Ctrl+` `` (backtick) para seguir con los comandos de abajo.

> **Nota:** evitá clonar dentro de carpetas sincronizadas por OneDrive/Google Drive/Dropbox — a veces causan errores de permisos o archivos bloqueados durante `uv sync`.

---

## 2. Instalar uv

**Windows (PowerShell):**
```powershell
winget install --id=astral-sh.uv -e
```
O si ya tenés Python instalado:
```powershell
py -m pip install uv
```
Cerrá y abrí la terminal de nuevo después de instalar.

**macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
O con Homebrew:
```bash
brew install uv
```

**Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verificar instalación (todas las plataformas):**
```bash
uv --version
```
Salida esperada: algo como `uv 0.x.x`. Si da error de comando no encontrado, reiniciá la terminal.

---

## 3. Instalar dependencias del proyecto

Desde la carpeta raíz del proyecto (donde está `pyproject.toml`):
```bash
uv sync
```
Esto crea un entorno virtual `.venv` e instala automáticamente todas las dependencias, incluyendo OpenCV (`cv2`), Ultralytics YOLO y MediaPipe. Puede tardar varios minutos la primera vez porque descarga PyTorch.

> En macOS con chip Apple Silicon (M1/M2/M3) y en Linux, `uv sync` debería resolver automáticamente las versiones correctas de PyTorch para tu arquitectura.

---

## 4. Ejecutar el programa

Desde la raíz del proyecto:

**Windows (PowerShell o CMD):**
```powershell
uv run python -m deteccion_de_caidas.main
```

**macOS / Linux:**
```bash
uv run python -m deteccion_de_caidas.main
```

> Usamos `-m deteccion_de_caidas.main` (y no `python src\deteccion_de_caidas\main.py` directo) porque el código usa imports relativos dentro del paquete — ejecutarlo como script suelto rompe esos imports.

En macOS, la primera vez el sistema va a pedir permiso de **Cámara** para la Terminal o VS Code (System Settings → Privacy & Security → Camera). Aceptá el permiso y volvé a ejecutar el comando.

### Salir del programa

Presioná la tecla:
```
q
```

---

## 5. Actualizar el proyecto a la última versión

Desde la raíz del proyecto:
```bash
git status
git fetch origin
git pull origin master
```

### Si tenés cambios locales propios que querés conservar
Resolvé los conflictos manualmente en los archivos marcados como `both modified`, luego:
```bash
git add <archivo-resuelto>
git commit
```

### Si NO te importa perder cambios locales y solo querés la versión más reciente del remoto
```bash
git merge --abort
git fetch origin
git reset --hard origin/master
```
Después de actualizar, siempre volvé a sincronizar dependencias por si cambiaron:
```bash
uv sync
```

---

## 6. Solución de errores comunes

### `uv: command not found` / `uv no se reconoce como comando`
`uv` no quedó en el PATH o no reiniciaste la terminal. Reinstalá con los comandos de la sección 2 y **cerrá y abrí la terminal de nuevo**.

### `ImportError: attempted relative import with no known parent package`
Estás ejecutando el archivo directo (`python src/deteccion_de_caidas/main.py`) en vez de como módulo. Usá siempre:
```bash
uv run python -m deteccion_de_caidas.main
```

### `error: Pulling is not possible because you have unmerged files`
Quedó un merge sin terminar de un `git pull` anterior con conflictos. Ver la sección 5 ("Si NO te importa perder cambios locales").

### `FileNotFoundError: Unable to open file at hand_landmarker.task`
El código busca el archivo `.task` de forma relativa a la carpeta desde donde se ejecuta el comando, no desde la ubicación del script. En `signal_for_help.py`, el `model_path` debe construirse como ruta absoluta relativa al propio archivo:
```python
import os
# ...
model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
```

### `ValueError: Invalid CUDA 'device=0' requested` / `torch.cuda.is_available(): False`
El código intenta usar GPU (`device=0`), pero tu computadora no tiene GPU CUDA disponible (es normal en la mayoría de notebooks, y en todas las Mac). En `base_detector.py`, cambiá:
```python
results = self.model.track(frame, persist=True, verbose=False, device=0)
```
por:
```python
results = self.model.track(frame, persist=True, verbose=False, device="cpu")
```
Con CPU el procesamiento va a ser más lento que con GPU, pero funciona.

### La cámara no abre / ventana se abre y se cierra sola
- Verificá que ninguna otra aplicación esté usando la cámara (Zoom, Teams, otra instancia del programa, etc.).
- En Windows: Configuración → Privacidad → Cámara → permitir acceso a aplicaciones de escritorio.
- En macOS: System Settings → Privacy & Security → Camera → habilitar para Terminal/VS Code.
- En Linux: verificá que tu usuario tenga permisos sobre `/dev/video0` (grupo `video`):
  ```bash
  sudo usermod -aG video $USER
  ```
  (cerrá sesión y volvé a entrar para que tome efecto).

### `uv sync` falla o queda colgado
- Verificá tu conexión a internet (descarga PyTorch, que pesa varios cientos de MB).
- Si estás en una carpeta sincronizada por OneDrive/Drive/Dropbox, movés el proyecto a una carpeta local normal (por ejemplo `C:\Proyectos\` o `~/Proyectos/`) y volvé a intentar.
- Como último recurso, borrá el entorno y volvé a crearlo:
  ```bash
  rm -rf .venv    # macOS/Linux
  ```
  ```powershell
  Remove-Item -Recurse -Force .venv   # Windows PowerShell
  ```
  ```bash
  uv sync
  ```

---

## Notas

- El programa usa la cámara del equipo para realizar la detección en tiempo real.
- El rendimiento (FPS) depende del hardware: en CPU va a ser notablemente más lento que en GPU.
