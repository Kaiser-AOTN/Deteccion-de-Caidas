# Detección de caídas

Este proyecto detecta caídas usando OpenCV y Ultralytics YOLO.

## Requisitos

- Python 3.12 o superior
- uv instalado
- Cámara disponible en la computadora

## Instalar uv

### Windows

```powershell
winget install --id=astral-sh.uv -e
```

O si ya tienes Python:

```powershell
py -m pip install uv
```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clonar e iniciar el proyecto

```bash
git clone <url-del-repositorio>
cd <carpeta-del-proyecto>
```

## Instalar dependencias

Desde la carpeta del proyecto:

```bash
uv sync
```

## Ejecutar el programa

### Windows (PowerShell o CMD)

```powershell
cd C:\ruta\al\proyecto
uv run python .\src\deteccion_de_caidas\deteccion.py
```

### Linux

```bash
cd /ruta/al/proyecto
uv run python src/deteccion_de_caidas/deteccion.py
```

## Salir del programa

Presiona la tecla:

```text
q
```

## Solución rápida de errores

Si aparece un error de comando no encontrado:

```bash
uv --version
```

Si no responde, vuelve a instalar uv y cierra y abre la terminal.

Si falla la ejecución por dependencias:

```bash
uv sync
```

## Nota

El programa usa la cámara del equipo para realizar la detección en tiempo real.
