# 🩺🔐 Image Encryption Scheme — Implementación académica

### Basado en el artículo *"An efficient and robust image encryption scheme for medical applications"*

**Autores originales:** A. Kanso y M. Ghebleh
**DOI:** 10.1016/j.cnsns.2014.12.005

---

## 📌 Descripción del proyecto

Este repositorio contiene la **implementación del esquema de cifrado y descifrado de imágenes** propuesto por **A. Kanso y M. Ghebleh** en su artículo *"An efficient and robust image encryption scheme for medical applications"*.

El trabajo ha sido desarrollado por **estudiantes de 4º curso del Grado en Ingeniería Informática del Software**, como parte de la asignatura **Criptografía**.

El objetivo principal es estudiar y reproducir el sistema de cifrado basado en **mapas caóticos**, **división por bloques**, y **enmascarado pseudoaleatorio**, orientado especialmente a la protección de **imágenes médicas** (por ejemplo TAC, resonancias, radiografías, etc.).

---

## 📂 Estructura del proyecto

* `Claves/` : Contiene las claves generadas para las 3 imágenes de prueba.
* `Imagenes/` : Contiene las imágenes originales, cifradas y descifradas.
* `Resultados/` : Contiene histogramas y análisis generados tras ejecutar los tests.
* `Scripts/` : Contiene la interfaz gráfica y scripts de ejecución.
* `src/` : Contiene toda la lógica del cifrado y descifrado.
* `Test/` : Scripts para análisis y evaluación de las métricas y la efectividad del cifrado.

---

## 🧪 Sección de Test

En la carpeta `Test/` se encuentran los scripts que realizan análisis detallados de las imágenes, incluyendo:

* Cálculo de **PSNR, MSE, correlación horizontal**.
* Evaluación de **NPCR y UACI** para verificar la sensibilidad a cambios en la imagen.
* Generación de imágenes de diferencia y binarizadas (`Diff.png`, `Diff_binary.png`).

### Ejecución de los tests

```bash
python Test/analisis_descifrado.py
```

Esto generará los análisis de las imágenes de prueba y guardará los resultados de los histogramas en `Resultados/`.



---

## 🖥 Interfaz Gráfica

La interfaz se encuentra en `Scripts/gui_app.py`. Permite cifrar y descifrar imágenes usando la generación de claves en JSON, observando los diferentes pasos de las rondas.

### Ejecución de la interfaz

1. Navegar a la carpeta `Scripts/`.
2. Ejecutar:

```bash
python gui_app.py
```

3. Se abrirá la ventana donde podrás cargar imágenes, ejecutar el cifrado/descifrado y guardar resultados.

---

## ⚙️ Preparación del entorno

1. Crear un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

3. Asegúrate de tener instaladas librerías como `numpy`, `opencv-python`, `matplotlib`, `scikit-image` y `customtkinter`.

---

## 📝 Notas adicionales

* Las imágenes generadas durante la ejecución se guardarán en las carpetas correspondientes (`Resultados/` para histogramas, `Imagenes/` para imágenes de prueba).
* Siempre activa la clave correcta antes de ejecutar análisis usando la función `activar_key` o cargando el JSON desde la interfaz.
* El tamaño de bloque por defecto es `16x16` y el número de rondas es `3`, configurable en los scripts de análisis y cifrado.

---

Este README.md sirve como guía completa para ejecutar, probar y entender el proyecto de cifrado de imágenes médicas.
