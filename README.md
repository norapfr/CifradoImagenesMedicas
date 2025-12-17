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

El proyecto se organiza en las siguientes carpetas, cada una con una función específica orientada a facilitar el desarrollo, las pruebas y la evaluación del sistema de cifrado:

- **`Claves/`**  
  Esta carpeta se ha creado para que los usuarios dispongan de un espacio accesible donde almacenar las claves generadas durante las pruebas realizadas a través de la interfaz gráfica. Contiene las claves correspondientes a las tres imágenes de prueba empleadas en el proyecto.

- **`Imagenes/`**  
  Incluye las imágenes originales, cifradas y descifradas utilizadas durante el desarrollo y las pruebas. Además de servir como repositorio de resultados, esta carpeta se empleó para almacenar las imágenes de prueba utilizadas en los experimentos.

- **`Resultados/`**  
  Contiene los histogramas, métricas y análisis generados tras la ejecución de los distintos tests, permitiendo evaluar el comportamiento estadístico y la efectividad del cifrado.

- **`Scripts/`**  
  Alberga la interfaz gráfica del sistema y los scripts necesarios para la ejecución del cifrado, descifrado y pruebas automatizadas.

- **`src/`**  
  Contiene toda la lógica principal del cifrado y descifrado, organizada en distintas subcarpetas según la funcionalidad implementada.  
  Para la presentación del proyecto se realizaron pruebas utilizando distintos tamaños de bloque y diferentes imágenes, con el objetivo de analizar la rapidez, eficiencia y resistencia frente a ataques diferenciales. En concreto, se trabajó con cuatro imágenes: `mri1`, `mri2`, `mri3` y `PruebaPres`, siendo esta última la empleada durante la presentación.  
  Las pruebas se realizaron con bloques de tamaño `1x1` y `16x16` para todas las imágenes, y adicionalmente con bloques de `8x8` en el caso de `PruebaPres`, con el fin de obtener una comparación más precisa de los resultados.  
  Asimismo, dentro de esta carpeta se incluyen imágenes que muestran la comparación entre la imagen cifrada original y la imagen tras aplicar un ataque diferencial. Estas comparaciones pueden observarse en los archivos `Diff` y `Diff_binary.png`, donde el resultado binarizado permite apreciar de forma más clara las diferencias.  
  Por último, también se adjuntan las imágenes intermedias correspondientes a las dos fases del algoritmo a lo largo de las tres rondas, tanto para el cifrado como para el descifrado, aunque este proceso puede visualizarse de forma más dinámica mediante la interfaz gráfica.

- **`Test/`**  
  Contiene los scripts destinados al análisis y evaluación de las métricas, así como a la validación de la efectividad del cifrado frente a distintos escenarios de prueba.

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
